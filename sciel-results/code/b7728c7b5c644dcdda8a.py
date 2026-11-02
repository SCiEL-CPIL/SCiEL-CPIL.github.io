"""Rewritten program: repulsion-annealing + BCC init + gradient polish for 50-sphere packing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 120.0
MARGIN = 5.0  # seconds safety margin


def _score(pts):
    """Compute the packing radius for N points in [-0.5, 0.5]^3."""
    # Wall distance: min over all coords of (0.5 - |coord|)
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    min_wall = np.min(wall)
    # Pairwise distances
    d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return min(min_wall, min_pair)


def _neg_score(pts):
    return -_score(pts)


def _bcc_init(n):
    """Initialize n points on a BCC-like lattice within the cube."""
    # BCC: simple cubic + body center
    # Try to find a grid that gives at least n points
    # For 50 points, a 4x4x4 SC has 64, plus body centers... let's just use a structured grid
    # Actually, let's use a 3x3x3 BCC = 27*2=54 points (close to 50)
    # Or 4x4x3 = 48 SC + some body centers
    
    # Simpler: generate a BCC lattice and take first n
    pts = []
    # BCC unit cell has points at (0,0,0) and (0.5,0.5,0.5) in cell coords
    # We need to fit in [-0.5, 0.5]^3
    # Let's use a grid of BCC cells
    for gx in range(3):
        for gy in range(3):
            for gz in range(3):
                # Corner point
                cx = -0.5 + (gx + 0.5) * (1.0 / 3.0)
                cy = -0.5 + (gy + 0.5) * (1.0 / 3.0)
                cz = -0.5 + (gz + 0.5) * (1.0 / 3.0)
                pts.append([cx, cy, cz])
                # Body center
                bx = -0.5 + (gx + 1.0) * (1.0 / 3.0)
                by = -0.5 + (gy + 1.0) * (1.0 / 3.0)
                bz = -0.5 + (gz + 1.0) * (1.0 / 3.0)
                if bx <= 0.5 and by <= 0.5 and bz <= 0.5:
                    pts.append([bx, by, bz])
    
    pts = np.array(pts[:n], dtype=float)
    if len(pts) < n:
        # Fill remaining with random points
        remaining = n - len(pts)
        extra = np.random.uniform(-0.4, 0.4, size=(remaining, 3))
        pts = np.vstack([pts, extra])
    return pts


def _random_init(n, seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))


def _repulsion_anneal(pts, steps=2000, dt_start=0.01, dt_end=0.001):
    """Push spheres apart using repulsive forces with annealing."""
    pts = pts.copy().astype(float)
    n = len(pts)
    
    for step in range(steps):
        # Linearly decay step size
        t = step / max(1, steps - 1)
        dt = dt_start * (dt_end / dt_start) ** t
        
        # Compute repulsive forces
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = np.random.randn(3) * 1e-6
                    dist = np.linalg.norm(d)
                # Repulsion: stronger when close
                f = d / (dist ** 2) * 0.1
                forces[i] += f
                forces[j] -= f
        
        # Wall repulsion: push back toward center if near wall
        for i in range(n):
            for k in range(3):
                if pts[i, k] > 0.4:
                    forces[i, k] -= (pts[i, k] - 0.4) * 10.0
                elif pts[i, k] < -0.4:
                    forces[i, k] -= (pts[i, k] + 0.4) * 10.0
        
        # Update positions
        pts += dt * forces
        # Clamp to cube
        pts = np.clip(pts, -0.49, 0.49)
    
    return pts


def _gradient_optimize(pts, time_limit=10.0):
    """Use L-BFGS-B to maximize the packing radius."""
    start = time.time()
    
    def objective(x):
        return _neg_score(x.reshape(-1, 3))
    
    def jac(x):
        pts = x.reshape(-1, 3)
        n = len(pts)
        # Compute gradient numerically or analytically
        # Analytical: the min is determined by either a pair or a wall
        wall = 0.5 - np.max(np.abs(pts), axis=1)
        d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        r = min(np.min(wall), min_pair)
        
        grad = np.zeros_like(x)
        # Find which constraint is active
        if np.min(wall) <= min_pair:
            # Wall constraint active
            min_idx = np.argmin(wall)
            coord = np.argmax(np.abs(pts[min_idx]))
            sign = np.sign(pts[min_idx, coord])
            if sign == 0:
                sign = 1
            grad[min_idx * 3 + coord] = -sign
        else:
            # Pair constraint active
            flat_idx = np.argmin(d)
            i, j = np.unravel_index(flat_idx, (n, n))
            if i > j:
                i, j = j, i
            dvec = pts[i] - pts[j]
            dist = np.linalg.norm(dvec)
            if dist < 1e-15:
                dvec = np.array([1e-8, 0, 0])
                dist = 1e-8
            unit = dvec / dist
            grad[i * 3:(i+1)*3] = -unit
            grad[j * 3:(j+1)*3] = unit
        
        return grad
    
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_x = x0.copy()
    best_f = objective(x0)
    
    while time.time() - start < time_limit:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B', jac=jac,
                          bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
            # Add small perturbation to escape local minima
            if res.success and time.time() - start < time_limit - 2:
                pert = np.random.randn(N * 3) * 0.01
                pert_x = np.clip(best_x + pert, -0.49, 0.49)
                if objective(pert_x) < best_f:
                    best_x = pert_x
                    best_f = objective(pert_x)
        except Exception:
            break
    
    return best_x.reshape(-1, 3)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - MARGIN
    
    best_pts = None
    best_score = -1.0
    
    # Try multiple initializations
    inits = []
    
    # 1. BCC initialization
    inits.append(_bcc_init(N))
    
    # 2. Random initializations with different seeds
    for seed in [1, 2, 3, 42, 123, 999]:
        inits.append(_random_init(N, seed=seed))
    
    # 3. Simple cubic grid (fallback)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N].astype(float))
    
    for init_pts in inits:
        if time.time() > deadline:
            break
        
        # Repulsion annealing phase
        annealed = _repulsion_anneal(init_pts, steps=1000, dt_start=0.005, dt_end=0.0005)
        
        # Evaluate
        s = _score(annealed)
        if s > best_score:
            best_score = s
            best_pts = annealed.copy()
        
        # Gradient optimization
        if time.time() > deadline - 15:
            break
        
        remaining_time = deadline - time.time() - 5
        if remaining_time < 2:
            break
        
        optimized = _gradient_optimize(annealed, time_limit=min(remaining_time, 15.0))
        s_opt = _score(optimized)
        if s_opt > best_score:
            best_score = s_opt
            best_pts = optimized.copy()
    
    # Final polish on best result
    if best_pts is not None and time.time() < deadline - 3:
        final_time = deadline - time.time() - 2
        if final_time > 2:
            polished = _gradient_optimize(best_pts, time_limit=final_time)
            s_final = _score(polished)
            if s_final > best_score:
                best_pts = polished
                best_score = s_final
    
    if best_pts is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N].astype(float)
    
    # Ensure valid output
    best_pts = np.clip(best_pts, -0.5, 0.5)
    best_pts = np.nan_to_num(best_pts, nan=0.0, posinf=0.0, neginf=0.0)
    
    return best_pts
