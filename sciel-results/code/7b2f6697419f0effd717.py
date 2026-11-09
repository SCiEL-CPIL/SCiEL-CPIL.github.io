"""
Packing 300 equal circles in a unit circle.
Strategy: Multi-start with Simulated Annealing for global escape, followed by L-BFGS-B local refinement.
"""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_radius(points):
    """Compute the maximum radius for the given set of centers."""
    if points is None or len(points) != N:
        return 0.0
    norms = np.linalg.norm(points, axis=1)
    min_boundary = float(1.0 - np.max(norms))
    if min_boundary <= 0:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair = float(np.min(dists[:, 1])) / 2.0
    return min(min_boundary, min_pair)

def _soft_min_radius(points, beta=100.0):
    """
    Differentiable approximation of the minimum radius.
    Returns the soft-min radius.
    """
    norms = np.linalg.norm(points, axis=1)
    boundary_margins = 1.0 - norms
    
    tree = cKDTree(points)
    # Query k=2 to get self and nearest neighbor
    dists, _ = tree.query(points, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    # Soft min of pair distances
    min_pair = np.min(pair_dists)
    # Use log-sum-exp trick for stability
    exp_terms_p = np.exp(-beta * (pair_dists - min_pair))
    soft_min_pair = min_pair - np.log(np.sum(exp_terms_p)) / beta
    
    # Soft min of boundary margins
    min_boundary = np.min(boundary_margins)
    exp_terms_b = np.exp(-beta * (boundary_margins - min_boundary))
    soft_min_boundary = min_boundary - np.log(np.sum(exp_terms_b)) / beta
    
    # Soft min of the two
    m = min(soft_min_pair, soft_min_boundary)
    exp_p = np.exp(-beta * (soft_min_pair - m))
    exp_b = np.exp(-beta * (soft_min_boundary - m))
    soft_min_radius = m - np.log(exp_p + exp_b) / beta
    
    return soft_min_radius

def _objective(x, beta=100.0):
    """Objective function for minimization (negative soft-min radius)."""
    points = x.reshape(N, 2)
    return -_soft_min_radius(points, beta)

def _generate_hex_lattice(spacing, rotation=0.0, jitter=0.0):
    """Generate a hexagonal lattice pattern."""
    k = int(1.0 / spacing) + 3
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            # Rotate
            x_rot = x * np.cos(rotation) - y * np.sin(rotation)
            y_rot = x * np.sin(rotation) + y * np.cos(rotation)
            if np.hypot(x_rot, y_rot) <= 1.0 - spacing * 0.5:
                pts.append([x_rot, y_rot])
    if len(pts) < N:
        return None
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    pts = pts[idx]
    
    if jitter > 0:
        rng = np.random.default_rng(42)
        pts += rng.normal(0, jitter, pts.shape)
        
    return pts

def _generate_rings(spacing, angle_offset=0.0):
    """Generate concentric rings."""
    pts = []
    n_rings = int(1.0 / spacing) + 1
    ring_radii = np.linspace(spacing * 0.8, 1.0 - spacing * 0.5, n_rings)
    counts = []
    total = 0
    for k, r in enumerate(ring_radii):
        if k == 0 and r < 0.1:
            c = 1
        else:
            c = max(6, int(2 * np.pi * r / spacing))
        counts.append(c)
        total += c
        
    # Adjust counts to match N
    if total < N:
        deficit = N - total
        for i in range(len(counts) - 1, -1, -1):
            add = min(deficit, max(1, counts[i] // 2))
            counts[i] += add
            deficit -= add
            if deficit <= 0: break
    elif total > N:
        excess = total - N
        for i in range(len(counts) - 1, -1, -1):
            rem = min(excess, counts[i] - (1 if i == 0 and ring_radii[i] < 0.1 else 6))
            counts[i] -= rem
            excess -= rem
            if excess <= 0: break
            
    for k, r in enumerate(ring_radii):
        c = counts[k]
        if c <= 0: continue
        if c == 1 and r < 0.1:
            pts.append([0.0, 0.0])
        else:
            angles = 2 * np.pi * np.arange(c) / c + (np.pi / c) * (k % 2) + angle_offset
            pts.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
            
    pts = np.vstack(pts)
    if len(pts) != N:
        if len(pts) < N:
            extra = N - len(pts)
            angles = 2 * np.pi * np.arange(extra) / extra
            rr = 0.9
            pts = np.vstack([pts, np.column_stack([rr * np.cos(angles), rr * np.sin(angles)])])
        else:
            pts = pts[:N]
    return pts

def _simulated_annealing(points, deadline, start_temp=0.01, end_temp=0.001, max_iters=500):
    """
    Simulated Annealing to escape local minima.
    """
    current_pts = points.copy()
    current_r = _compute_radius(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    rng = np.random.default_rng(100)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
        
        # Random move
        idx = rng.integers(0, N)
        step_size = temp * 10
        ang = rng.uniform(0, 2 * np.pi)
        move = step_size * np.array([np.cos(ang), np.sin(ang)])
        
        new_pts = current_pts.copy()
        new_pts[idx] += move
        
        new_r = _compute_radius(new_pts)
        
        if new_r > current_r:
            current_pts = new_pts
            current_r = new_r
            if new_r > best_r:
                best_pts = new_pts.copy()
                best_r = new_r
        else:
            # Accept with probability exp(-(new_r - current_r) / temp)
            delta = current_r - new_r
            if delta < 0:
                delta = 0
            prob = np.exp(-delta / temp) if temp > 1e-10 else 0
            if rng.uniform() < prob:
                current_pts = new_pts
                current_r = new_r
                
    return best_pts, best_r

def _local_refine_lbfgs(points, deadline, beta=150.0):
    """
    L-BFGS-B local refinement.
    """
    x0 = points.flatten()
    
    try:
        res = minimize(
            _objective,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 2000}
        )
        if res.x is not None:
            polished = res.x.reshape(N, 2)
            r_polished = _compute_radius(polished)
            return polished, r_polished
    except Exception:
        pass
        
    return points, _compute_radius(points)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    candidates = []
    
    # 1. Generate diverse initial candidates
    # Hex lattices
    for trial in range(10):
        spacing = 0.052 + trial * 0.002
        rot = trial * 0.1
        pts = _generate_hex_lattice(spacing, rotation=rot, jitter=0.001)
        if pts is not None:
            r = _compute_radius(pts)
            candidates.append((pts, r))
            
    # Rings
    for trial in range(10):
        spacing = 0.054 + trial * 0.002
        offset = trial * 0.1
        pts = _generate_rings(spacing, angle_offset=offset)
        if pts is not None:
            r = _compute_radius(pts)
            candidates.append((pts, r))
            
    # Random
    rng = np.random.default_rng(123)
    for _ in range(5):
        pts = rng.uniform(-0.9, 0.9, (N, 2))
        # Project to inside circle
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 0.9
        pts[mask] *= 0.9 / norms[mask]
        r = _compute_radius(pts)
        candidates.append((pts, r))

    if not candidates:
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

    candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_points, best_r = candidates[0]
    
    # 2. Global Search: Simulated Annealing on top candidates
    # Allocate ~40s for SA
    sa_deadline = start_time + 40.0
    for i in range(min(5, len(candidates))):
        if time.time() > sa_deadline:
            break
        cand_pts, cand_r = candidates[i]
        sa_pts, sa_r = _simulated_annealing(cand_pts, sa_deadline, start_temp=0.005, end_temp=0.0005, max_iters=200)
        if sa_pts is not None and sa_r > best_r:
            best_points, best_r = sa_pts, sa_r
            
    # 3. Local Refinement: L-BFGS-B on best candidates
    # Allocate remaining time
    opt_deadline = deadline - 5.0
    
    # Try L-BFGS-B on the current best and a few diverse starts
    opt_starts = [best_points.copy()]
    for cand_pts, cand_r in candidates[:10]:
        if np.linalg.norm(cand_pts - best_points) > 0.05:
            opt_starts.append(cand_pts.copy())
        if len(opt_starts) >= 5:
            break
            
    for i, start_pts in enumerate(opt_starts):
        if time.time() > opt_deadline:
            break
        
        # Use a higher beta for more fidelity
        polished_pts, polished_r = _local_refine_lbfgs(start_pts, opt_deadline, beta=200.0)
        if polished_pts is not None and polished_r > best_r:
            best_points, best_r = polished_pts, polished_r

    # 4. Final Polish: Coordinate Descent
    polish_deadline = deadline - 1.0
    if time.time() < polish_deadline:
        points = best_points.copy()
        current_r = best_r
        rng = np.random.default_rng(999)
        
        for it in range(500):
            if time.time() > polish_deadline:
                break
            idx = rng.integers(0, N)
            original = points[idx].copy()
            best_local = original
            best_local_r = current_r
            
            for _ in range(5):
                dx = rng.uniform(-0.001, 0.001)
                dy = rng.uniform(-0.001, 0.001)
                cand = original + np.array([dx, dy])
                points[idx] = cand
                r_full = _compute_radius(points)
                if r_full > best_local_r + 1e-14:
                    best_local_r = r_full
                    best_local = cand.copy()
                else:
                    points[idx] = original
                    
            points[idx] = best_local
            if best_local_r > current_r + 1e-14:
                current_r = best_local_r
                best_points = points.copy()

    return best_points

# EVOLVE-BLOCK-END
