"""Circle packing in unit square: 26 circles, maximize sum of radii.
Strategy: Grow-and-relax radius optimization combined with SLSQP on active constraints.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-9
DEADLINE = 140.0  # Safety margin from 150s limit


def _max_radius(pos, r, iters=50):
    """Maximize radii given fixed positions via Jacobi iterations."""
    n = len(r)
    pos = np.asarray(pos)
    r = np.array(r, dtype=float)
    for _ in range(iters):
        for i in range(n):
            dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(n):
                if i != j:
                    d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                    dmin = min(dmin, d - r[j])
            r[i] = max(0.0, dmin)
    return r


def _validity(circles):
    """Check if circles are valid (inside square, non-overlapping). Returns (is_valid, sum_radii)."""
    n = len(circles)
    s = 0.0
    for i in range(n):
        x, y, r = circles[i]
        if r < 0:
            return False, 0.0
        if x < r + EPS or x > 1 - r - EPS or y < r + EPS or y > 1 - r - EPS:
            return False, 0.0
        s += r
        for j in range(i + 1, n):
            x2, y2, r2 = circles[j]
            d2 = (x - x2) ** 2 + (y - y2) ** 2
            if d2 < (r + r2) ** 2 - 1e-9:
                return False, 0.0
    return True, s


def _active_pairs(pos, r, tol=1e-6):
    """Return list of (i, j) pairs that are in contact or near-contact."""
    n = len(pos)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d <= r[i] + r[j] + tol:
                pairs.append((i, j))
    return pairs


def _slsqp_objective(vars, pos, r, active_pairs, active_walls):
    """Objective for SLSQP: minimize negative sum of radii.
    vars = [x0, y0, x1, y1, ..., r0, r1, ...]
    """
    n = N
    x = vars[:2 * n].reshape(n, 2)
    rad = vars[2 * n:]
    if np.any(rad < 0):
        return 1e6
    # Wall constraints
    for i in range(n):
        if x[i, 0] < rad[i] or x[i, 0] > 1 - rad[i] or x[i, 1] < rad[i] or x[i, 1] > 1 - rad[i]:
            return 1e6
    # Pair constraints
    for (i, j) in active_pairs:
        d = np.hypot(x[i, 0] - x[j, 0], x[i, 1] - x[j, 1])
        if d < rad[i] + rad[j] - 1e-9:
            return 1e6
    return -np.sum(rad)


def _slsqp_grad(vars, pos, r, active_pairs, active_walls):
    """Gradient for SLSQP objective."""
    n = N
    x = vars[:2 * n].reshape(n, 2)
    rad = vars[2 * n:]
    grad = np.zeros_like(vars)
    
    # Gradient of -sum(rad) w.r.t. rad
    grad[2 * n:] = -1.0
    
    # Gradient of wall constraints (if active)
    for i in active_walls:
        # Wall: x[i,0] - rad[i] >= 0 => -x[i,0] + rad[i] <= 0
        # If active, add gradient
        # Actually, SLSQP handles constraints via 'ineq' or 'eq'. 
        # Here we use a penalty-like approach or explicit constraints.
        # Let's use explicit constraints in minimize.
        pass
    
    return grad


def _optimize_slsqp(pos, r, time_budget=5.0):
    """Run SLSQP optimization on active constraints."""
    t_start = time.time()
    active_pairs = _active_pairs(pos, r, tol=1e-5)
    
    # Identify active walls
    active_walls = []
    for i in range(N):
        if pos[i, 0] <= r[i] + 1e-5 or pos[i, 0] >= 1 - r[i] - 1e-5 or \
           pos[i, 1] <= r[i] + 1e-5 or pos[i, 1] >= 1 - r[i] - 1e-5:
            active_walls.append(i)
    
    # Build initial vars
    vars0 = np.zeros(4 * N)
    vars0[:2 * N] = pos.flatten()
    vars0[2 * N:] = r
    
    # Constraints for SLSQP
    constraints = []
    # Wall constraints: x[i] - r[i] >= 0 => x[i] - r[i] >= 0
    for i in range(N):
        # x[i,0] - r[i] >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[2 * i] - v[2 * N + i],
            'jac': lambda v, i=i: np.concatenate([np.array([1.0, 0.0]), np.zeros(2 * N - 1), np.array([-1.0])])
        })
        # x[i,1] - r[i] >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[2 * i + 1] - v[2 * N + i],
            'jac': lambda v, i=i: np.concatenate([np.array([0.0, 1.0]), np.zeros(2 * N - 1), np.array([-1.0])])
        })
        # (1 - x[i,0]) - r[i] >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: (1 - v[2 * i]) - v[2 * N + i],
            'jac': lambda v, i=i: np.concatenate([np.array([-1.0, 0.0]), np.zeros(2 * N - 1), np.array([-1.0])])
        })
        # (1 - x[i,1]) - r[i] >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: (1 - v[2 * i + 1]) - v[2 * N + i],
            'jac': lambda v, i=i: np.concatenate([np.array([0.0, -1.0]), np.zeros(2 * N - 1), np.array([-1.0])])
        })
    
    # Pair constraints for active pairs
    for (i, j) in active_pairs:
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i, j=j: np.hypot(v[2*i] - v[2*j], v[2*i+1] - v[2*j+1]) - v[2*N+i] - v[2*N+j],
            'jac': None  # Let SLSQP approximate
        })
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x[i]
        bounds.append((0.0, 1.0))  # y[i]
    for i in range(N):
        bounds.append((0.0, 0.5))  # r[i]
    
    try:
        res = minimize(
            _slsqp_objective,
            vars0,
            args=(pos, r, active_pairs, active_walls),
            method='SLSQP',
            jac=None,  # Let SLSQP approximate
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 50, 'ftol': 1e-8, 'disp': False}
        )
        
        if res.success and res.fun < 0:
            new_vars = res.x
            new_pos = new_vars[:2 * N].reshape(N, 2)
            new_r = new_vars[2 * N:]
            # Validate
            circles = list(zip(new_pos[:, 0], new_pos[:, 1], new_r))
            valid, score = _validity(circles)
            if valid:
                return new_pos, new_r, score
    except Exception:
        pass
    
    return pos, r, -np.sum(r)


def construct():
    t_start = time.time()
    best_score = -1.0
    best_circles = None
    
    def update_best(circles):
        nonlocal best_score, best_circles
        valid, s = _validity(circles)
        if valid and s > best_score:
            best_score = s
            best_circles = [tuple(c) for c in circles]
    
    # Generate diverse initial configurations
    def get_initializations():
        inits = []
        # 1. Grid
        for side in [5, 6, 7]:
            cells = []
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side * 0.9))
                    idx += 1
            if len(cells) >= N:
                inits.append(cells[:N])
        
        # 2. Hexagonal packing approximation
        for offset in [0.0, 0.1, 0.2]:
            pts = []
            y = 0.5 / 5
            row = 0
            while len(pts) < N and y < 1.0:
                x = 0.5 / 5 + (row % 2) * (0.5 / 5) * 0.866
                while x < 1.0 and len(pts) < N:
                    pts.append((x + offset, y + offset, 0.5 / 5 * 0.9))
                    x += 0.5 / 5
                y += 0.5 / 5 * 0.866
                row += 1
            if len(pts) >= N:
                inits.append(pts[:N])
        
        # 3. Random
        rng = np.random.default_rng(42)
        for _ in range(5):
            pts = []
            for _ in range(N):
                x = rng.random()
                y = rng.random()
                r = 0.02 + 0.05 * rng.random()
                pts.append((x, y, r))
            inits.append(pts)
            
        return inits

    inits = get_initializations()
    
    for init in inits:
        if time.time() > DEADLINE:
            break
            
        pos = np.array([(p[0], p[1]) for p in init], dtype=float)
        r = np.array([p[2] for p in init], dtype=float)
        
        # Initial radius maximization
        r = _max_radius(pos, r, iters=50)
        circles = list(zip(pos[:, 0], pos[:, 1], r))
        update_best(circles)
        
        # Grow-and-relax loop with SLSQP refinement
        for iteration in range(20):
            if time.time() > DEADLINE:
                break
            
            # Step 1: Grow radii
            r = _max_radius(pos, r, iters=20)
            circles = list(zip(pos[:, 0], pos[:, 1], r))
            update_best(circles)
            
            # Step 2: SLSQP refinement on active constraints
            if time.time() < DEADLINE - 5:
                new_pos, new_r, score = _optimize_slsqp(pos, r, time_budget=5.0)
                if score > best_score:
                    pos, r = new_pos, new_r
                    circles = list(zip(pos[:, 0], pos[:, 1], r))
                    update_best(circles)
                
                # Step 3: Relax (random perturbation to escape local optima)
                if iteration % 5 == 4:
                    rng = np.random.default_rng(hash((init[0][0], iteration)) % 2**32)
                    for i in range(N):
                        if rng.random() < 0.3:
                            pos[i, 0] = np.clip(pos[i, 0] + rng.gauss(0, 0.01), 0.0, 1.0)
                            pos[i, 1] = np.clip(pos[i, 1] + rng.gauss(0, 0.01), 0.0, 1.0)
                    r = _max_radius(pos, r, iters=10)
                    circles = list(zip(pos[:, 0], pos[:, 1], r))
                    update_best(circles)
    
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return np.array(best_circles)


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
