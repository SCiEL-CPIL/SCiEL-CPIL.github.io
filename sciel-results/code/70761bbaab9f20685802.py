import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _penalty_objective(z, penalty_weight):
    x = z[0:N]
    y = z[N:2*N]
    r = z[2*N:3*N]
    
    # Boundary violations
    pen_bound = 0.0
    pen_bound += np.sum(np.maximum(0, r - x)**2)
    pen_bound += np.sum(np.maximum(0, r - (1 - x))**2)
    pen_bound += np.sum(np.maximum(0, r - y)**2)
    pen_bound += np.sum(np.maximum(0, r - (1 - y))**2)
    
    # Overlap violations (vectorized)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    sum_r = r[:, None] + r[None, :]
    overlap = np.maximum(0, sum_r - dist)
    np.fill_diagonal(overlap, 0)
    pen_overlap = np.sum(overlap**2)
    
    return -np.sum(r) + penalty_weight * (pen_bound + pen_overlap)

def _penalty_grad(z, penalty_weight):
    x = z[0:N].copy()
    y = z[N:2*N].copy()
    r = z[2*N:3*N].copy()
    
    grad = np.zeros(3 * N)
    
    # Boundary gradients
    # d/dx: -1 for r > x, +1 for r > 1-x
    grad[0:N] -= 1.0  # from -sum(r)
    grad[N:2*N] -= 1.0
    grad[2*N:3*N] -= 1.0
    
    # Boundary penalty gradients
    # For r - x > 0: d/dx = -2(r-x), d/dr = 2(r-x)
    mask_x = r - x > 0
    if np.any(mask_x):
        grad[0:N][mask_x] -= penalty_weight * 2 * (r[mask_x] - x[mask_x])
        grad[2*N:3*N][mask_x] += penalty_weight * 2 * (r[mask_x] - x[mask_x])
    
    mask_1mx = r - (1 - x) > 0
    if np.any(mask_1mx):
        grad[0:N][mask_1mx] += penalty_weight * 2 * (r[mask_1mx] - (1 - x[mask_1mx]))
        grad[2*N:3*N][mask_1mx] += penalty_weight * 2 * (r[mask_1mx] - (1 - x[mask_1mx]))
    
    mask_y = r - y > 0
    if np.any(mask_y):
        grad[N:2*N][mask_y] -= penalty_weight * 2 * (r[mask_y] - y[mask_y])
        grad[2*N:3*N][mask_y] += penalty_weight * 2 * (r[mask_y] - y[mask_y])
    
    mask_1my = r - (1 - y) > 0
    if np.any(mask_1my):
        grad[N:2*N][mask_1my] += penalty_weight * 2 * (r[mask_1my] - (1 - y[mask_1my]))
        grad[2*N:3*N][mask_1my] += penalty_weight * 2 * (r[mask_1my] - (1 - y[mask_1my]))
    
    # Overlap penalty gradients
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    sum_r = r[:, None] + r[None, :]
    overlap = np.maximum(0, sum_r - dist)
    np.fill_diagonal(overlap, 0)
    
    # Gradient of penalty w.r.t. x[i]: -2 * sum_j overlap_ij * (x[i]-x[j])/dist_ij
    # For each pair (i,j) with overlap > 0:
    #   d/dx[i] += penalty_weight * 2 * overlap_ij * (x[i]-x[j]) / dist_ij
    #   d/dx[j] -= penalty_weight * 2 * overlap_ij * (x[i]-x[j]) / dist_ij
    
    safe_dist = np.where(dist > 1e-10, dist, 1e-10)
    # overlap_grad[i,j] = 2 * overlap_ij * (x[i]-x[j]) / dist_ij
    overlap_grad_x = 2 * overlap * (dx / safe_dist)
    overlap_grad_y = 2 * overlap * (dy / safe_dist)
    overlap_grad_r = 2 * overlap  # d/dr[i] += sum_j 2*overlap_ij
    
    grad[0:N] += penalty_weight * np.sum(overlap_grad_x, axis=1)
    grad[N:2*N] += penalty_weight * np.sum(overlap_grad_y, axis=1)
    grad[2*N:3*N] += penalty_weight * np.sum(overlap_grad_r, axis=1)
    
    return grad

def _hex_initial():
    """Hexagonal lattice initialization"""
    pts = []
    # Hexagonal packing in unit square
    # Use a triangular lattice with some jitter
    n_per_row = 5
    for i in range(n_per_row):
        for j in range(6):
            x = (j + 0.5) / 6.0 + (0.5 if i % 2 == 1 else 0) / 6.0
            y = (i + 0.5) / 5.0
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            pts.append([x + np.random.uniform(-0.02, 0.02), y + np.random.uniform(-0.02, 0.02)])
            if len(pts) >= N:
                break
        if len(pts) >= N:
            break
    
    # If not enough, add random
    while len(pts) < N:
        pts.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
    
    pts = pts[:N]
    return np.array(pts)

def _grid_initial():
    """Grid initialization with perturbation"""
    side = int(np.ceil(np.sqrt(N)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= N:
                break
            x = (i + 0.5) / side + np.random.uniform(-0.03, 0.03)
            y = (j + 0.5) / side + np.random.uniform(-0.03, 0.03)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            pts.append([x, y])
    
    while len(pts) < N:
        pts.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
    
    return np.array(pts[:N])

def _voronoi_initial():
    """Voronoi-like initialization: start with grid, then relax to maximize min distance"""
    side = int(np.ceil(np.sqrt(N)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= N:
                break
            pts.append([(i + 0.5) / side, (j + 0.5) / side])
    
    while len(pts) < N:
        pts.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
    
    pts = np.array(pts[:N])
    
    # Simple relaxation: move each point away from its nearest neighbor
    for _ in range(50):
        for i in range(N):
            min_dist = np.inf
            nearest = -1
            for j in range(N):
                if i == j:
                    continue
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
                    nearest = j
            if nearest >= 0 and min_dist < 0.2:
                # Move away from nearest
                direction = pts[i] - pts[nearest]
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                pts[i] += 0.01 * direction
                pts[i] = np.clip(pts[i], 0.05, 0.95)
    
    return pts

def _bounds():
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))
    return bounds

def _enforce_validity(sol, shrink=0.999):
    """Ensure solution is strictly valid by shrinking radii if needed"""
    if sol is None:
        return None
    
    x = sol[:, 0].copy()
    y = sol[:, 1].copy()
    r = sol[:, 2].copy()
    
    # Check and fix boundary violations
    for i in range(N):
        r[i] = min(r[i], x[i] * shrink, (1 - x[i]) * shrink, y[i] * shrink, (1 - y[i]) * shrink)
    
    # Check and fix overlap violations
    for _ in range(10):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if r[i] + r[j] > dist * shrink:
                    # Scale down both radii proportionally
                    excess = (r[i] + r[j]) / dist
                    scale = shrink / excess
                    r[i] *= scale
                    r[j] *= scale
                    changed = True
        if not changed:
            break
    
    return np.column_stack([x, y, r])

def _score_valid(sol):
    """Check if solution is valid and return sum of radii, or 0 if invalid"""
    if sol is None:
        return 0.0
    
    x = sol[:, 0]
    y = sol[:, 1]
    r = sol[:, 2]
    
    tol = 1e-7
    for i in range(N):
        if r[i] > x[i] + tol or r[i] > 1 - x[i] + tol or r[i] > y[i] + tol or r[i] > 1 - y[i] + tol:
            return 0.0
        if r[i] < -tol:
            return 0.0
    
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if r[i] + r[j] > dist + tol:
                return 0.0
    
    return np.sum(r)

def _optimize_from(z0, bounds, max_time=10.0):
    """Optimize from initial point with adaptive penalty phases"""
    start = time.time()
    
    # Phase 1: High penalty to satisfy constraints
    penalty_weights = [1000.0, 500.0, 200.0, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
    
    best_z = z0.copy()
    best_obj = _penalty_objective(z0, 1000.0)
    
    for pw in penalty_weights:
        if time.time() - start > max_time:
            break
        
        try:
            res = minimize(
                _penalty_objective, best_z,
                jac=lambda z: _penalty_grad(z, pw),
                args=(pw,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_z = res.x.copy()
        except Exception:
            continue
    
    return best_z

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill
    
    best_sum = 0.0
    best_sol = None
    
    bounds = _bounds()
    
    # Initialization strategies
    inits = [_hex_initial(), _grid_initial(), _voronoi_initial()]
    
    # Add some random inits
    for _ in range(3):
        pts = np.random.uniform(0.05, 0.95, size=(N, 2))
        inits.append(pts)
    
    np.random.shuffle(inits)
    
    for init_pts in inits:
        if time.time() > deadline:
            break
        
        # Compute initial radii
        r0 = np.zeros(N)
        for i in range(N):
            min_dist = 1.0
            for j in range(N):
                if i != j:
                    d = np.linalg.norm(init_pts[i] - init_pts[j])
                    min_dist = min(min_dist, d)
            min_dist = min(min_dist, init_pts[i][0], 1 - init_pts[i][0], init_pts[i][1], 1 - init_pts[i][1])
            r0[i] = min_dist * 0.45
        
        z0 = np.concatenate([init_pts[:, 0], init_pts[:, 1], r0])
        
        # Optimize
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
        
        z_opt = _optimize_from(z0, bounds, max_time=min(remaining_time - 2, 12.0))
        
        # Convert back
        x_opt = z_opt[0:N]
        y_opt = z_opt[N:2*N]
        r_opt = z_opt[2*N:3*N]
        
        sol = np.column_stack([x_opt, y_opt, r_opt])
        
        # Enforce validity
        sol_valid = _enforce_validity(sol, shrink=0.998)
        
        score = _score_valid(sol_valid)
        if score > best_sum:
            best_sum = score
            best_sol = sol_valid
    
    # Local refinement restarts from best solution
    if best_sol is not None and time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Perturb best solution
            perturbation = np.random.normal(0, 0.02, size=(N, 2))
            x_pert = np.clip(best_sol[:, 0] + perturbation[:, 0], 0.01, 0.99)
            y_pert = np.clip(best_sol[:, 1] + perturbation[:, 1], 0.01, 0.99)
            r_pert = best_sol[:, 2] * 0.9  # Slightly shrink to allow movement
            
            z0 = np.concatenate([x_pert, y_pert, r_pert])
            
            remaining_time = deadline - time.time()
            if remaining_time < 5:
                break
            
            z_opt = _optimize_from(z0, bounds, max_time=min(remaining_time - 2, 10.0))
            
            x_opt = z_opt[0:N]
            y_opt = z_opt[N:2*N]
            r_opt = z_opt[2*N:3*N]
            
            sol = np.column_stack([x_opt, y_opt, r_opt])
            sol_valid = _enforce_validity(sol, shrink=0.998)
            
            score = _score_valid(sol_valid)
            if score > best_sum:
                best_sum = score
                best_sol = sol_valid
    
    # Fallback
    if best_sol is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_sol = np.array(cells[:N])
    
    return best_sol
