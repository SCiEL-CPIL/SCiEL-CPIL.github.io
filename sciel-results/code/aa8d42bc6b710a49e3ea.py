import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_score(positions):
    """Compute the radius r for a given set of positions."""
    if len(positions) != N:
        return 0.0
    if not np.all(np.isfinite(positions)):
        return 0.0
    
    # Check if inside unit circle
    dists_center = np.linalg.norm(positions, axis=1)
    if np.any(dists_center > 1.0):
        return 0.0
        
    # Calculate minimum distance between pairs
    min_pair_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_pair_dist:
                min_pair_dist = d
                
    r_pair = min_pair_dist / 2.0
    r_boundary = 1.0 - np.max(dists_center)
    
    return min(r_pair, r_boundary)

def _objective_and_grad(x, N):
    """
    Objective function and gradient for L-BFGS-B.
    x: [x_1, y_1, ..., x_N, y_N, r]
    Minimizes -r + lambda * Penalty
    """
    positions = x[:2*N].reshape(N, 2)
    r = x[2*N]
    
    # Calculate distances
    # Pairwise distances
    min_pair_dist = np.inf
    # We need gradients, so we compute terms directly
    
    grad = np.zeros_like(x)
    obj = -r
    
    lambda_pen = 1000.0
    
    # Pairwise constraints: |c_i - c_j| >= 2r
    for i in range(N):
        for j in range(i+1, N):
            d_vec = positions[i] - positions[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-10:
                d = 1e-10
                d_vec = np.array([1e-10, 0])
            
            violation = 2*r - d
            if violation > 0:
                # Penalty: violation^2
                # Grad wrt positions[i]: 2 * violation * (d_vec / d)
                # Grad wrt positions[j]: 2 * violation * (-d_vec / d)
                # Grad wrt r: 2 * violation * 2 = 4 * violation
                
                factor = 2 * violation / d
                grad[2*i] += lambda_pen * factor * d_vec[0]
                grad[2*i+1] += lambda_pen * factor * d_vec[1]
                grad[2*j] -= lambda_pen * factor * d_vec[0]
                grad[2*j+1] -= lambda_pen * factor * d_vec[1]
                
                grad[2*N] += lambda_pen * 4 * violation
                
                obj += lambda_pen * violation * violation
            else:
                # No violation, but we might want a small push to expand?
                # No, the -r term handles expansion.
                pass
                
    # Boundary constraints: |c_i| <= 1 - r
    for i in range(N):
        c = positions[i]
        d = np.linalg.norm(c)
        if d < 1e-10:
            d = 1e-10
            c_norm = np.array([1.0, 0.0])
        else:
            c_norm = c / d
            
        limit = 1.0 - r
        violation = d - limit
        if violation > 0:
            # Penalty: violation^2
            # Grad wrt c_i: 2 * violation * c_norm
            # Grad wrt r: 2 * violation * 1 (since limit = 1-r, d-limit = d-1+r, deriv wrt r is 1)
            
            factor = 2 * violation
            grad[2*i] += lambda_pen * factor * c_norm[0]
            grad[2*i+1] += lambda_pen * factor * c_norm[1]
            
            grad[2*N] += lambda_pen * factor
            
            obj += lambda_pen * violation * violation
            
    return obj, grad

def _run_optimizer(initial_positions, r_init, time_limit):
    N_local = N
    x0 = np.zeros(2*N_local + 1)
    x0[:2*N_local] = initial_positions.flatten()
    x0[2*N_local] = r_init
    
    bounds = [(-1.0, 1.0)] * (2*N_local) + [(0.0, 0.15)]
    
    start_time = time.time()
    
    # Use L-BFGS-B
    # We want to maximize r, so we minimize -r + penalty.
    # The optimizer will try to increase r until penalties kick in.
    
    try:
        res = minimize(
            _objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
        )
        return res.x
    except Exception:
        return x0

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    best_positions = None
    best_score = -1.0
    
    # Initialization 1: Hexagonal Lattice
    def init_hex():
        pts = []
        # Generate a dense hex grid and pick best N
        spacing = 0.2
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                # Sort by distance from center to keep central dense part
                idx = np.argsort(radius)
                return points[idx[:N]]
            spacing *= 0.95
            if spacing < 0.1:
                break
        # Fallback random
        return np.random.uniform(-0.5, 0.5, (N, 2))

    # Initialization 2: Ring Structure
    def init_rings():
        pts = np.zeros((N, 2))
        r_est = 0.125
        # 1, 6, 12, 18, 13
        counts = [1, 6, 12, 18, 13]
        radii = [0, 2*r_est, 2*r_est*np.sqrt(3), 4*r_est, 2*r_est*np.sqrt(7)]
        
        idx = 0
        for ring_idx, count in enumerate(counts):
            r_ring = radii[ring_idx]
            for k in range(count):
                if ring_idx == 0:
                    angle = 0
                else:
                    angle = 2 * np.pi * k / count
                pts[idx, 0] = r_ring * np.cos(angle)
                pts[idx, 1] = r_ring * np.sin(angle)
                idx += 1
        return pts

    # Initialization 3: Random
    def init_random():
        return np.random.uniform(-0.8, 0.8, (N, 2))

    inits = [init_hex(), init_rings(), init_random()]
    
    for i, init_pos in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Run optimizer
        # Start with a small r to let it grow
        r_start = 0.05
        result_x = _run_optimizer(init_pos, r_start, deadline)
        
        if result_x is None:
            continue
            
        pos = result_x[:2*N].reshape(N, 2)
        r_val = result_x[2*N]
        
        # Validate and score
        score = _compute_score(pos)
        
        if score > best_score:
            best_score = score
            best_positions = pos
            
        # Check time
        if time.time() > deadline - 20:
             break

    if best_positions is None:
        # Fallback to seed behavior
        return init_hex()
        
    return best_positions
