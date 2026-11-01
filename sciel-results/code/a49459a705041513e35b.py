"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

DEADLINE = 110.0  # seconds, safety margin before 120s limit


def _score(x):
    """Compute the packing radius for a flat array of 2N coordinates."""
    pts = x.reshape(N, 2)
    # Boundary constraints: min distance to boundary
    # Square is [-0.5, 0.5]^2
    dist_boundary = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    )
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    min_pair = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt(np.sum((pts[i] - pts[j])**2))
            if d < min_pair:
                min_pair = d
    min_pair = min_pair / 2.0
    
    return min(min_boundary, min_pair)


def _score_fast(x):
    """Vectorized score computation."""
    pts = x.reshape(N, 2)
    # Boundary
    dist_boundary = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    )
    min_boundary = np.min(dist_boundary)
    
    # Pairwise: use broadcasting for speed
    # This creates a N x N x 2 array, which is 200*200*2 = 80000 floats - manageable
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)


def _hex_init(n_circles):
    """Generate a hexagonal packing initial configuration."""
    # For hexagonal packing, the optimal arrangement for circles in a square
    # We'll use a hex grid pattern
    points = []
    # Hex grid spacing
    # In hex packing, rows are offset
    # Let's figure out how many rows we need
    # For n circles, roughly sqrt(n) rows and columns
    
    # Use a hexagonal lattice approach
    # Each row has either k or k+1 points
    # Start with a reasonable estimate
    
    best_points = None
    best_score = 0
    
    for row_count in range(10, 30):
        # Distribute n_circles into row_count rows
        base_per_row = n_circles // row_count
        extra = n_circles % row_count
        
        pts = []
        y_spacing = 1.0 / (row_count + 0.5)
        x_spacing = y_spacing / np.sqrt(3) * 2
        
        for i in range(row_count):
            n_in_row = base_per_row + (1 if i < extra else 0)
            y = -0.5 + (i + 0.5) * y_spacing
            if i % 2 == 0:
                # Even row
                x_start = -0.5 + x_spacing / 2
                for j in range(n_in_row):
                    x = x_start + j * x_spacing
                    if -0.49 <= x <= 0.49:
                        pts.append([x, y])
            else:
                # Odd row, offset
                x_start = -0.5 + x_spacing
                for j in range(n_in_row):
                    x = x_start + j * x_spacing
                    if -0.49 <= x <= 0.49:
                        pts.append([x, y])
        
        if len(pts) >= n_circles:
            pts = pts[:n_circles]
            pts = np.array(pts)
            s = _score_fast(pts.flatten())
            if s > best_score:
                best_score = s
                best_points = pts
    
    if best_points is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(n_circles)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:n_circles]
    
    return best_points


def _nlp_optimize(x0, max_time=30.0):
    """Optimize using SLSQP with explicit constraints."""
    x0 = np.array(x0, dtype=float).flatten()
    start_time = time.time()
    
    # Objective: maximize the minimum radius -> minimize negative radius
    # But SLSQP minimizes, so we minimize -r
    # However, we can't easily express r as a smooth function
    # Instead, use the approach of maximizing a constant subject to constraints
    # Or use a smooth surrogate
    
    # Let's use a soft-min approach for the objective
    # r = min(min_pair/2, min_boundary)
    # We want to maximize r
    # Use log-sum-exp soft-min: -r_soft = -log(sum(exp(-1/r_i * temperature)))... 
    # Actually, let's just use SLSQP with the actual constraints
    
    # Constraints:
    # For each pair (i,j): |p_i - p_j| >= 2r
    # For each point i, each boundary: distance >= r
    # But r is a variable!
    
    # Let's add r as a variable: x = [p1, p2, ..., pN, r]
    # Total variables: 2N + 1
    # Objective: maximize r -> minimize -r
    # Constraints:
    # |p_i - p_j|^2 >= (2r)^2 for all i<j  ->  |p_i - p_j|^2 - 4r^2 >= 0
    # 0.5 - p_ik >= r for k in {0,1} -> 0.5 - p_ik - r >= 0
    # p_ik + 0.5 >= r -> p_ik + 0.5 - r >= 0
    
    n_vars = 2 * N + 1
    
    def objective(v):
        return -v[-1]  # minimize -r
    
    constraints = []
    
    # Pairwise constraints: |p_i - p_j|^2 - 4r^2 >= 0
    # This is quadratic, SLSQP can handle it
    for i in range(N):
        for j in range(i + 1, N):
            def pair_con(v, i=i, j=j):
                pi = v[2*i:2*i+2]
                pj = v[2*j:2*j+2]
                r = v[-1]
                return np.sum((pi - pj)**2) - 4*r*r
            constraints.append({'type': 'ineq', 'fun': pair_con})
    
    # Boundary constraints
    for i in range(N):
        # Right boundary: 0.5 - x_i >= r
        def right_con(v, i=i):
            return 0.5 - v[2*i] - v[-1]
        constraints.append({'type': 'ineq', 'fun': right_con})
        
        # Left boundary: x_i + 0.5 >= r
        def left_con(v, i=i):
            return v[2*i] + 0.5 - v[-1]
        constraints.append({'type': 'ineq', 'fun': left_con})
        
        # Top boundary: 0.5 - y_i >= r
        def top_con(v, i=i):
            return 0.5 - v[2*i+1] - v[-1]
        constraints.append({'type': 'ineq', 'fun': top_con})
        
        # Bottom boundary: y_i + 0.5 >= r
        def bottom_con(v, i=i):
            return v[2*i+1] + 0.5 - v[-1]
        constraints.append({'type': 'ineq', 'fun': bottom_con})
    
    # Bounds: points in [-0.5, 0.5], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((-0.5, 0.5))
        bounds.append((-0.5, 0.5))
    bounds.append((0, 0.5))
    
    # Initial guess
    v0 = np.zeros(n_vars)
    v0[:2*N] = x0
    v0[-1] = _score_fast(x0)  # initial radius estimate
    
    # Run SLSQP
    result = minimize(
        objective, v0, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-12}
    )
    
    elapsed = time.time() - start_time
    if result.success and result.x is not None:
        # Extract points and check validity
        pts = result.x[:2*N].reshape(N, 2)
        # Clamp to be safe
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        s = _score_fast(pts.flatten())
        return pts, s, elapsed
    
    return x0.reshape(N, 2), _score_fast(x0), elapsed


def construct():
    start_time = time.time()
    
    # Generate initial configuration using hex packing
    x0 = _hex_init(N)
    best_pts = x0.copy()
    best_score = _score_fast(x0.flatten())
    
    # Try SLSQP optimization with time limit
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        pts, score, elapsed = _nlp_optimize(x0, max_time=min(remaining - 5, 60))
        if score > best_score:
            best_pts = pts
            best_score = score
    
    # If we have more time, try another optimization from a perturbed start
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10:
        # Perturb and re-optimize
        np.random.seed(42)
        x_perturbed = best_pts + np.random.randn(N, 2) * 0.01
        x_perturbed[:, 0] = np.clip(x_perturbed[:, 0], -0.49, 0.49)
        x_perturbed[:, 1] = np.clip(x_perturbed[:, 1], -0.49, 0.49)
        
        pts, score, elapsed = _nlp_optimize(x_perturbed, max_time=min(remaining - 5, 40))
        if score > best_score:
            best_pts = pts
            best_score = score
    
    # Final check and return
    final_score = _score_fast(best_pts.flatten())
    if final_score < best_score:
        # Something went wrong, use the previous best
        pass
    
    return best_pts
# EVOLVE-BLOCK-END
