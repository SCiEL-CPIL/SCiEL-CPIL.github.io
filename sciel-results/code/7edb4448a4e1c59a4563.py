"""Packing 50 equal circles in a unit square [-1/2, 1/2]^2."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
SIDE = 1.0
HALF = 0.5
DEADLINE = 140.0  # safety margin before 150s kill

def _compute_radius(pts):
    """Compute the valid common radius for a set of points."""
    n = len(pts)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Wall distance
    wall = np.min(np.min(HALF - np.abs(pts), axis=1))
    return 0.5 * min(min_pair_dist, wall * 2)

def _objective_and_grad(x, beta):
    """Soft-min surrogate objective and gradient for maximizing min radius.
    
    We minimize -r where r is approximated by a soft-min.
    Actually, let's minimize a penalty for violations.
    
    Better approach: minimize a function that is 0 when all constraints are satisfied
    and positive otherwise, but we want to maximize the minimum radius.
    
    Let's use the approach: minimize sum of exp(-beta * (d_ij - 2r)) for all pairs
    and sum of exp(-beta * (wall_i - r)) for walls, plus a term to maximize r.
    
    Actually, the cleanest is: for a fixed r, check feasibility. But we want to optimize.
    
    Let's use: minimize F(x, r) = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, r - wall_i)^2
    This is not smooth. Use soft version.
    
    Alternative: minimize -log(sum of exp terms) ... 
    
    Let me use the standard approach for packing:
    Minimize: -r + penalty
    where penalty = sum_{i<j} exp(-beta*(d_ij - 2r)) / (beta * d_ij) type terms
    
    Actually, simplest smooth approach:
    For each pair (i,j), define gap_ij = d_ij - 2r
    For each point i, wall gap = (HALF - |x_i|) + (HALF - |y_i|) ... no, min(HALF-|x|, HALF-|y|)
    
    Let's define:
    f(x, r) = -r + (1/beta) * [ sum_{i<j} log(1 + exp(-beta * (d_ij - 2r))) + sum_i log(1 + exp(-beta * (w_i - r))) ]
    where w_i = min(HALF - |x_i|, HALF - |y_i|)
    
    This is smooth. As beta -> inf, it approaches -r + max(0, 2r - d_ij) + max(0, r - w_i)
    
    Gradient w.r.t. x:
    d/dx_k of log(1+exp(-beta*(d_ij - 2r))) = -beta * exp(-beta*(d_ij-2r))/(1+exp(-beta*(d_ij-2r))) * d/dx_k (d_ij - 2r)
    = -beta * sigmoid(-beta*(d_ij-2r)) * (d/dx_k d_ij)
    where sigmoid(z) = 1/(1+exp(-z))
    
    d/dx_k d_ij = (x_k_i - x_k_j) / d_ij for k in {0,1}
    
    d/dx_k of log(1+exp(-beta*(w_i - r))) = -beta * sigmoid(-beta*(w_i-r)) * d/dx_k w_i
    d/dx_k w_i: w_i = min(HALF-|x_i|, HALF-|y_i|). This is non-smooth. 
    Approximate: use soft-min for walls too.
    
    Let w_i = soft_min(HALF - x_i, HALF + x_i, HALF - y_i, HALF + y_i) with a small beta_w.
    
    This is getting complex. Let me use a simpler but effective approach.
    """
    pass

def _solve_with_lbgfgs(initial_points, beta_schedule, time_limit):
    """Use L-BFGS-B with soft-min surrogate to maximize radius."""
    x0 = initial_points.flatten()
    n = N
    
    def obj_and_grad(x):
        pts = x.reshape(-1, 2)
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-16)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall_dists = np.min(HALF - np.abs(pts), axis=1)
        
        # Use a variable r that we also optimize
        # Actually, let's fix the structure: we optimize positions, and the radius is determined.
        # To make it smooth, use log-sum-exp.
        
        # The radius is min(min_pair_dist/2, min_wall_dist)
        # Soft version: -log-sum-exp(-beta * values) / beta
        # where values are [d_ij/2 for all pairs, wall_i for all i]
        
        pair_vals = dists[~np.eye(n, dtype=bool)] / 2
        wall_vals = wall_dists
        all_vals = np.concatenate([pair_vals, wall_vals])
        
        # Soft min: -1/beta * log(sum(exp(-beta * vals)))
        # For numerical stability
        m = np.min(all_vals)
        soft_min = -1.0 / beta * np.log(np.sum(np.exp(-beta * (all_vals - m)))) - m
        
        # Gradient
        # d/dx_k soft_min = -1/beta * d/dx_k [-log(sum(exp(-beta*vals)))]
        # = 1/beta * sum_j exp(-beta*val_j) * (-beta * d/dx_k val_j) / sum_j exp(-beta*val_j)
        # = -sum_j exp(-beta*val_j) * d/dx_k val_j / sum_j exp(-beta*val_j)
        
        exp_terms = np.exp(-beta * (all_vals - m))
        weights = exp_terms / np.sum(exp_terms)
        
        grad = np.zeros_like(x)
        
        # Pair contributions
        pair_idx = 0
        for i in range(n):
            for j in range(i+1, n):
                w = weights[pair_idx]
                # val = dists[i,j]/2
                # d/dx_k (dists[i,j]/2) = (pts[i,k] - pts[j,k]) / (2 * dists[i,j])
                if dists[i, j] > 1e-10:
                    grad[2*i] += -w * (pts[i, 0] - pts[j, 0]) / dists[i, j]
                    grad[2*i+1] += -w * (pts[i, 1] - pts[j, 1]) / dists[i, j]
                    grad[2*j] += -w * (pts[j, 0] - pts[i, 0]) / dists[i, j]
                    grad[2*j+1] += -w * (pts[j, 1] - pts[i, 1]) / dists[i, j]
                pair_idx += 1
        
        # Wall contributions
        for i in range(n):
            w = weights[pair_idx + i]
            # wall = min(HALF-|x|, HALF-|y|)
            # Use soft min of the 4 wall distances
            # For simplicity, approximate gradient
            hw_x = HALF - abs(pts[i, 0])
            hw_y = HALF - abs(pts[i, 1])
            if hw_x < hw_y:
                # wall = HALF - |x|
                if pts[i, 0] >= 0:
                    grad[2*i] += -w * 1.0
                else:
                    grad[2*i] += -w * (-1.0)
            else:
                if pts[i, 1] >= 0:
                    grad[2*i+1] += -w * 1.0
                else:
                    grad[2*i+1] += -w * (-1.0)
        
        return soft_min, grad
    
    # Multi-stage beta optimization
    best_x = x0.copy()
    best_r = _compute_radius(initial_points)
    
    for beta in beta_schedule:
        # Use previous best as initial
        x_start = best_x.copy()
        
        result = minimize(
            lambda x: obj_and_grad(x)[0],
            x_start,
            jac=lambda x: obj_and_grad(x)[1],
            method='L-BFGS-B',
            bounds=[(-HALF, HALF)] * (2*n),
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 50000}
        )
        
        if result.success or result.fun < obj_and_grad(x_start)[0]:
            new_pts = result.x.reshape(-1, 2)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_x = result.x.copy()
    
    return best_x.reshape(-1, 2), best_r

def _construct_hexagonal():
    """Create a hexagonal packing initial guess."""
    # Hexagonal lattice: rows offset
    # For 50 circles, roughly 7 rows of 7-8
    points = []
    # Try different row configurations
    best_points = None
    best_r = 0
    
    for rows in range(5, 10):
        # Distribute 50 into rows
        base = N // rows
        extra = N % rows
        row_counts = [base + (1 if i < extra else 0) for i in range(rows)]
        
        pts = []
        for i, count in enumerate(row_counts):
            y = (i + 0.5) / rows - 0.5
            if i % 2 == 0:
                x_start = -HALF + 0.5 / count
                x_step = 1.0 / count
            else:
                x_start = -HALF + 1.0 / count
                x_step = 1.0 / count
            
            for j in range(count):
                x = x_start + j * x_step
                pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) == N:
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best_points = pts
    
    if best_points is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
        best_r = _compute_radius(best_points)
    
    return best_points, best_r

def construct():
    start_time = time.time()
    
    # Start with hexagonal packing
    init_pts, init_r = _construct_hexagonal()
    
    # Also try a few random initializations
    rng = np.random.default_rng(42)
    candidates = [(init_pts, init_r)]
    
    for trial in range(5):
        if time.time() - start_time > DEADLINE - 60:
            break
        # Random initialization with some structure
        pts = rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))
        r = _compute_radius(pts)
        candidates.append((pts, r))
    
    # Sort by initial radius
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_pts = candidates[0][0]
    best_r = candidates[0][1]
    
    # Try L-BFGS-B optimization on top candidates
    beta_schedules = [
        [100, 300, 1000, 3000],
        [50, 150, 500, 1500],
    ]
    
    for idx, (pts, r) in enumerate(candidates[:3]):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        try:
            opt_pts, opt_r = _solve_with_lbgfgs(pts, beta_schedules[idx % len(beta_schedules)], time.time() - start_time)
            if opt_r > best_r:
                best_r = opt_r
                best_pts = opt_pts
        except Exception:
            continue
    
    # Final check: ensure all points are within bounds
    best_pts = np.clip(best_pts, -HALF + 1e-10, HALF - 1e-10)
    
    # Verify radius
    final_r = _compute_radius(best_pts)
    
    return best_pts

# Test
if __name__ == "__main__":
    import time
    start = time.time()
    pts = construct()
    elapsed = time.time() - start
    r = _compute_radius(pts)
    print(f"Radius: {r:.15f}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Shape: {pts.shape}")
    print(f"Min coord: {np.min(pts):.6f}, Max coord: {np.max(pts):.6f}")
