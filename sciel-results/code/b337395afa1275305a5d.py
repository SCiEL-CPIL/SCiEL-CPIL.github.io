"""
Iterative Feasibility Sphere Packer.

Strategy:
1. Initialize points on a grid.
2. Start with a small radius r.
3. Define a smooth penalty function E(x) = sum of squared violations of:
   - Pairwise distance: max(0, 2*r - dist_ij)^2
   - Boundary: max(0, r - (0.5 - |x_k|))^2
4. Use L-BFGS-B with analytic gradients to minimize E(x).
5. If min E(x) < tol, increment r and repeat.
6. Return the best feasible configuration found.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 100
TOL = 1e-8
R_STEP = 0.0005
MAX_R_ITERS = 500
OPT_MAX_ITER = 200

def _score(points):
    """Return the maximal radius for the given set of centres."""
    if len(points) != N:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Inside the cube [-0.5, 0.5]^3
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _penalty_and_grad(x, r):
    """
    Compute penalty energy and gradient for given flat x and radius r.
    Penalty: sum over pairs i<j of max(0, 2r - d_ij)^2 
             + sum over points i and coords k of max(0, r - (0.5 - |x_ik|))^2
    """
    pts = x.reshape(N, 3)
    energy = 0.0
    grad = np.zeros_like(x)
    
    # Pairwise constraints
    # We can vectorize for speed, but for N=100, a loop is fine if optimized.
    # Let's try vectorized calculation for distance matrix.
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Violations: v_ij = max(0, 2r - dists_ij)
    # Only consider upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    d_ij = dists[iu]
    v = np.maximum(0, 2*r - d_ij)
    
    # Energy contribution
    energy += np.sum(v ** 2)
    
    # Gradient contribution
    # d/dx (v^2) = 2v * dv/dx
    # dv/dx_i = - (x_i - x_j) / d_ij  (for the pair i,j)
    # This is tricky to vectorize perfectly for the full gradient without loops or sparse tricks.
    # For N=100, 4950 pairs. A loop over pairs might be acceptable if the inner loop is fast.
    # Alternatively, use the fact that grad is sparse per pair.
    
    # Let's use a loop over pairs for gradient accumulation to be safe and simple.
    # 4950 iterations per function call. With 200 iterations, that's ~1M pair ops.
    # Python loops are slow. Let's try to vectorize the gradient calculation.
    
    # grad_k = sum_{j != i} 2 * v_ij * (-(x_i - x_j)_k / d_ij)
    # Note: v_ij is 0 if d_ij >= 2r.
    
    # Create a matrix of weights W_ij = 2 * v_ij / d_ij for i != j, else 0.
    W = np.zeros((N, N))
    mask = v > 0
    # We need to map iu indices back to full matrix for W
    W[iu] = v[mask] / d_ij[mask]  # This is v/d. We need 2*v/d.
    W[iu] *= 2.0
    W = W + W.T  # Symmetric
    
    # grad_i = - W_i @ (x_i - x_j) summed over j?
    # Actually, force on i from j is proportional to (x_i - x_j) * W_ij.
    # grad_i = - sum_j W_ij * (x_i - x_j)
    # grad = - (W * x) + (W.sum(axis=1)[:, None] * x)  ? No.
    # grad_i = - sum_j W_ij x_i + sum_j W_ij x_j = -x_i * sum_j W_ij + sum_j (W_ij x_j)
    
    sum_W = W.sum(axis=1)
    Wx = W @ pts  # (W @ pts)_i = sum_j W_ij x_j
    
    grad_pts = -sum_W[:, None] * pts + Wx
    # Wait, sign check.
    # Energy = sum (2r - d)^2. d = |x_i - x_j|.
    # dE/dx_i = 2(2r-d) * d(2r-d)/dx_i = 2v * (- (x_i - x_j)/d) = -2v/d * (x_i - x_j).
    # So grad_i = - sum_j (2 v_ij / d_ij) (x_i - x_j).
    # My W_ij above was 2 v_ij / d_ij.
    # So grad_i = - sum_j W_ij (x_i - x_j) = - x_i sum_j W_ij + sum_j W_ij x_j.
    # This matches grad_pts = -sum_W[:, None] * pts + Wx.
    
    # Boundary constraints
    # For each point i, coord k:
    # margin = 0.5 - |x_ik|
    # v_b = max(0, r - margin)
    # Energy += v_b^2
    # Grad += 2 v_b * d v_b / d x_ik
    
    abs_pts = np.abs(pts)
    margins = 0.5 - abs_pts
    v_b = np.maximum(0, r - margins)
    energy += np.sum(v_b ** 2)
    
    # Gradient for boundary
    # d/dx (r - (0.5 - |x|))^2 = 2 (r - (0.5 - |x|)) * d/dx (r - 0.5 + |x|)
    # d/dx |x| = sign(x)
    # So grad = 2 v_b * sign(x)
    
    grad_b = 2 * v_b * np.sign(pts)
    # If x=0, sign is 0, which is a subgradient. L-BFGS-B should handle it or we can nudge.
    # To avoid 0 gradient at 0, we can use a small epsilon or just let it be.
    # Actually, if x=0, the constraint is active if r > 0.5. 
    # The function is V-shaped, gradient is undefined. 
    # Using sign(0)=0 is a valid subgradient.
    
    grad_pts += grad_b
    
    grad = grad_pts.flatten()
    
    return energy, grad

def _try_feasibility(points, r, max_iter=OPT_MAX_ITER):
    """
    Try to find a configuration with radius >= r.
    Returns (success, new_points, final_energy).
    """
    x0 = points.flatten()
    
    def objective(x):
        e, _ = _penalty_and_grad(x, r)
        return e
    
    def jac(x):
        _, g = _penalty_and_grad(x, r)
        return g
        
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=jac, bounds=bounds, 
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        final_energy = res.fun
        final_points = res.x.reshape(N, 3)
        # Check if actually feasible
        # The penalty is 0 if and only if all constraints are satisfied.
        # Due to numerical errors, check if energy is small.
        if final_energy < 1e-6:
            return True, final_points, final_energy
        else:
            return False, final_points, final_energy
    except Exception:
        return False, points, float('inf')

def _grid_init():
    """Generate initial grid points."""
    nx, ny, nz = 5, 5, 4
    # Spread points evenly
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess
    points = _grid_init()
    
    # Estimate initial radius from grid
    # Grid spacing is ~0.2. Radius ~0.1.
    # Let's start a bit lower to ensure feasibility.
    current_r = 0.05
    
    best_points = points.copy()
    best_r = 0.0
    
    # Calculate initial score to have a baseline
    init_score = _score(points)
    best_r = init_score
    current_r = init_score * 0.9
    
    # Iterative radius increase
    while time.time() < deadline - 10.0:
        # Try to find feasible configuration for current_r
        success, new_points, energy = _try_feasibility(points, current_r, max_iter=OPT_MAX_ITER)
        
        if success:
            # Update best
            best_points = new_points.copy()
            best_r = current_r
            points = new_points.copy()
            
            # Increase radius
            current_r += R_STEP
            
            # If we've increased a lot, maybe check actual score to be safe?
            # The penalty function ensures feasibility for current_r.
            # So _score(points) should be >= current_r.
        else:
            # Failed to find feasible config for current_r
            # Try a smaller step? Or just break?
            # If we fail, it means current_r is likely too high.
            # We can try to reduce current_r slightly and retry, or just stop.
            # Since we are increasing, if we fail, we should probably stop or back off.
            # Let's back off a bit and try again to see if it was a numerical fluke.
            current_r -= R_STEP * 0.5
            if current_r <= best_r:
                break
            
            # Try again with the backed-off radius
            # To avoid infinite loop, limit retries
            # For now, just break if we can't improve.
            break
    
    # Final polish: try to optimize the score directly for a few steps?
    # Or just return the best feasible points.
    # The points from the last successful feasibility check are good.
    
    # One more check: maybe we can do a final local optimization on the score itself
    # to squeeze out a tiny bit more?
    if time.time() < deadline - 5.0:
        def neg_score(x):
            pts = x.reshape(N, 3)
            return -_score(pts)
        
        def neg_score_grad(x):
            # Gradient of -score is complex. 
            # Let's just use numerical gradient or skip this.
            # Skipping to save time and complexity.
            pass
            
        # Actually, let's just do a few steps of penalty minimization at the current best_r + small epsilon
        # to see if we can push further.
        pass

    return best_points

# EVOLVE-BLOCK-END
