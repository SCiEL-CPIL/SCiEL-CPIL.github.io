import numpy as np
import time
from scipy.optimize import minimize

N = 200

def _score(c):
    """Compute the score (radius) for a given configuration c (N, 2)."""
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    # Square is [-0.5, 0.5]
    # Distance to left wall: x + 0.5
    # Distance to right wall: 0.5 - x
    # Distance to bottom wall: y + 0.5
    # Distance to top wall: 0.5 - y
    min_x = np.min(c[:, 0]) + 0.5
    max_x = 0.5 - np.max(c[:, 0])
    min_y = np.min(c[:, 1]) + 0.5
    max_y = 0.5 - np.max(c[:, 1])
    min_wall_dist = np.min([min_x, max_x, min_y, max_y])
    
    # The radius is half the min distance for pairs, but for walls it's the distance itself?
    # Task says: r = min( min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|) )
    # So pair distance contributes d/2. Wall distance contributes d.
    return np.min([min_pair_dist / 2.0, min_wall_dist])

def _soft_min_score(c, tau=50.0):
    """
    Smooth approximation of the negative score for minimization.
    Objective: minimize -score = -min(dists)
    Soft-min: -1/tau * log(sum(exp(-tau * d_k)))
    Here d_k are the "effective distances" that determine the radius.
    For pairs: d_k = |c_i - c_j| / 2
    For walls: d_k = 0.5 - |c_ik|
    """
    c = c.reshape(-1, 2)
    
    # Pairwise terms
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    # Wall terms
    # x walls
    left = c[:, 0] + 0.5
    right = 0.5 - c[:, 0]
    # y walls
    bottom = c[:, 1] + 0.5
    top = 0.5 - c[:, 1]
    
    # Combine all terms into a flat array
    # Filter out inf from diagonal
    pair_flat = pair_terms[np.triu_indices(N, k=1)]
    wall_flat = np.concatenate([left, right, bottom, top])
    all_dists = np.concatenate([pair_flat, wall_flat])
    
    # Soft min
    # To avoid overflow in exp(-tau * d), we can shift by min
    min_d = np.min(all_dists)
    # If min_d is very small, exp(-tau * (d - min_d)) is exp(0) = 1 for the min term.
    # Actually, log-sum-exp trick:
    # log(sum(exp(-tau * d))) = -tau * min_d + log(sum(exp(-tau * (d - min_d))))
    # soft_min = -1/tau * ( -tau * min_d + log(sum(exp(-tau * (d - min_d)))) )
    #          = min_d - 1/tau * log(sum(exp(-tau * (d - min_d))))
    
    # We want to MAXIMIZE soft_min, so we MINIMIZE -soft_min.
    # Let's return -soft_min for the optimizer.
    
    # Optimization for speed:
    # exp_args = -tau * (all_dists - min_d)
    # sum_exp = np.sum(np.exp(exp_args))
    # soft_min_val = min_d - (1.0/tau) * np.log(sum_exp)
    # return -soft_min_val
    
    # However, if tau is large, exp_args can be very negative, which is fine.
    # But if tau is small, it's a smooth average.
    
    # Let's just use the standard log-sum-exp for stability.
    # Note: all_dists should be positive.
    
    # Gradient is needed for L-BFGS-B.
    # grad of -soft_min w.r.t c
    # This is complex to derive analytically for the full vector.
    # Maybe use finite differences? No, L-BFGS-B is faster with analytic grad.
    # But deriving the gradient of log-sum-exp of distances is doable.
    # Let S = sum(exp(-tau * d_k))
    # f = -soft_min = -min_d + 1/tau * log(S)
    # df/dc = 1/tau * (1/S) * sum( exp(-tau * d_k) * (-tau * dd_k/dc) )
    #       = - (1/S) * sum( exp(-tau * d_k) * dd_k/dc )
    # where dd_k/dc is the gradient of the distance d_k.
    
    # This is O(N^2) per gradient eval. Acceptable.
    
    # Let's implement the function and gradient.
    
    # Precompute indices for pairs
    # We already have pair_flat. We need to map back to (i,j) for gradient.
    # This is getting complicated to vectorize perfectly.
    
    # Alternative: Use the "active set" gradient ascent for the main loop,
    # and just do a few steps of L-BFGS-B at the end with a simple objective
    # that uses the current active set?
    
    # Actually, the active set gradient ascent is quite effective on its own
    # if run for many iterations.
    # Let's stick to the active set gradient ascent for the bulk of the time,
    # and maybe a simple coordinate descent or just more gradient ascent steps at the end.
    
    # Let's refine the active set gradient ascent.
    # It doesn't need scipy.
    
    return None # Placeholder

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # We want to pack ~200 circles.
    # Hexagonal packing density is higher.
    # Let's generate a grid of potential centers and select the best 200?
    # Or just generate a hex grid and take the first 200.
    
    # Let's create a hex grid that covers the square [-0.5, 0.5]
    # Assume initial radius r0 = 0.03 (guess)
    # dx = 2*r0, dy = sqrt(3)*r0
    # We need to fill the area.
    
    # Better: Start with a square grid of 15x15 = 225 points, remove 25 randomly?
    # Or just 15x15 grid and optimize. The optimizer will push them into a better shape.
    # A hex init is better.
    
    # Let's generate a hex grid.
    # Rows: y = -0.5 + offset + i * dy
    # Cols: x = -0.5 + offset + j * dx + (i%2)*dx/2
    
    # We don't know the optimal radius yet.
    # Let's just use a uniform grid of 15x15 = 225 points, and remove the 25 most "central" or random ones?
    # No, just take the first 200 from a hex grid.
    
    # Generate hex points
    points = []
    # Let's estimate r ~ 0.035
    # dx = 0.07, dy = 0.06
    # Number of rows: 1 / 0.06 ~ 16
    # Number of cols: 1 / 0.07 ~ 14
    # 16 * 14 = 224. Close to 200.
    
    r_init = 0.035
    dx = 2 * r_init
    dy = np.sqrt(3) * r_init
    
    i = 0
    while len(points) < N + 50: # Generate a bit extra
        y = -0.5 + r_init + i * dy
        if y > 0.5:
            break
        j = 0
        while True:
            x = -0.5 + r_init + j * dx + (i % 2) * (dx / 2.0)
            if x > 0.5:
                break
            points.append([x, y])
            j += 1
        i += 1
    
    points = np.array(points[:N])
    
    # If we didn't get enough, pad with random points inside
    if len(points) < N:
        extra = np.random.uniform(-0.4, 0.4, size=(N - len(points), 2))
        points = np.vstack([points, extra])
    
    # 2. Gradient Ascent on Min-Distance
    # Active Set Method
    
    alpha = 1e-3
    tol = 1e-6
    delta = 1e-4 # Tolerance for "active" constraints
    
    for step in range(5000):
        if time.time() > deadline:
            break
            
        c = points
        # Calculate distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        min_x = c[:, 0] + 0.5
        max_x = 0.5 - c[:, 0]
        min_y = c[:, 1] + 0.5
        max_y = 0.5 - c[:, 1]
        
        # Current radius
        min_pair = np.min(dists) / 2.0
        min_wall = np.min([np.min(min_x), np.min(max_x), np.min(min_y), np.min(max_y)])
        r = np.min([min_pair, min_wall])
        
        # If r is 0, break? No, keep pushing.
        
        # Compute Gradient
        grad = np.zeros_like(c)
        
        # Pair constraints
        # Find pairs with dist < 2*r + delta
        # Note: r is the min radius. So min_pair_dist = 2*r.
        # Active if dist < 2*r + delta
        mask = dists < (2.0 * r + delta)
        # Exclude diagonal
        mask = mask & (np.arange(N)[:, None] != np.arange(N)[None, :])
        
        # For each active pair (i,j), add gradient
        # grad_i += (c_i - c_j) / dist
        # grad_j -= (c_i - c_j) / dist
        
        # Vectorized update is tricky with variable number of active constraints per point.
        # But we can loop over active pairs.
        # Number of active pairs is small (usually < 100).
        
        active_pairs = np.argwhere(mask)
        for i, j in active_pairs:
            d = dists[i, j]
            if d < 1e-9:
                continue
            vec = (c[i] - c[j]) / d
            grad[i] += vec
            grad[j] -= vec
            
        # Wall constraints
        # Active if dist < r + delta
        # Left wall: x + 0.5 < r + delta
        mask_l = min_x < (r + delta)
        grad[mask_l, 0] += 1.0
        
        # Right wall: 0.5 - x < r + delta
        mask_r = max_x < (r + delta)
        grad[mask_r, 0] -= 1.0
        
        # Bottom wall: y + 0.5 < r + delta
        mask_b = min_y < (r + delta)
        grad[mask_b, 1] += 1.0
        
        # Top wall: 0.5 - y < r + delta
        mask_t = max_y < (r + delta)
        grad[mask_t, 1] -= 1.0
        
        # Normalize gradients?
        # The gradient of the min function is a convex combination of the active gradients.
        # If we just sum them, the magnitude can be large.
        # Let's normalize each point's gradient vector to unit length if non-zero.
        # This ensures we move in the "correct" direction with step size alpha.
        
        norms = np.linalg.norm(grad, axis=1)
        mask_nonzero = norms > 1e-9
        grad[mask_nonzero] /= norms[mask_nonzero, None]
        
        # Update
        c = c + alpha * grad
        
        # Clip to square
        c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
        c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
        
        points = c
        
    # 3. Final Polish
    # Use a few steps of L-BFGS-B on the soft-min?
    # Or just run the gradient ascent with smaller alpha for more steps?
    
    # Let's do a final refinement with smaller step size
    alpha_final = 1e-5
    for step in range(1000):
        if time.time() > deadline:
            break
            
        c = points
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_x = c[:, 0] + 0.5
        max_x = 0.5 - c[:, 0]
        min_y = c[:, 1] + 0.5
        max_y = 0.5 - c[:, 1]
        
        min_pair = np.min(dists) / 2.0
        min_wall = np.min([np.min(min_x), np.min(max_x), np.min(min_y), np.min(max_y)])
        r = np.min([min_pair, min_wall])
        
        grad = np.zeros_like(c)
        
        mask = dists < (2.0 * r + delta)
        mask = mask & (np.arange(N)[:, None] != np.arange(N)[None, :])
        active_pairs = np.argwhere(mask)
        for i, j in active_pairs:
            d = dists[i, j]
            if d < 1e-9:
                continue
            vec = (c[i] - c[j]) / d
            grad[i] += vec
            grad[j] -= vec
            
        mask_l = min_x < (r + delta)
        grad[mask_l, 0] += 1.0
        mask_r = max_x < (r + delta)
        grad[mask_r, 0] -= 1.0
        mask_b = min_y < (r + delta)
        grad[mask_b, 1] += 1.0
        mask_t = max_y < (r + delta)
        grad[mask_t, 1] -= 1.0
        
        norms = np.linalg.norm(grad, axis=1)
        mask_nonzero = norms > 1e-9
        grad[mask_nonzero] /= norms[mask_nonzero, None]
        
        c = c + alpha_final * grad
        c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
        c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
        points = c
        
    return points

# Run construct once to ensure it works? No, the evaluator calls it.

if __name__ == "__main__":
    # Simple test
    res = construct()
    print("Score:", _score(res))
