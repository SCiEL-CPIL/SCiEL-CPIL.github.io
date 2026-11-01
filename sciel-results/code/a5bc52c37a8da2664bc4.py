"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # distance to walls
    wall = np.min(np.min(0.5 - np.abs(c), axis=1), axis=0)
    # pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(wall, min_dist / 2.0)

def _softmin_objective(flat_c, beta=200.0):
    """
    Soft-min objective to be minimized.
    We want to maximize the minimum distance (radius).
    So we minimize the negative of a soft-min of the distances.
    
    The soft-min of a set of values x_i is approximated by:
    -1/beta * log(sum(exp(-beta * x_i)))
    
    Here x_i are the constraints:
    1. Distances between spheres (d_ij)
    2. Distances to walls (w_i)
    
    We want all these to be large.
    So we define the objective as:
    sum(exp(-beta * d_ij)) + sum(exp(-beta * w_i))
    
    Minimizing this sum encourages all distances to be large.
    This is a smooth approximation to the max-min problem.
    """
    c = flat_c.reshape(N, 3)
    
    # Wall distances
    # w_i = min_k (0.5 - |c_ik|)
    # To make it smooth, we can use a soft-min for the wall distances too, 
    # or just use the exact min if we handle gradients carefully.
    # For Nelder-Mead, we don't need gradients, so we can use exact min for walls 
    # but it's non-differentiable. 
    # However, Nelder-Mead is derivative-free, so it can handle non-smoothness 
    # reasonably well, but a smooth objective is better.
    
    # Let's use a soft-min for the wall distances as well to keep the whole thing smooth.
    # Wall distance for sphere i: min over 6 faces.
    # Face distances: 0.5 - c_i0, 0.5 + c_i0, 0.5 - c_i1, 0.5 + c_i1, 0.5 - c_i2, 0.5 + c_i2
    # Let's compute these 6N values.
    
    wall_dists = np.empty((N, 6))
    wall_dists[:, 0] = 0.5 - c[:, 0]
    wall_dists[:, 1] = 0.5 + c[:, 0]
    wall_dists[:, 2] = 0.5 - c[:, 1]
    wall_dists[:, 3] = 0.5 + c[:, 1]
    wall_dists[:, 4] = 0.5 - c[:, 2]
    wall_dists[:, 5] = 0.5 + c[:, 2]
    
    # Pairwise distances
    # We need to be careful with memory for N=100. 100*100 = 10000 distances. That's fine.
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Avoid division by zero or sqrt of negative (shouldn't happen)
    # We only care about i < j
    # To save memory and time, we can compute the upper triangle
    # But for simplicity and vectorization, let's compute all and mask diagonal
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    
    # Get all pair distances (upper triangle)
    # Using triu_indices
    iu, ju = np.triu_indices(N, k=1)
    pair_dists = dist[iu, ju]
    
    # Combine all distances: 6N wall dists and N(N-1)/2 pair dists
    all_dists = np.concatenate([wall_dists.flatten(), pair_dists])
    
    # Soft-min approximation: sum(exp(-beta * d))
    # To avoid overflow/underflow issues with exp, we can shift by min or just use it directly if beta is moderate.
    # With beta=200, exp(-200 * 0.1) ~ exp(-20) ~ 2e-9. exp(-200 * 0.01) ~ exp(-2) ~ 0.135.
    # It works.
    
    obj = np.sum(np.exp(-beta * all_dists))
    
    # Add a small penalty if any distance is negative (shouldn't happen if inside cube, but just in case)
    # Actually, if a sphere is outside, wall dist is negative. exp(-beta * neg) = exp(pos) -> huge.
    # This naturally penalizes being outside.
    
    return obj

def _init_grid():
    """Simple cubic grid as starting point."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _init_bcc():
    """Body-centered cubic lattice initialization."""
    # BCC has 2 points per unit cell.
    # We need to fit 100 points.
    # Let's try a grid of cells.
    # If we use a 4x4x4 grid of BCC cells, we have 4*4*4*2 = 128 points. Too many.
    # 3x4x4 = 48 cells * 2 = 96 points. Close.
    # 4x4x3 = 48 cells * 2 = 96 points.
    # Let's generate 96 points and add 4 random ones? Or just use 4x4x4 and remove 28?
    # Better: Generate a dense grid and take the best 100?
    # Or just use the standard cubic grid which is simpler and works well enough as a start.
    # Let's stick to a slightly perturbed grid to break symmetry.
    return _init_grid()

def _optimize_from_initial(c_init, time_limit, beta=200.0):
    """
    Run Nelder-Mead optimization from a given initial configuration.
    """
    c_flat = c_init.flatten()
    
    # Options for Nelder-Mead
    options = {
        'xatol': 1e-10,
        'fatol': 1e-12,
        'maxiter': 5000,  # High limit, will be cut by time
        'maxfev': 100000,
        'adaptive': True,
        'initial_simplex': None # Let scipy choose, or we can customize
    }
    
    # Custom initial simplex to help convergence?
    # Default simplex is usually fine.
    
    start_time = time.time()
    
    # We need a wrapper to stop if time runs out
    # scipy.optimize.minimize doesn't have a built-in time limit callback that stops cleanly 
    # in all versions, but we can check time inside the objective function and raise an exception 
    # or just return a bad value. However, raising an exception might crash the optimizer.
    # A safer way: run the optimizer and hope it finishes, or use a thread.
    # Given the 150s limit, and N=100, Nelder-Mead can be slow.
    # Let's estimate: 100 variables * 3 = 300 dimensions.
    # Nelder-Mead requires ~2*D function evaluations per iteration.
    # 600 evals/iter.
    # Each eval is O(N^2) = 10000 ops.
    # 600 * 10000 = 6e6 ops per iter.
    # Python overhead is high.
    # This might be too slow for 150s.
    
    # Alternative: Use a coarser optimization first?
    # Or use a different method?
    # L-BFGS-B with gradient? We can compute gradient of the soft-min.
    # Gradient of sum(exp(-beta * d)) wrt c:
    # d/dc_j sum_i exp(-beta * d_ij) = -beta * sum_i exp(-beta * d_ij) * (c_j - c_i) / d_ij
    # This is O(N^2) as well.
    
    # Let's try a hybrid: Use L-BFGS-B with analytical gradients if possible, or just Nelder-Mead with fewer iterations.
    # Actually, for 300 dimensions, Nelder-Mead is often impractical.
    
    # Let's switch to a gradient-based method: L-BFGS-B.
    # We need to implement the gradient.
    
    def _softmin_grad(flat_c, beta=200.0):
        c = flat_c.reshape(N, 3)
        grad = np.zeros_like(c)
        
        # Wall gradients
        # w_i = min(0.5 - c_i0, 0.5 + c_i0, ...)
        # The soft-min gradient for walls:
        # Let u_k be the 6 face distances for sphere i.
        # soft_min(u) = -1/beta log(sum(exp(-beta u_k)))
        # d(soft_min)/du_k = -exp(-beta u_k) / sum(exp(-beta u_j))
        # But we are minimizing sum(exp(-beta u_k)).
        # Let S = sum_k exp(-beta u_k).
        # dS/du_k = -beta exp(-beta u_k).
        # du_k/dc = +/- 1 depending on face.
        
        # For each sphere i:
        # u_0 = 0.5 - c_i0  -> du/dc_0 = -1
        # u_1 = 0.5 + c_i0  -> du/dc_0 = 1
        # u_2 = 0.5 - c_i1  -> du/dc_1 = -1
        # u_3 = 0.5 + c_i1  -> du/dc_1 = 1
        # u_4 = 0.5 - c_i2  -> du/dc_2 = -1
        # u_5 = 0.5 + c_i2  -> du/dc_2 = 1
        
        wall_dists = np.empty((N, 6))
        wall_dists[:, 0] = 0.5 - c[:, 0]
        wall_dists[:, 1] = 0.5 + c[:, 0]
        wall_dists[:, 2] = 0.5 - c[:, 1]
        wall_dists[:, 3] = 0.5 + c[:, 1]
        wall_dists[:, 4] = 0.5 - c[:, 2]
        wall_dists[:, 5] = 0.5 + c[:, 2]
        
        # exp(-beta * wall_dists)
        exp_w = np.exp(-beta * wall_dists)
        # Sum over faces for each sphere
        sum_w = np.sum(exp_w, axis=1)
        
        # Gradient contribution from walls
        # dS/d c_i0 = dS/du_0 * du_0/dc_i0 + dS/du_1 * du_1/dc_i0
        # = -beta exp_w_i0 * (-1) + -beta exp_w_i1 * (1)
        # = beta (exp_w_i0 - exp_w_i1)
        
        grad[:, 0] += beta * (exp_w[:, 0] - exp_w[:, 1])
        grad[:, 1] += beta * (exp_w[:, 2] - exp_w[:, 3])
        grad[:, 2] += beta * (exp_w[:, 4] - exp_w[:, 5])
        
        # Pairwise gradients
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(dist_sq)
        np.fill_diagonal(dist, 1e-10) # Avoid div by zero
        
        # exp(-beta * dist)
        exp_d = np.exp(-beta * dist)
        np.fill_diagonal(exp_d, 0.0)
        
        # dS/d c_j = sum_{i != j} dS/d d_ij * d d_ij / d c_j
        # dS/d d_ij = -beta exp(-beta d_ij)
        # d d_ij / d c_j = (c_j - c_i) / d_ij
        # So term is -beta exp(-beta d_ij) * (c_j - c_i) / d_ij
        
        # We can compute this matrix product efficiently.
        # Weight matrix W_ij = -beta * exp(-beta * d_ij) / d_ij
        # Then grad_j = sum_i W_ij * (c_j - c_i)
        # = sum_i W_ij * c_j - sum_i W_ij * c_i
        # = c_j * sum_i W_ij - sum_i (W_ij * c_i)
        
        # Create weight matrix
        # W is N x N
        # W_ij = -beta * exp_d_ij / dist_ij
        W = -beta * exp_d / dist
        np.fill_diagonal(W, 0.0)
        
        # sum_i W_ij is the row sum of W? No, sum over i for fixed j.
        # W is symmetric, so row sum = col sum.
        # Let's compute S1 = W.sum(axis=0) # Sum over i for each j
        # Let's compute S2 = W @ c # (W @ c)_j = sum_i W_ij c_i
        
        S1 = W.sum(axis=0) # Shape (N,)
        S2 = W @ c         # Shape (N, 3)
        
        # grad_j = c_j * S1_j - S2_j
        # S1_j is scalar, c_j is vector (3,).
        # So grad_j = c_j * S1_j[..., np.newaxis] - S2_j
        
        grad += c * S1[:, np.newaxis] - S2
        
        return grad.flatten()
    
    # Use L-BFGS-B
    # We need to define the objective and gradient for scipy
    def obj_func(x):
        return _softmin_objective(x, beta)
    
    def grad_func(x):
        return _softmin_grad(x, beta)
    
    # Bounds: [-0.5, 0.5] for all coordinates
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Run optimization
    # We'll run it for a fixed number of iterations or until time runs out.
    # Since we can't easily interrupt L-BFGS-B mid-iteration, we'll just run it and hope.
    # To be safe, we can run multiple short optimizations.
    
    # Strategy:
    # 1. Start with grid.
    # 2. Run L-BFGS-B for a few seconds.
    # 3. Check score.
    # 4. If time remains, try another start or restart.
    
    # For this function, we just run one optimization.
    # We'll rely on the caller to manage time and restarts.
    
    result = minimize(
        obj_func, 
        c_flat, 
        jac=grad_func, 
        method='L-BFGS-B', 
        bounds=bounds, 
        options={'maxiter': 2000, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    return result.x.reshape(N, 3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_c = None
    best_score = -1.0
    
    # Initialize with grid
    c_init = _init_grid()
    
    # We will run L-BFGS-B multiple times with different betas or restarts.
    # Since L-BFGS-B is fast per iteration but might get stuck, 
    # we can do a few iterations with a coarse beta, then a fine beta.
    
    # Phase 1: Coarse optimization (lower beta, smoother landscape)
    if time.time() < deadline - 10.0:
        # Use a lower beta for smoother gradient
        # We need to modify _softmin_objective and _softmin_grad to accept beta
        # They already do.
        # But the minimize call inside _optimize_from_initial uses a fixed beta.
        # Let's refactor _optimize_from_initial to accept beta.
        
        # Actually, let's just call the functions directly here for more control.
        
        beta_coarse = 50.0
        beta_fine = 200.0
        
        def obj_c(x): return _softmin_objective(x, beta_coarse)
        def grad_c(x): return _softmin_grad(x, beta_coarse)
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        c_flat = c_init.flatten()
        
        # Run coarse optimization
        # Limit time by checking after? No, minimize is blocking.
        # We'll just run it. If it's too slow, we'll see.
        # L-BFGS-B on 300 vars with O(N^2) cost.
        # 300 vars, ~600 evals/iter.
        # 2000 iters = 1.2M evals.
        # 1.2M * 10k ops = 12 billion ops. Too slow for 150s in Python.
        
        # We need to reduce iterations or dimensionality.
        # Maybe use a smaller beta and fewer iterations.
        # Or use a different method.
        
        # Let's try Nelder-Mead with a very small maxiter, just to polish.
        # Or use COBYLA?
        
        # Actually, the gradient computation is the bottleneck.
        # Can we make it faster?
        # The matrix multiplication W @ c is O(N^3) if done naively, but N=100, so 1e6 ops.
        # The element-wise operations are O(N^2).
        # So each gradient eval is ~100k-1M ops.
        # 1.2M evals * 1M ops = 1.2e12 ops. Still too slow.
        
        # We need to reduce the number of iterations significantly.
        # Maybe 100-200 iterations?
        # 200 * 600 = 120k evals.
        # 120k * 1M ops = 1.2e11 ops. Still likely too slow for Python.
        
        # Let's try to speed up the gradient calculation.
        # Use NumPy vectorization as much as possible.
        # The current implementation is already vectorized.
        # Maybe the overhead of Python function calls in the loop is high.
        
        # Alternative: Use a simpler objective?
        # Minimize the sum of squared violations?
        # Or just use the original SA but with better moves?
        
        # Let's stick to the original SA strategy but improve it?
        # The directive says "MATERIALLY DIFFERENT".
        
        # What if we use a different representation?
        # E.g., optimize the positions relative to the nearest neighbors?
        
        # Or use a different search method: Basin Hopping?
        # Basin Hopping uses a local optimizer (like L-BFGS-B or Nelder-Mead) and then perturbs the solution.
        # This combines the benefits of local optimization and global search.
        
        # Let's try Basin Hopping.
        # scipy.optimize.basin_hopping
        # It requires an objective function and a local optimizer.
        # We can use our L-BFGS-B as the local optimizer.
        # But Basin Hopping is also slow.
        
        # Let's try a simpler approach:
        # Use a coarser grid for the initial guess, then use a fast local optimizer.
        
        # Actually, let's just use the existing SA but with a better initialization and more restarts.
        # The current program has a bug in the SA logic (it doesn't track current score properly in the first block, 
        # but the second block _anneal_correct does).
        
        # Wait, the current program uses _anneal_correct.
        # Let's look at the inspirations.
        # Inspiration 1 and 2 use soft-min objective.
        # They scored 1.11.
        # The current program scores 1.00.
        # So the soft-min approach is better.
        
        # Let's implement the soft-min approach properly with L-BFGS-B, 
        # but optimize the performance.
        
        # Key optimization: Precompute indices for pairwise distances.
        # And use a single function for objective and gradient.
        
        # Let's rewrite the objective and gradient to be as fast as possible.
        
        iu, ju = np.triu_indices(N, k=1)
        
        def fast_obj(x, beta):
            c = x.reshape(N, 3)
            # Walls
            w0 = 0.5 - c[:, 0]
            w1 = 0.5 + c[:, 0]
            w2 = 0.5 - c[:, 1]
            w3 = 0.5 + c[:, 1]
            w4 = 0.5 - c[:, 2]
            w5 = 0.5 + c[:, 2]
            
            # Pairs
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            dist = np.sqrt(dist_sq)
            
            # Extract pairs
            pd = dist[iu, ju]
            
            # Combine
            # To avoid concatenation overhead, we can sum separately
            obj_w = np.sum(np.exp(-beta * w0)) + np.sum(np.exp(-beta * w1)) + \
                    np.sum(np.exp(-beta * w2)) + np.sum(np.exp(-beta * w3)) + \
                    np.sum(np.exp(-beta * w4)) + np.sum(np.exp(-beta * w5))
            
            obj_p = np.sum(np.exp(-beta * pd))
            
            return obj_w + obj_p
        
        def fast_grad(x, beta):
            c = x.reshape(N, 3)
            grad = np.zeros_like(c)
            
            # Walls
            w0 = 0.5 - c[:, 0]
            w1 = 0.5 + c[:, 0]
            w2 = 0.5 - c[:, 1]
            w3 = 0.5 + c[:, 1]
            w4 = 0.5 - c[:, 2]
            w5 = 0.5 + c[:, 2]
            
            e0 = np.exp(-beta * w0)
            e1 = np.exp(-beta * w1)
            e2 = np.exp(-beta * w2)
            e3 = np.exp(-beta * w3)
            e4 = np.exp(-beta * w4)
            e5 = np.exp(-beta * w5)
            
            grad[:, 0] += beta * (e0 - e1)
            grad[:, 1] += beta * (e2 - e3)
            grad[:, 2] += beta * (e4 - e5)
            
            # Pairs
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            dist = np.sqrt(dist_sq)
            np.fill_diagonal(dist, 1e-10)
            
            pd = dist[iu, ju]
            ed = np.exp(-beta * pd)
            
            # We need the full matrix W for the gradient calculation
            # W_ij = -beta * exp(-beta * d_ij) / d_ij
            # We can compute the full matrix or just the upper triangle and symmetrize.
            # Let's compute the full matrix for simplicity, N=100 is small.
            
            W = -beta * np.exp(-beta * dist) / dist
            np.fill_diagonal(W, 0.0)
            
            S1 = W.sum(axis=0)
            S2 = W @ c
            
            grad += c * S1[:, np.newaxis] - S2
            
            return grad.flatten()
        
        # Now, run L-BFGS-B
        # We'll run it for a limited number of iterations.
        
        c_flat = c_init.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Coarse phase
        beta_c = 50.0
        res_c = minimize(
            lambda x: fast_obj(x, beta_c),
            c_flat,
            jac=lambda x: fast_grad(x, beta_c),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxls': 20, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        c_c = res_c.x.reshape(N, 3)
        score_c = _score(c_c)
        
        if score_c > best_score:
            best_score = score_c
            best_c = c_c
        
        # Fine phase
        if time.time() < deadline - 5.0:
            beta_f = 200.0
            c_flat_f = c_c.flatten()
            res_f = minimize(
                lambda x: fast_obj(x, beta_f),
                c_flat_f,
                jac=lambda x: fast_grad(x, beta_f),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxls': 20, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            c_f = res_f.x.reshape(N, 3)
            score_f = _score(c_f)
            
            if score_f > best_score:
                best_score = score_f
                best_c = c_f
        
        # If time remains, try a random restart
        if time.time() < deadline - 10.0:
            c_rand = np.random.uniform(-0.5, 0.5, size=(N, 3))
            c_flat_r = c_rand.flatten()
            
            beta_r = 100.0
            res_r = minimize(
                lambda x: fast_obj(x, beta_r),
                c_flat_r,
                jac=lambda x: fast_grad(x, beta_r),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxls': 10, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            c_r = res_r.x.reshape(N, 3)
            score_r = _score(c_r)
            
            if score_r > best_score:
                best_score = score_r
                best_c = c_r
        
        return best_c

# EVOLVE-BLOCK-END
