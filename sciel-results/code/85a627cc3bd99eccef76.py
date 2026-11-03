import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    def exact_radius(c):
        # c is (N, 3)
        # Compute min pairwise distance / 2
        # Compute min distance to boundary
        # Return min of the two
        
        # Pairwise
        # Use broadcasting for speed, N=50 is small enough for full matrix
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        dists[np.arange(N), np.arange(N)] = np.inf
        min_pair_dist = np.min(dists) / 2.0
        
        # Boundary
        # c is in [-0.5, 0.5]
        # dist to boundary is 0.5 - max(abs(c))
        max_abs = np.max(np.abs(c), axis=1)
        min_boundary_dist = np.min(0.5 - max_abs)
        
        return min(min_pair_dist, min_boundary_dist)

    def smooth_obj_grad(c, beta):
        # c is (N, 3)
        # Returns (objective, gradient)
        
        # Compute x values
        # x_pair = |ci - cj| / 2
        # x_bound = 0.5 - |cik|
        
        # We need sum exp(-beta * x)
        # And weights w = exp(-beta * x) / sum
        
        # Vectorized pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        dists[np.arange(N), np.arange(N)] = np.inf # Avoid 0/0 in gradient later
        
        x_pairs = dists / 2.0
        
        # Boundary
        max_abs = np.max(np.abs(c), axis=1)
        x_bounds = 0.5 - max_abs
        
        # Combine all x into one vector for log-sum-exp
        # Note: x_pairs has N*(N-1)/2 unique values, but matrix has N*N.
        # It's easier to just use the upper triangle or all off-diagonals.
        # Using all off-diagonals doubles the weight, which is fine for softmax, 
        # but we must be consistent in gradient.
        # Let's stick to unique pairs to save computation and be precise.
        
        # Get indices for i < j
        i_idx, j_idx = np.triu_indices(N, k=1)
        x_pair_vec = x_pairs[i_idx, j_idx]
        
        # Boundary values
        # x_bounds is (N, 3) -> flatten
        x_bound_vec = x_bounds.flatten() # Wait, x_bounds is (N,), derived from max over axis 1.
        # Actually, boundary constraint is min over k of (0.5 - |c_ik|).
        # So for each sphere i, we have 3 terms: 0.5-|xi|, 0.5-|yi|, 0.5-|zi|.
        # x_bounds should be these 3N values.
        
        abs_c = np.abs(c) # (N, 3)
        x_bound_matrix = 0.5 - abs_c
        x_bound_vec = x_bound_matrix.flatten() # (3N,)
        
        # Total x vector
        x_all = np.concatenate([x_pair_vec, x_bound_vec])
        
        # Log-sum-exp calculation
        # To avoid underflow, use max shift
        m = np.min(x_all)
        # If x_all contains inf (from diagonal if not masked properly in vec), handle it.
        # But we masked diagonal in matrix, and took triu, so no inf in x_pair_vec.
        # x_bound_vec is finite.
        
        # Actually, if any x is very small, exp(-beta*x) is large.
        # Standard log-sum-exp:
        # log(sum(exp(-beta * x))) = -beta * m + log(sum(exp(-beta * (x - m))))
        
        shifted = -beta * (x_all - m)
        # Clip to avoid overflow in exp? exp(shifted) <= 1.
        # If x-m is large positive, shifted is large negative -> exp is 0.
        # If x-m is small, shifted is near 0.
        
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        # Objective: (1/beta) * log(sum_exp) - m
        # J = (1/beta) * (-beta * m + log(sum_exp)) = -m + (1/beta) * log(sum_exp)
        obj = -m + (1.0 / beta) * np.log(sum_exp)
        
        # Gradient
        # weights w_k = exp(-beta * x_k) / sum_exp = exp_shifted_k / sum_exp
        w_all = exp_shifted / sum_exp
        
        grad = np.zeros_like(c)
        
        # 1. Pairwise contributions
        # w_pair = w_all[:len(x_pair_vec)]
        w_pair = w_all[:len(x_pair_vec)]
        
        # grad = -beta * sum w_k * grad(x_k)
        # For pair (i, j):
        # x_ij = 0.5 * |ci - cj|
        # grad_ci x_ij = 0.5 * (ci - cj) / |ci - cj|
        # grad_cj x_ij = 0.5 * (cj - ci) / |cj - cj|
        
        # We have w_pair[i] corresponding to pair (i_idx[i], j_idx[i])
        
        # Vectorized accumulation
        # For each pair k:
        # ci = c[i_idx[k]], cj = c[j_idx[k]]
        # d = ci - cj
        # dist = |d|
        # term_i = w_pair[k] * 0.5 * d / dist
        # term_j = w_pair[k] * 0.5 * (-d) / dist
        
        ci = c[i_idx]
        cj = c[j_idx]
        d = ci - cj
        dists_vec = np.linalg.norm(d, axis=1)
        
        # Avoid division by zero (should not happen if distinct)
        dists_vec[dists_vec < 1e-12] = 1e-12
        
        # term = w_pair[:, np.newaxis] * 0.5 * d / dists_vec[:, np.newaxis]
        terms = w_pair[:, np.newaxis] * 0.5 * d / dists_vec[:, np.newaxis]
        
        # Accumulate into grad
        # np.add.at(grad, i_idx, terms)
        # np.add.at(grad, j_idx, -terms)
        
        # Since N is small, we can use np.add.at
        np.add.at(grad, i_idx, terms)
        np.add.at(grad, j_idx, -terms)
        
        # 2. Boundary contributions
        # w_bound = w_all[len(x_pair_vec):]
        w_bound = w_all[len(x_pair_vec):]
        
        # x_bound_ik = 0.5 - |c_ik|
        # grad_ci x_bound_ik = -sign(c_ik) * e_k
        
        # w_bound is (3N,). Reshape to (N, 3)
        w_bound_mat = w_bound.reshape(N, 3)
        
        # grad_bound_i = - w_bound_mat[i] * sign(c[i])
        # Note: sign(0) is 0. If c is 0, gradient is 0.
        
        signs = np.sign(c)
        grad_bound = -w_bound_mat * signs
        
        grad += grad_bound
        
        # Total gradient is -beta * (sum of above)
        # Wait, my formula was:
        # grad J = -beta * sum w_k grad x_k
        # The terms above are sum w_k grad x_k.
        # So we multiply by -beta.
        
        grad = -beta * grad
        
        return obj, grad

    def optimize_start(c0, beta_max=1000.0, beta_step=2.0):
        c = c0.copy()
        beta = 1.0
        best_c = c0.copy()
        best_r = exact_radius(c0)
        
        while time.time() < deadline - 5.0:
            # Check if we should stop
            if beta > beta_max:
                break
            
            # Run L-BFGS-B
            res = minimize(
                lambda x: smooth_obj_grad(x, beta)[0],
                c,
                jac=lambda x: smooth_obj_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            c_new = res.x
            r_new = exact_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            
            # If improvement is tiny, increase beta
            # Or just increase beta every step to sharpen
            beta *= beta_step
            
            c = c_new
            
        return best_c, best_r

    # Initializations
    best_c = None
    best_r = 0.0
    
    # 1. Simple Cubic Grid (Seed)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c_seed = grid[:N]
    c_res, r_res = optimize_start(c_seed)
    if r_res > best_r:
        best_r = r_res
        best_c = c_res

    # 2. BCC Lattice
    # Generate BCC points
    # Unit cell side s.
    # We want to fit 50 points.
    # 50 is not a perfect cube number.
    # Let's try a 3x3x3 grid of BCC cells? 3*3*3*2 = 54 points.
    # Or 4x4x4? 128 points.
    # Let's just generate a dense BCC lattice and pick the best 50?
    # Or scale a 3x3x3 BCC (54 pts) and drop 4?
    
    # Simpler: Random starts
    np.random.seed(42)
    for i in range(5):
        if time.time() > deadline - 10.0:
            break
        c_rand = np.random.uniform(-0.5, 0.5, (N, 3))
        c_res, r_res = optimize_start(c_rand)
        if r_res > best_r:
            best_r = r_res
            best_c = c_res

    # Final check
    if best_c is None:
        return c_seed
    
    return best_c

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    def exact_radius(c):
        # c is (N, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        dists[np.arange(N), np.arange(N)] = np.inf
        min_pair_dist = np.min(dists) / 2.0
        
        max_abs = np.max(np.abs(c), axis=1)
        min_boundary_dist = np.min(0.5 - max_abs)
        
        return min(min_pair_dist, min_boundary_dist)

    def smooth_obj_grad(c, beta):
        # Vectorized pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        dists[np.arange(N), np.arange(N)] = np.inf
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        x_pair_vec = dists[i_idx, j_idx] / 2.0
        
        abs_c = np.abs(c)
        x_bound_matrix = 0.5 - abs_c
        x_bound_vec = x_bound_matrix.flatten()
        
        x_all = np.concatenate([x_pair_vec, x_bound_vec])
        
        m = np.min(x_all)
        shifted = -beta * (x_all - m)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        obj = -m + (1.0 / beta) * np.log(sum_exp)
        
        w_all = exp_shifted / sum_exp
        
        grad = np.zeros_like(c)
        
        w_pair = w_all[:len(x_pair_vec)]
        
        ci = c[i_idx]
        cj = c[j_idx]
        d = ci - cj
        dists_vec = np.linalg.norm(d, axis=1)
        dists_vec[dists_vec < 1e-12] = 1e-12
        
        terms = w_pair[:, np.newaxis] * 0.5 * d / dists_vec[:, np.newaxis]
        
        np.add.at(grad, i_idx, terms)
        np.add.at(grad, j_idx, -terms)
        
        w_bound = w_all[len(x_pair_vec):]
        w_bound_mat = w_bound.reshape(N, 3)
        signs = np.sign(c)
        grad_bound = -w_bound_mat * signs
        grad += grad_bound
        
        grad = -beta * grad
        
        return obj, grad

    def optimize_start(c0, beta_max=2000.0, beta_step=1.5):
        c = c0.copy()
        beta = 1.0
        best_c = c0.copy()
        best_r = exact_radius(c0)
        
        while time.time() < deadline - 5.0:
            if beta > beta_max:
                break
            
            res = minimize(
                lambda x: smooth_obj_grad(x, beta)[0],
                c,
                jac=lambda x: smooth_obj_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            c_new = res.x
            r_new = exact_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            
            beta *= beta_step
            c = c_new
            
        return best_c, best_r

    best_c = None
    best_r = 0.0
    
    # 1. Simple Cubic Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c_seed = grid[:N]
    c_res, r_res = optimize_start(c_seed)
    if r_res > best_r:
        best_r = r_res
        best_c = c_res

    # 2. Random Starts
    np.random.seed(42)
    for i in range(10):
        if time.time() > deadline - 10.0:
            break
        c_rand = np.random.uniform(-0.5, 0.5, (N, 3))
        c_res, r_res = optimize_start(c_rand)
        if r_res > best_r:
            best_r = r_res
            best_c = c_res

    if best_c is None:
        return c_seed
    
    return best_c
# EVOLVE-BLOCK-END
