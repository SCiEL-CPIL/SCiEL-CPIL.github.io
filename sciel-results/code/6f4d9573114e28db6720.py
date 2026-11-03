"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import cKDTree

def _compute_radius(centers):
    """Compute the radius score: min of half min-distance and min boundary distance."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint: centers must be inside [-0.5, 0.5]^2
    boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary_margin)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r = min(min_dist / 2.0, min_boundary)
    return r

def _soft_min_objective_and_grad(centers, beta=100.0):
    """
    Soft-min objective using log-sum-exp approximation.
    We want to maximize the minimum distance, so we minimize the negative.
    Objective = -soft_min(min_dist/2, min_boundary)
    
    Actually, let's define:
    d = min( min_dist/2, min_boundary )
    We want to maximize d.
    Let's use the "inflation" approach or a smooth approximation of the minimum.
    
    A common smooth approximation for min(a, b) is -log(exp(beta*a) + exp(beta*b))/beta is for sum.
    For min, we can use: -log(exp(-beta*a) + exp(-beta*b))/beta is not quite right.
    
    Let's use a different approach:
    Objective to MINIMIZE:
    f(x) = sum_i [ -log( sum_j exp(-beta * (d_ij)) ) ]  -- this is complex.
    
    Simpler: Maximize the minimum of all pairwise distances and boundary distances.
    Let D be the set of all distances (pairwise and to boundary).
    We want max(min(D)).
    
    Smooth approximation of max(min(D)):
    g(x) = -1/beta * log( sum_{d in D} exp(-beta * d) )
    This is the soft-min. As beta -> inf, it converges to min(D).
    We want to MAXIMIZE g(x).
    So we MINIMIZE -g(x).
    
    f(x) = -g(x) = 1/beta * log( sum_{d in D} exp(-beta * d) )
    
    Gradient of f with respect to x:
    df/dx = 1/beta * (1/S) * sum_d (exp(-beta*d) * (-beta) * dd/dx)
    where S = sum_d exp(-beta*d)
    df/dx = -1/S * sum_d exp(-beta*d) * dd/dx
    
    Let w_d = exp(-beta*d) / S
    df/dx = - sum_d w_d * dd/dx
    
    For pairwise distance d_ij = ||x_i - x_j||:
    dd_ij/dx_i = (x_i - x_j) / d_ij
    dd_ij/dx_j = (x_j - x_i) / d_ij
    
    For boundary distance d_i = 0.5 - max(|x_ix|, |x_iy|):
    This is non-differentiable at |x_ix| = |x_iy|.
    We can use a smooth approximation for the boundary too, or just handle the gradient carefully.
    Actually, d_i = min(0.5 - x_ix, 0.5 + x_ix, 0.5 - x_iy, 0.5 + x_iy)
    This is the minimum of 4 linear functions.
    The gradient of the min of linear functions is the gradient of the active one(s).
    If we use the soft-min for the boundary too, it becomes smooth.
    
    Let's use the soft-min for ALL constraints (pairwise and boundary).
    
    D = { d_ij for all i<j } U { d_ix for all i } U { d_iy for all i }
    where d_ix = 0.5 - x_ix, d_iy = 0.5 - x_iy, etc.
    Actually, boundary distance for point i is:
    b_i = min(0.5 - x_ix, 0.5 + x_ix, 0.5 - x_iy, 0.5 + x_iy)
    
    Let's define the full set of "constraints" as distances.
    Total number of constraints: N*(N-1)/2 + 4*N
    For N=200, that's 19900 + 800 = 20700 constraints.
    This is manageable.
    """
    centers = np.asarray(centers, dtype=float)
    
    # To speed up gradient calculation, we can precompute some things.
    # But for now, let's compute directly.
    
    # Pairwise distances
    # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # d_sq = np.sum(diff**2, axis=2)
    # d = np.sqrt(d_sq)
    # np.fill_diagonal(d, np.inf)
    
    # This O(N^2) memory is 200*200*8 = 320KB, which is fine.
    
    # However, computing the gradient for 20000 terms every step is slow in pure Python/NumPy loops.
    # We need to vectorize the gradient computation.
    
    # Let's compute the gradient using vectorized operations.
    
    # 1. Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d_sq = np.sum(diff**2, axis=2)
    # Set diagonal to 0 to avoid division by zero, but we mask them out later
    np.fill_diagonal(d_sq, 1.0)
    d = np.sqrt(d_sq)
    
    # Boundary distances
    # d_ix = 0.5 - centers[:, 0]
    # d_iy = 0.5 - centers[:, 1]
    # d_jx = 0.5 + centers[:, 0]
    # d_jy = 0.5 + centers[:, 1]
    
    # Let's create a unified structure for the gradient.
    # grad = np.zeros_like(centers)
    
    # Weights for pairwise
    # w_ij = exp(-beta * d_ij) / S
    # S = sum of all exp(-beta * d)
    
    # To avoid overflow/underflow, we can subtract the max.
    # But with beta=100 and d~0.03, beta*d ~ 3, exp(-3) ~ 0.05. Fine.
    
    # Let's compute all distances and their exponentials.
    
    # Pairwise
    # We only need i < j to avoid double counting in the sum, but for gradient symmetry...
    # Actually, the objective is symmetric.
    # f = 1/beta * log( sum_{i<j} exp(-beta*d_ij) + sum_i (exp(-beta*b_i) + ...) )
    
    # Let's just compute the gradient for the pairwise part and boundary part separately and sum them.
    
    # --- Pairwise Part ---
    # d_ij for i != j
    # grad_i += -w_ij * (x_i - x_j)/d_ij
    # grad_j += -w_ij * (x_j - x_i)/d_ij
    
    # Vectorized:
    # diff is (N, N, 2)
    # d is (N, N)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    exp_pair = np.where(mask, np.exp(-beta * d), 0.0)
    
    # Sum of pairwise exponentials
    S_pair = np.sum(exp_pair)
    
    # Gradient contribution from pairwise
    # term_ij = w_ij * diff_ij / d_ij
    # w_ij = exp_pair_ij / S_total
    
    # We'll compute the full S_total later. For now, let's assume we add contributions.
    
    # --- Boundary Part ---
    # b_i = min(0.5-x_i, 0.5+x_i, 0.5-y_i, 0.5+y_i)
    # Soft min approximation for boundary:
    # b_i_soft = -1/beta * log( sum_{k in {ix, iy, jx, jy}} exp(-beta * d_ik) )
    # Wait, the objective is min(pairwise_dist/2, boundary_dist).
    # Actually, the radius is r = min(min_dist/2, min_boundary).
    # So we are minimizing over:
    # 1. d_ij / 2 for all i<j
    # 2. b_i for all i
    
    # So the set of values to take the min of is:
    # V = { d_ij / 2 } U { b_i }
    
    # Let's recompute with this in mind.
    
    # Values:
    # vals_pair = d / 2.0
    # vals_bound = b_i
    
    # b_i is the min of 4 distances to the boundary lines.
    # Let's use the soft-min for b_i as well, or just use the hard min if we're careful with gradients.
    # Given the complexity, let's stick to the hard min for boundary in the gradient,
    # approximating the gradient as that of the closest boundary.
    
    # Actually, for a robust smooth objective, let's use the soft-min for the FINAL radius.
    # r_soft = -1/beta * log( sum_{v in V} exp(-beta * v) )
    # We want to maximize r_soft.
    
    # Let's compute the sum of exp(-beta * v) for all v in V.
    
    # V has N*(N-1)/2 pairwise terms (d_ij/2) and N boundary terms (b_i).
    
    # Compute pairwise terms
    # d_ij / 2
    pair_vals = d / 2.0
    np.fill_diagonal(pair_vals, np.inf)  # exclude self
    
    # Boundary terms
    # b_i = min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i)
    # For gradient, we need the active constraint.
    # Let's compute the 4 boundary distances for each point.
    b1 = 0.5 - centers[:, 0]
    b2 = 0.5 + centers[:, 0]
    b3 = 0.5 - centers[:, 1]
    b4 = 0.5 + centers[:, 1]
    
    # b_i = min(b1, b2, b3, b4)
    # For the soft-min sum, we should include all 4? No, b_i is the min.
    # If we use the hard min b_i in the set V, then the soft-min over V is:
    # r_soft = -1/beta * log( sum_{i<j} exp(-beta * d_ij/2) + sum_i exp(-beta * b_i) )
    
    # This is smooth IF b_i is smooth. But b_i is not (it's a min of linears).
    # However, the function min(a,b,c,d) is differentiable almost everywhere.
    # The gradient is the gradient of the active constraint.
    
    # Let's compute the gradient of the objective f(x) = -r_soft(x)
    # f(x) = 1/beta * log( S )
    # where S = sum_{i<j} exp(-beta * d_ij/2) + sum_i exp(-beta * b_i)
    
    # df/dx = 1/beta * (1/S) * dS/dx
    # dS/dx = sum_{i<j} exp(-beta * d_ij/2) * (-beta/2) * d(d_ij)/dx
    #       + sum_i exp(-beta * b_i) * (-beta) * d(b_i)/dx
    
    # So df/dx = -1/S * [ sum_{i<j} exp(-beta * d_ij/2) * (1/2) * d(d_ij)/dx
    #                   + sum_i exp(-beta * b_i) * d(b_i)/dx ]
    
    # Let's compute this.
    
    # 1. Pairwise part
    # term_pair = exp(-beta * d_ij/2) * (1/2) * diff_ij / d_ij
    # Note: d(d_ij)/dx_i = diff_ij / d_ij, d(d_ij)/dx_j = -diff_ij / d_ij
    
    # Vectorized:
    # diff is (N, N, 2)
    # d is (N, N)
    # exp_pair is (N, N)
    
    # We need to sum over j for each i, and over i for each j.
    # This is like a matrix-vector product but with element-wise multiplication.
    
    # Let's compute the gradient contribution for each point i.
    # grad_i = -1/S * [ sum_{j != i} exp_pair[i,j] * (1/2) * diff[i,j] / d[i,j] ]
    
    # diff[i,j] = x_i - x_j
    # So the sum is: sum_j w_ij * (x_i - x_j) / d_ij
    # = x_i * sum_j w_ij / d_ij - sum_j w_ij * x_j / d_ij
    
    # This is efficient!
    
    # Let's compute:
    # W[i,j] = exp_pair[i,j] * (1/2) / d[i,j]  for i != j, 0 otherwise.
    # Then grad_i_pair = x_i * sum_j W[i,j] - sum_j W[i,j] * x_j
    
    # Boundary part:
    # grad_i_bound = -1/S * exp(-beta * b_i) * d(b_i)/dx_i
    # d(b_i)/dx_i is:
    # if b1 is min: (-1, 0)
    # if b2 is min: (1, 0)
    # if b3 is min: (0, -1)
    # if b4 is min: (0, 1)
    
    # If there's a tie, we average.
    
    # Let's compute the active boundary for each point.
    # b_vals = np.column_stack([b1, b2, b3, b4])
    # min_idx = np.argmin(b_vals, axis=1)
    
    # gradients:
    # grad_b = np.zeros((N, 2))
    # grad_b[min_idx==0] = (-1, 0)
    # ... etc.
    
    # This is a valid structural approach. Let's implement it.
    
    # Compute S
    S = S_pair + np.sum(np.exp(-beta * np.min(b_vals, axis=1)))  # Wait, b_vals needs to be computed
    
    # Let's restructure the function for clarity and performance.
    
    pass  # Placeholder, actual implementation below

def _soft_min_objective_and_grad_impl(centers, beta=50.0):
    centers = np.asarray(centers, dtype=float)
    
    # 1. Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(d_sq, 1.0)
    d = np.sqrt(d_sq)
    
    # 2. Boundary distances
    b1 = 0.5 - centers[:, 0]
    b2 = 0.5 + centers[:, 0]
    b3 = 0.5 - centers[:, 1]
    b4 = 0.5 + centers[:, 1]
    b_vals = np.column_stack([b1, b2, b3, b4])
    b_min = np.min(b_vals, axis=1)
    
    # 3. Compute S = sum exp(-beta * v)
    # v_pair = d / 2.0 (with diagonal inf/0)
    # v_bound = b_min
    
    # Use log-sum-exp for stability if beta is large, but beta=50 is small.
    
    exp_pair = np.exp(-beta * d / 2.0)
    np.fill_diagonal(exp_pair, 0.0)
    S_pair = np.sum(exp_pair)
    
    exp_bound = np.exp(-beta * b_min)
    S_bound = np.sum(exp_bound)
    
    S = S_pair + S_bound
    
    if S <= 0:
        return 0.0, np.zeros_like(centers)
    
    # Objective: f = 1/beta * log(S)
    obj = (1.0 / beta) * np.log(S)
    
    # Gradient:
    # df/dx = -1/S * [ sum_{i!=j} exp_pair[i,j] * 0.5 * diff[i,j] / d[i,j] * ??? ]
    # Wait, the derivative of exp(-beta * d/2) wrt x is:
    # exp(-beta * d/2) * (-beta/2) * d(d)/dx
    # So dS/dx = sum ... exp(...) * (-beta/2) * d(d)/dx + sum exp(-beta*b) * (-beta) * d(b)/dx
    # df/dx = 1/beta * 1/S * dS/dx = -1/S * [ sum exp_pair * 0.5 * d(d)/dx + sum exp_bound * d(b)/dx ]
    
    # --- Pairwise Gradient ---
    # We want: grad_i = -1/S * sum_{j!=i} exp_pair[i,j] * 0.5 * (diff[i,j] / d[i,j])
    # Let W[i,j] = exp_pair[i,j] * 0.5 / d[i,j]
    # Then grad_i = -1/S * sum_j W[i,j] * diff[i,j]
    # diff[i,j] = centers[i] - centers[j]
    # sum_j W[i,j] * (centers[i] - centers[j]) = centers[i] * sum_j W[i,j] - sum_j W[i,j] * centers[j]
    
    # Compute W
    # Avoid division by zero
    W = np.zeros_like(d)
    mask = d > 1e-8
    W[mask] = exp_pair[mask] * 0.5 / d[mask]
    np.fill_diagonal(W, 0.0)
    
    # Sum over j
    sum_W = np.sum(W, axis=1)  # (N,)
    
    # sum_j W[i,j] * centers[j]
    # This is W @ centers
    W_centered = W.dot(centers)  # (N, 2)
    
    grad_pair = centers * sum_W[:, np.newaxis] - W_centered
    
    # --- Boundary Gradient ---
    # d(b_min)/dx_i
    # Find the active boundary
    min_idx = np.argmin(b_vals, axis=1)
    
    grad_bound = np.zeros_like(centers)
    # b1: (-1, 0)
    grad_bound[min_idx == 0, 0] -= exp_bound[min_idx == 0]
    # b2: (1, 0)
    grad_bound[min_idx == 1, 0] += exp_bound[min_idx == 1]
    # b3: (0, -1)
    grad_bound[min_idx == 2, 1] -= exp_bound[min_idx == 2]
    # b4: (0, 1)
    grad_bound[min_idx == 3, 1] += exp_bound[min_idx == 3]
    
    # Total Gradient
    grad = -(1.0 / S) * (grad_pair + grad_bound)
    
    return obj, grad

def _generate_hex_initial(n, jitter=0.0):
    """Generate a hexagonal grid initialization."""
    r_est = 0.038
    dx = 2 * r_est
    dy = dx * np.sqrt(3) / 2
    
    best_points = None
    best_score = -np.inf
    
    for n_rows in range(12, 18):
        n_cols = int(np.ceil(n / n_rows))
        if n_rows * n_cols < n:
            continue
            
        points = []
        for row in range(n_rows):
            y = row * dy
            offset = 0 if row % 2 == 0 else dx / 2
            for col in range(n_cols):
                x = col * dx + offset
                points.append((x, y))
        
        points = np.array(points[:n])
        
        # Center the points
        center = points.mean(axis=0)
        points -= center
        
        # Scale to fit
        max_extent = np.max(np.abs(points))
        if max_extent > 0:
            scale = 0.48 / max_extent
            points *= scale
            
        if jitter > 0:
            points += np.random.normal(0, jitter, points.shape)
        
        if np.all(np.abs(points) <= 0.5):
            score = _compute_radius(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
    if best_points is not None:
        return best_points
        
    # Fallback
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]

def _optimize_with_l_bfgs(centers, deadline, max_iter=500):
    """Optimize using L-BFGS-B with the soft-min objective."""
    
    def func(x):
        obj, _ = _soft_min_objective_and_grad_impl(x, beta=50.0)
        return obj
        
    def jac(x):
        _, grad = _soft_min_objective_and_grad_impl(x, beta=50.0)
        return grad
        
    # Flatten
    x0 = centers.flatten()
    
    # Bounds: [-0.5, 0.5] for each coordinate
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    result = minimize(func, x0, jac=jac, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
    
    best_centers = result.x.reshape(N, 2)
    best_score = _compute_radius(best_centers)
    
    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # 1. Initialize
    best_centers = _generate_hex_initial(N)
    best_score = _compute_radius(best_centers)
    
    # 2. Optimization with L-BFGS-B
    if time.time() < deadline - 10:
        centers, score = _optimize_with_l_bfgs(best_centers, deadline - 10, max_iter=300)
        if score > best_score:
            best_score = score
            best_centers = centers
            
    # 3. Local refinement with small random moves (Simulated Annealing)
    if time.time() < deadline:
        centers = best_centers.copy()
        current_score = best_score
        T = 1e-4
        T_min = 1e-7
        cooling = 0.995
        
        for it in range(10000):
            if time.time() > deadline:
                break
                
            # Move a random point
            idx = np.random.randint(N)
            move = np.random.normal(0, 0.005, 2)
            new_centers = centers.copy()
            new_centers[idx] += move
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_score = _compute_radius(new_centers)
            
            # Metropolis acceptance
            if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / T):
                centers = new_centers
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_centers = centers.copy()
                    
            T = max(T * cooling, T_min)
            
    return best_centers
# EVOLVE-BLOCK-END
