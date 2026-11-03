import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _softmin(vals, beta):
    # Stable log-sum-exp for softmin
    # softmin(v) = -1/beta * log(sum(exp(-beta * v)))
    # To avoid overflow, subtract max(-beta * v)
    neg_beta_v = -beta * vals
    max_neg_beta_v = np.max(neg_beta_v)
    # If max is -inf, handle separately, but vals are finite
    sum_exp = np.sum(np.exp(neg_beta_v - max_neg_beta_v))
    return -(1.0 / beta) * (np.log(sum_exp) + max_neg_beta_v)

def _compute_radius(centers):
    # Compute the true non-smooth radius
    n = centers.shape[0]
    min_dist = np.inf
    
    # Pairwise distances
    # To save memory/time, we can do this in chunks or just full matrix for N=50
    # 50x50 matrix is small
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Boundary distances
    # dist to wall = 0.5 - |coord|
    # min over all coords
    min_boundary_dist = np.min(0.5 - np.abs(centers))
    
    return min(min_pair_dist, min_boundary_dist)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Initial best from seed (grid)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_centers = grid[:N]
    best_radius = _compute_radius(initial_centers)
    best_centers = initial_centers.copy()
    
    # Objective and Gradient functions
    def objective_and_grad(x, beta):
        centers = x.reshape(N, 3)
        
        # 1. Pairwise terms
        # We need sum of exp(-beta * v_k) * grad(v_k)
        # v_k = ||c_i - c_j|| / 2
        # grad_{c_i} v_k = (c_i - c_j) / (2 * ||c_i - c_j||)
        # grad_{c_j} v_k = (c_j - c_i) / (2 * ||c_i - c_j||)
        
        # 2. Boundary terms
        # v_k = 0.5 - |c_ik|
        # grad_{c_ik} v_k = -sign(c_ik)
        
        # We will compute the weighted sum of gradients directly.
        # Weight w_k = exp(-beta * v_k) / S, where S = sum(exp(-beta * v_k))
        # Note: objective is -softmin = 1/beta * log(S)
        # grad objective = (1/beta) * (1/S) * sum(exp(-beta v_k) * grad v_k) * (-beta) ?
        # Let's re-derive.
        # f = -softmin(v) = 1/beta * log(S)
        # df/dx = 1/beta * (1/S) * dS/dx
        # dS/dx = sum exp(-beta v_k) * (-beta * dv_k/dx)
        # df/dx = 1/beta * 1/S * (-beta) * sum exp(-beta v_k) * dv_k/dx
        # df/dx = -1/S * sum exp(-beta v_k) * dv_k/dx
        
        # We want to MINIMIZE df/dx? No, we want to MAXIMIZE softmin.
        # So we MINIMIZE -softmin.
        # Let L = -softmin(v).
        # dL/dx = - df/dx = 1/S * sum exp(-beta v_k) * dv_k/dx.
        
        # So we compute G = sum exp(-beta v_k) * dv_k/dx, then divide by S.
        
        # Initialize gradient array
        grad_sum = np.zeros_like(centers)
        
        # Pairwise contributions
        # We iterate over i < j
        # For efficiency, we can vectorize, but N=50 is small enough for loops if needed, 
        # but let's try to vectorize the weights first.
        
        # Compute all pairwise distances and weights
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Avoid division by zero
        dists = np.where(dists < 1e-10, 1e-10, dists)
        
        v_pair = dists / 2.0
        # Mask out diagonal
        mask = ~np.eye(N, dtype=bool)
        v_pair_masked = np.where(mask, v_pair, 0) # Only use i!=j
        
        # Boundary contributions
        v_bound = 0.5 - np.abs(centers)
        
        # Combine all v_k into a flat array for softmin calculation
        # Number of pair terms: N*(N-1)/2
        # Number of bound terms: N*3
        # We need the sum of exp(-beta * v_k) for all k.
        
        # Let's compute the weights w_k = exp(-beta * v_k)
        # To avoid overflow/underflow, we can normalize by max.
        # But for the gradient, we need the relative weights.
        
        # Let's just compute S = sum(exp(-beta * v_k))
        # And G = sum(exp(-beta * v_k) * grad_v_k)
        
        # Pairwise part
        # v_pair is NxN. We only want i<j.
        # exp_pair = exp(-beta * v_pair)
        # We need sum over i<j of exp_pair[i,j] * (c_i - c_j) / (2 * dist[i,j])
        # Note: grad_{c_i} v_{ij} = (c_i - c_j) / (2 * dist_{ij})
        # grad_{c_j} v_{ij} = (c_j - c_i) / (2 * dist_{ij})
        
        # Let's compute the matrix of unit vectors scaled by 1/(2*dist)
        # U[i,j] = (c_i - c_j) / (2 * dist[i,j])
        
        # To handle the mask i<j, we can sum over all i,j and divide by 2?
        # Sum_{i,j, i!=j} exp(-beta v_{ij}) * U[i,j]
        # Note U[i,j] = -U[j,i]
        # So Sum_{i,j} ... = 2 * Sum_{i<j} ...
        # So we can compute the full sum and divide by 2.
        
        # However, we also need S_pair = sum_{i<j} exp(-beta v_{ij})
        # S_pair = 0.5 * sum_{i!=j} exp(-beta v_{ij})
        
        exp_pair = np.exp(-beta * v_pair)
        # Zero out diagonal
        np.fill_diagonal(exp_pair, 0)
        
        S_pair = 0.5 * np.sum(exp_pair)
        
        # Gradient contribution from pairs
        # G_pair = 0.5 * sum_{i,j} exp_pair[i,j] * U[i,j]
        # U[i,j] = (c_i - c_j) / (2 * dist[i,j])
        
        # We can compute this as:
        # G_pair = 0.5 * ( sum_j (c_i - c_j) * exp_pair[i,j] / (2 * dist[i,j]) )_i
        # Let's compute the matrix W = exp_pair / (2 * dist)
        W = exp_pair / (2.0 * dists)
        np.fill_diagonal(W, 0)
        
        # G_pair_i = 0.5 * sum_j W[i,j] * (c_i - c_j)
        # = 0.5 * ( c_i * sum_j W[i,j] - sum_j W[i,j] * c_j )
        
        sum_W = np.sum(W, axis=1) # Shape (N,)
        # sum_j W[i,j] * c_j is matrix multiplication W.T @ c? No, W @ c
        # W is NxN. c is Nx3.
        # (W @ c)[i] = sum_j W[i,j] c_j
        W_c = W.dot(centers)
        
        G_pair = 0.5 * (centers * sum_W[:, np.newaxis] - W_c)
        
        # Boundary part
        # v_bound[i,k] = 0.5 - |c_ik|
        # grad_{c_ik} v_bound[i,k] = -sign(c_ik)
        # If c_ik == 0, sign is 0, but derivative is undefined. We can use 0 or subgradient.
        # exp_bound = exp(-beta * v_bound)
        exp_bound = np.exp(-beta * v_bound)
        S_bound = np.sum(exp_bound)
        
        # G_bound[i,k] = exp_bound[i,k] * (-sign(c_ik))
        signs = np.sign(centers)
        # If center is 0, sign is 0. This is fine.
        G_bound = -exp_bound * signs
        
        # Total S and G
        S_total = S_pair + S_bound
        G_total = G_pair + G_bound
        
        # Gradient of L = -softmin
        # dL/dx = G_total / S_total
        grad = G_total / S_total
        
        # Objective value
        # L = -softmin = 1/beta * log(S_total)
        # We return L and grad
        obj = (1.0 / beta) * np.log(S_total)
        
        return obj, grad.ravel()

    # Optimization loop
    num_restarts = 20
    betas = [5, 10, 20, 50, 100, 200]
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        if r == 0:
            x0 = best_centers.ravel()
        else:
            # Random initialization within the cube
            # Add some noise to previous best or pure random
            if r % 2 == 0:
                x0 = best_centers.ravel() + np.random.normal(0, 0.05, N*3)
                x0 = np.clip(x0, -0.5, 0.5)
            else:
                x0 = np.random.uniform(-0.4, 0.4, N*3)
                
        # Annealing over betas
        current_x = x0
        for beta in betas:
            if time.time() > deadline:
                break
                
            try:
                res = minimize(
                    objective_and_grad,
                    current_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True, # We return (f, grad)
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                current_x = res.x
            except Exception:
                pass
                
        # Evaluate true radius
        centers = current_x.reshape(N, 3)
        # Ensure inside cube (should be due to bounds)
        centers = np.clip(centers, -0.5, 0.5)
        
        r_curr = _compute_radius(centers)
        
        if r_curr > best_radius:
            best_radius = r_curr
            best_centers = centers.copy()
            
    return best_centers
# EVOLVE-BLOCK-END
