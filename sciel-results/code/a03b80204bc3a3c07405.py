import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Initialization
    # Grid initialization
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random noise to break symmetry
    np.random.seed(42)
    centers += np.random.normal(0, 0.001, centers.shape)
    centers = np.clip(centers, -0.49, 0.49)

    def calc_radius(c):
        # Exact radius calculation
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    best_r = calc_radius(centers)
    best_centers = centers.copy()

    # 2. Objective and Gradient for Soft-Min
    # x_k are the distances:
    # - N(N-1)/2 pairwise distances / 2
    # - 6N wall distances (distance to each of the 6 faces)
    
    # We store the weights for the current beta
    # To avoid recomputing expensive parts in the gradient if possible, 
    # but here N is small, so direct computation is fine.
    
    def objective_and_grad(c_flat, beta):
        c = c_flat.reshape(N, 3)
        
        # --- Pairwise Distances ---
        # c: (N, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]  # (N, N, 3)
        d = np.linalg.norm(diff, axis=2)  # (N, N)
        np.fill_diagonal(d, np.inf)
        
        x_pair = d / 2.0  # (N, N)
        
        # We only sum over i < j
        # Create a mask for upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # Flatten the valid parts
        x_pair_flat = x_pair[mask]
        
        # --- Wall Distances ---
        # For each sphere i and coordinate k, distance to +face and -face
        # x_i_k_plus = 0.5 - c[i,k]
        # x_i_k_minus = c[i,k] + 0.5
        
        c_x, c_y, c_z = c[:, 0], c[:, 1], c[:, 2]
        
        x_walls = np.concatenate([
            0.5 - c_x, c_x + 0.5,
            0.5 - c_y, c_y + 0.5,
            0.5 - c_z, c_z + 0.5
        ])
        
        # Combine all x_k
        x_all = np.concatenate([x_pair_flat, x_walls])
        
        # --- Log-Sum-Exp ---
        # J = (1/beta) * log(sum(exp(-beta * x_all)))
        # To avoid overflow/underflow, use stable log-sum-exp
        # max_x = np.max(x_all)
        # log_sum = max_x + np.log(np.sum(np.exp(-beta * (x_all - max_x))))
        # J = -log_sum / beta
        
        # Actually, we want to MINIMIZE J.
        # J = - softmin.
        # softmin = -1/beta * log(sum(exp(-beta * x)))
        # So J = 1/beta * log(sum(exp(-beta * x)))
        
        # Stable calculation:
        max_x = np.max(x_all)
        shifted = -beta * (x_all - max_x)
        log_sum_exp = max_x + np.log(np.sum(np.exp(shifted)))
        J = log_sum_exp / beta
        
        # --- Gradient ---
        # grad J = -1/beta * (sum(exp(-beta*x) * grad x) / sum(exp(-beta*x)))
        # Let E = exp(-beta * x_all)
        # S = sum(E)
        # W = E / S
        # grad J = - sum(W * grad x)
        
        E = np.exp(shifted)
        S = np.sum(E)
        W = E / S
        
        # Split weights back
        n_pairs = N * (N - 1) // 2
        W_pair = W[:n_pairs]
        W_wall = W[n_pairs:]
        
        G = np.zeros((N, 3))
        
        # --- Pairwise Gradient Contribution ---
        # W_pair corresponds to upper triangle of x_pair
        # We need to map these back to the (N, N) matrix
        W_pair_mat = np.zeros((N, N))
        W_pair_mat[mask] = W_pair
        # Symmetrize because the term for (i,j) contributes to both i and j
        # Wait, the sum in the log-sum-exp is over unique pairs {i,j}.
        # The weight W_{ij} applies to the distance d_ij.
        # The gradient of d_ij wrt c_i is (c_i - c_j) / (2 d_ij).
        # The gradient of d_ij wrt c_j is (c_j - c_i) / (2 d_ij).
        # So for a given pair (i,j) with weight w, we add:
        # w * (c_i - c_j) / (2 d_ij) to G[i]
        # w * (c_j - c_i) / (2 d_ij) to G[j]
        
        # Vectorized:
        # R = diff / (2 * d[..., None])  # (N, N, 3)
        # But d is inf on diagonal, so R is 0 on diagonal.
        # We need to handle division by zero safely.
        d_safe = np.where(d > 1e-12, d, 1.0)
        R = diff / (2.0 * d_safe[:, :, np.newaxis])
        # Set diagonal to 0
        np.fill_diagonal(R, 0.0)
        
        # G_pair[i] = sum_j W_pair_mat[i,j] * R[i,j]
        # Note: W_pair_mat is symmetric?
        # The weight w_{ij} is associated with the pair.
        # In the sum sum_{i<j} w_{ij} * grad_{c_i} x_{ij}, 
        # w_{ij} is the weight for the term x_{ij}.
        # So for G[i], we sum over j where (i,j) is a pair.
        # If we define W_pair_mat[i,j] = w_{ij} for i<j, and 0 otherwise.
        # Then G[i] += sum_j W_pair_mat[i,j] * R[i,j]  (for i<j)
        # And G[i] += sum_j W_pair_mat[j,i] * R[i,j]  (for j<i, where w_{ji}=w_{ij})
        
        # Let's make W_sym such that W_sym[i,j] = w_{ij} for all i!=j.
        W_sym = W_pair_mat + W_pair_mat.T
        
        # Now G_pair[i] = sum_j W_sym[i,j] * R[i,j]
        # This is a batched dot product.
        G_pair = np.einsum('ij,ijk->ik', W_sym, R)
        
        G += G_pair
        
        # --- Wall Gradient Contribution ---
        # W_wall is (6N,)
        # Order: 
        # 0..N-1: x+ (0.5 - c_x), grad -1
        # N..2N-1: x- (c_x + 0.5), grad +1
        # 2N..3N-1: y+ (0.5 - c_y), grad -1
        # 3N..4N-1: y- (c_y + 0.5), grad +1
        # 4N..5N-1: z+ (0.5 - c_z), grad -1
        # 5N..6N-1: z- (c_z + 0.5), grad +1
        
        # G_wall[i, 0] = - W_wall[i] + W_wall[N+i]
        # G_wall[i, 1] = - W_wall[2N+i] + W_wall[3N+i]
        # G_wall[i, 2] = - W_wall[4N+i] + W_wall[5N+i]
        
        G_wall = np.zeros((N, 3))
        G_wall[:, 0] = -W_wall[0:N] + W_wall[N:2*N]
        G_wall[:, 1] = -W_wall[2*N:3*N] + W_wall[3*N:4*N]
        G_wall[:, 2] = -W_wall[4*N:5*N] + W_wall[5*N:6*N]
        
        G += G_wall
        
        # Final gradient of J
        grad_J = G
        
        return J, grad_J.flatten()

    # 3. Optimization Loop
    beta = 5.0
    max_beta = 500.0
    beta_increment = 1.5
    
    current_centers = centers.copy()
    
    while beta < max_beta:
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Define function for minimize
        def obj_func(c_flat):
            val, _ = objective_and_grad(c_flat, beta)
            return val
        
        def grad_func(c_flat):
            _, grad = objective_and_grad(c_flat, beta)
            return grad
        
        res = minimize(
            obj_func, 
            current_centers.flatten(), 
            jac=grad_func, 
            method='L-BFGS-B', 
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
        )
        
        new_centers = res.x.reshape(N, 3)
        
        # Check validity and improve
        new_r = calc_radius(new_centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = new_centers.copy()
            current_centers = new_centers.copy()
        else:
            # If it didn't improve, keep current_centers but maybe update if it's still valid?
            # L-BFGS-B should converge to a local min of the soft objective.
            # If the soft min radius is worse, it might be due to beta being too low/high or local trap.
            # We'll keep the best one found so far.
            pass
            
        beta *= beta_increment
        
    # 4. Final Polish with High Beta
    # Run a few more iterations with a very high beta to sharpen the solution
    for _ in range(5):
        if time.time() - start_time > time_limit - 2.0:
            break
        
        beta = 1000.0
        def obj_func(c_flat):
            val, _ = objective_and_grad(c_flat, beta)
            return val
        
        def grad_func(c_flat):
            _, grad = objective_and_grad(c_flat, beta)
            return grad
        
        res = minimize(
            obj_func, 
            best_centers.flatten(), 
            jac=grad_func, 
            method='L-BFGS-B', 
            options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False}
        )
        
        new_centers = res.x.reshape(N, 3)
        new_r = calc_radius(new_centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = new_centers.copy()

    return best_centers
