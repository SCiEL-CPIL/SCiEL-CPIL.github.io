import time
import numpy as np
from scipy.optimize import minimize
import random

DIMENSIONS = [2, 3, 3]

def construct():
    """
    Attempts to find a rank-16 exact tensor decomposition for 2x3 by 3x3 matrix multiplication.
    Falls back to the known rank-17 decomposition if rank-16 is not found or verification fails.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=3
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety

    # Target rank to try first
    target_rank = 16
    
    # Function to compute the residual of the tensor equation
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # The tensor T_{i,j,k} = sum_r U_{i,r} * V_{j,r} * W_{k,r}
    # We want T to match the identity-like structure of matrix multiplication.
    # Specifically, C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor indices are (i, j, k) where i in [0, n), j in [0, m), k in [0, p).
    # The "target" tensor is 1 if i == k and j is the middle index? No.
    # Standard matrix mult: C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T_{i,j,k} is 1 if the operation A_{ij} * B_{jk} contributes to C_{ik}.
    # So T_{i,j,k} = 1 for all i,j,k? No.
    # The decomposition is: sum_r (U_r \otimes V_r \otimes W_r) = T_{mult}
    # where T_{mult}[i,j,k] = 1 if the product A_{ij}B_{jk} is part of the sum for C_{ik}.
    # Actually, the tensor T_{mult} has T_{i,j,k} = 1 for all i,j,k?
    # Let's check: C_{ik} = sum_j A_{ij} B_{jk}.
    # The term A_{ij} B_{jk} corresponds to tensor slice T_{i,j,k}.
    # For a fixed i, j, k, the coefficient is 1.
    # So the target tensor T is a 3D array of ones with shape (n, m, p)?
    # Wait. If T_{i,j,k} = 1, then sum_r U_{i,r} V_{j,r} W_{k,r} = 1.
    # This is the standard matrix multiplication tensor.
    
    # Let's construct the target tensor
    target = np.ones((n, m, p))
    
    # Reshape target to a 2D matrix for easier minimization
    # We are looking for U, V, W such that:
    # sum_r U_{i,r} V_{j,r} W_{k,r} = target[i,j,k]
    
    def flatten_params(params):
        r = len(params) // 3
        u_flat = params[:r*n*m]
        v_flat = params[r*n*m : 2*r*n*m]
        w_flat = params[2*r*n*m : 3*r*n*m]
        return u_flat, v_flat, w_flat

    def objective(params):
        u_flat, v_flat, w_flat = flatten_params(params)
        U = u_flat.reshape(n*m, target_rank)
        V = v_flat.reshape(m*p, target_rank)
        W = w_flat.reshape(p*n, target_rank)
        
        # Compute the reconstructed tensor
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # This is a tensor product.
        # We can compute this efficiently using einsum or loops.
        # For small dimensions, einsum is fine.
        # But we need to flatten it to compare with target.flatten()
        
        # Reconstruct T
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Compute residual
        residual = T - target
        return np.sum(residual**2)

    # Initial guess
    # We can try to initialize with random values or a heuristic.
    # Random initialization with multiple restarts is robust.
    
    best_rank = target_rank
    best_solution = None
    
    # Try to find a rank-16 solution
    max_restarts = 50
    restart_count = 0
    
    while time.time() < deadline and restart_count < max_restarts:
        # Initialize with random values
        # Scale: values should be around 1
        size = 3 * target_rank * (n*m if False else 1) # just for clarity
        num_params = 3 * target_rank * max(n*m, m*p, p*n) # This is wrong, sizes differ
        # U: n*m x r, V: m*p x r, W: p*n x r
        num_u = n*m * target_rank
        num_v = m*p * target_rank
        num_w = p*n * target_rank
        total_params = num_u + num_v + num_w
        
        x0 = np.random.uniform(-2, 2, size=total_params)
        
        # Define bounds to keep values in a reasonable range (helps convergence)
        bounds = [(-10, 10)] * total_params
        
        # Minimize
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if res.success:
            final_obj = res.fun
            if final_obj < 1e-6:
                # Potential solution found. Verify and round.
                u_flat, v_flat, w_flat = flatten_params(res.x)
                U = u_flat.reshape(n*m, target_rank)
                V = v_flat.reshape(m*p, target_rank)
                W = w_flat.reshape(p*n, target_rank)
                
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify exactness
                T_check = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(T_check, target, atol=1e-9):
                    best_rank = target_rank
                    best_solution = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    break # Found rank 16, stop searching
        
        restart_count += 1

    # If rank 16 was found, return it
    if best_solution:
        return best_solution

    # Otherwise, return the known rank-17 solution
    return _get_rank_17_solution()

def _get_rank_17_solution():
    """
    Returns the known rank-17 exact tensor decomposition for 2x3 by 3x3 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Term 1
    U[0][0] += 1; U[4][0] += 1
    V[0][0] += 1; V[4][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # Term 2
    U[3][1] += 1; U[4][1] += 1
    V[0][1] += 1
    W[1][1] += 1; W[3][1] -= 1
    
    # Term 3
    U[0][2] += 1
    V[1][2] += 1; V[4][2] -= 1
    W[2][2] += 1; W[3][2] += 1
    
    # Term 4
    U[4][3] += 1
    V[3][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[1][3] += 1
    
    # Term 5
    U[0][4] += 1; U[1][4] += 1
    V[4][4] += 1
    W[0][4] -= 1; W[2][4] += 1
    
    # Term 6
    U[3][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # Term 7
    U[1][6] += 1; U[4][6] -= 1
    V[3][6] += 1; V[4][6] += 1
    W[0][6] += 1
    
    # Term 8: A00B02
    U[0][7] += 1
    V[2][7] += 1
    W[4][7] += 1
    
    # Term 9: A01B12
    U[1][8] += 1
    V[5][8] += 1
    W[4][8] += 1
    
    # Term 10: A10B02
    U[3][9] += 1
    V[2][9] += 1
    W[5][9] += 1
    
    # Term 11: A11B12
    U[4][10] += 1
    V[5][10] += 1
    W[5][10] += 1
    
    # Term 12: A02B20
    U[2][11] += 1
    V[6][11] += 1
    W[0][11] += 1
    
    # Term 13: A02B21
    U[2][12] += 1
    V[7][12] += 1
    W[2][12] += 1
    
    # Term 14: A02B22
    U[2][13] += 1
    V[8][13] += 1
    W[4][13] += 1
    
    # Term 15: A12B20
    U[5][14] += 1
    V[6][14] += 1
    W[1][14] += 1
    
    # Term 16: A12B21
    U[5][15] += 1
    V[7][15] += 1
    W[3][15] += 1
    
    # Term 17: A12B22
    U[5][16] += 1
    V[8][16] += 1
    W[5][16] += 1
    
    return {"U": U, "V": V, "W": W}
