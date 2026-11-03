import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0

    # Schoolbook decomposition (rank 60)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                U_school[idx_a, col] = 1
                V_school[idx_b, col] = 1
                W_school[idx_c, col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Precompute index mappings for speed
    idx_a_list = []
    idx_b_list = []
    idx_c_list = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a_list.append(i * m + j)
                idx_b_list.append(j * p + k)
                idx_c_list.append(k * n + i)
    idx_a_arr = np.array(idx_a_list)
    idx_b_arr = np.array(idx_b_list)
    idx_c_arr = np.array(idx_c_list)
    num_terms = n * m * p  # 60

    def compute_tensor(U, V, W):
        """Compute T[i,j,k] = sum_r U[idx_a, r] * V[idx_b, r] * W[idx_c, r]"""
        # Vectorized computation
        # For each (i,j,k), T = dot(U[idx_a], V[idx_b] * W[idx_c])
        # U[idx_a] shape: (60, rank)
        # V[idx_b] * W[idx_c] shape: (60, rank)
        UW = V[idx_b_arr, :] * W[idx_c_arr, :]  # (60, rank)
        T = np.sum(U[idx_a_arr, :] * UW, axis=1)  # (60,)
        return T

    def compute_error(U, V, W):
        T = compute_tensor(U, V, W)
        return np.sum((T - 1.0) ** 2)

    # Strategy: Try to find a rank-r decomposition using ALS (alternating least squares)
    # For each target rank, initialize and run ALS iterations
    
    np.random.seed(42)
    
    # Try ranks from 59 down to some lower bound
    # For 3x4x5, the known lower bound is not well-established, but let's try
    for target_rank in [59, 58, 57, 56, 55, 54, 53, 52, 51, 50]:
        if time.time() > deadline - 15:
            break
        
        # Multiple random initializations
        found = False
        for init in range(5):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random small values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # ALS: alternate fixing two factors and solving for the third
            # When U and W are fixed, for each (i,j,k):
            #   T[i,j,k] = sum_r U[idx_a, r] * V[idx_b, r] * W[idx_c, r]
            #   = sum_r (U[idx_a, r] * W[idx_c, r]) * V[idx_b, r]
            # So V[idx_b, r] should satisfy: for all k, sum_r V[idx_b, r] * (U[idx_a, r] * W[idx_c, r]) = 1
            # This is a linear system for V.
            
            # More precisely, fix U and W. For each j (0..m-1):
            #   For each k: T[i,j,k] involves V[j*p+k, :]
            #   The contribution of V[j*p+k, r] to T[i,j,k] is U[i*m+j, r] * W[k*n+i, r]
            #   We need: sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1 for all i,k
            #   This gives n*p equations for p variables per k... wait, V has m*p rows.
            
            # Let's think differently. Fix U and W. The tensor T is linear in V.
            # T[i,j,k] = sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r])
            # We want T[i,j,k] = 1 for all (i,j,k).
            # This is a linear system in the entries of V.
            # Number of unknowns: m * p * target_rank = 4*5*target_rank = 20*target_rank
            # Number of equations: n*m*p = 60
            # For target_rank < 3, we have more unknowns than equations (underdetermined)
            # For target_rank >= 3, we have at least as many unknowns as equations
            
            # Build the linear system: A * V_flat = b
            # where V_flat is V.reshape(-1)
            # A has shape (60, m*p*target_rank)
            
            # This might be too large for direct solve. Let's use a different approach.
            
            # Alternative: Use the fact that we can solve for V given U and W using
            # least squares. But the system is underdetermined for small ranks.
            
            # Let's try a different ALS formulation:
            # Fix U and V, solve for W.
            # T[i,j,k] = sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r])
            # For fixed (i,j): T[i,j,k] = sum_r W[k*n+i, r] * C[k, r] where C[k,r] = U[i*m+j,r]*V[j*p+k,r]
            # We want this to equal 1 for all k.
            # This is: sum_r W[k*n+i, r] * C[k, r] = 1 for all k
            # For fixed i, this is a system in W[:, i] (the column of W corresponding to index i)
            # Wait, W is indexed by (k*n+i), so for fixed i, W[k*n+i, :] for k=0..p-1 are p rows.
            # The equation is: sum_r W[k*n+i, r] * C[k, r] = 1
            # This is p equations in p*target_rank unknowns (the entries of W[k*n+i, :] for all k)
            # Actually, for fixed i, we have p equations and p*target_rank unknowns.
            
            # This is getting complex. Let me try a simpler approach: 
            # Use the Burer-Monteiro style: fix one factor, solve for others via linear algebra.
            
            # Simpler approach: Use gradient-based but with better initialization.
            # Initialize by taking the first target_rank schoolbook columns and adding noise.
            U = U_school[:, :target_rank].copy()
            V = V_school[:, :target_rank].copy()
            W = W_school[:, :target_rank].copy()
            
            # Add small random perturbation to break symmetry
            U += np.random.randn(n * m, target_rank) * 0.01
            V += np.random.randn(m * p, target_rank) * 0.01
            W += np.random.randn(p * n, target_rank) * 0.01
            
            # Run ALS iterations
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Solve for U given V and W
                # T[i,j,k] = sum_r U[i*m+j, r] * D[i,j,k,r] where D[i,j,k,r] = V[j*p+k,r]*W[k*n+i,r]
                # For fixed (j,k): T[i,j,k] = sum_r U[i*m+j, r] * D[i,j,k,r]
                # We want T[i,j,k] = 1 for all i
                # This is: sum_r U[i*m+j, r] * D[i,j,k,r] = 1 for all i
                # For fixed (j,k), this is n equations in n*target_rank unknowns (U[i*m+j,:] for all i)
                
                # Let's solve for U by minimizing sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                # Gradient w.r.t U[i*m+j, r] = sum_k (T[i,j,k]-1) * V[j*p+k,r] * W[k*n+i,r]
                
                # Use a simple gradient step with line search
                lr = 0.01
                for _ in range(2):  # A few gradient steps per ALS iteration
                    # Compute gradients
                    T = compute_tensor(U, V, W)
                    err = T - 1.0  # (60,)
                    
                    # dU[i*m+j, r] = sum_k err[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                    # We need to map err back to (i,j,k)
                    err_3d = err.reshape(n, m, p)
                    
                    dU = np.zeros_like(U)
                    dV = np.zeros_like(V)
                    dW = np.zeros_like(W)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                e = err_3d[i, j, k]
                                if abs(e) < 1e-10:
                                    continue
                                idx_a = i * m + j
                                idx_b = j * p + k
                                idx_c = k * n + i
                                dU[idx_a, :] += e * V[idx_b, :] * W[idx_c, :]
                                dV[idx_b, :] += e * U[idx_a, :] * W[idx_c, :]
                                dW[idx_c, :] += e * U[idx_a, :] * V[idx_b, :]
                    
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                
                # Check convergence
                err_val = compute_error(U, V, W)
                if err_val < 1e-8:
                    # Found a valid decomposition
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                    found = True
                    break
            if found:
                break
        
        if found:
            break
    
    # Convert to half-integer if possible
    def to_half_int(mat):
        mat2 = mat * 2
        mat_int = np.round(mat2).astype(int)
        if np.max(np.abs(mat_int - mat2)) < 1e-6:
            return mat_int / 2.0
        return mat

    best_U = to_half_int(best_U)
    best_V = to_half_int(best_V)
    best_W = to_half_int(best_W)

    if best_rank == rank_school:
        best_U, best_V, best_W = U_school, V_school, W_school

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
