import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Base solution: 50-rank Strassen-based decomposition
    def generate_base_solution():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_strassen_block(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            
            c00_idx = k0 * n + i0
            c01_idx = (k0 + 1) * n + i0
            c10_idx = k0 * n + (i0 + 1)
            c11_idx = (k0 + 1) * n + (i0 + 1)
            
            # M1 = (a+d)(e+h)
            U[a_idx][col] += 1
            U[d_idx][col] += 1
            V[e_idx][col] += 1
            V[h_idx][col] += 1
            W[c00_idx][col] += 1
            W[c11_idx][col] += 1
            col += 1
            
            # M2 = (c+d)e
            U[c_idx][col] += 1
            U[d_idx][col] += 1
            V[e_idx][col] += 1
            W[c10_idx][col] += 1
            W[c11_idx][col] -= 1
            col += 1
            
            # M3 = a(f-h)
            U[a_idx][col] += 1
            V[f_idx][col] += 1
            V[h_idx][col] -= 1
            W[c01_idx][col] += 1
            W[c11_idx][col] += 1
            col += 1
            
            # M4 = d(g-e)
            U[d_idx][col] += 1
            V[g_idx][col] += 1
            V[e_idx][col] -= 1
            W[c00_idx][col] += 1
            W[c10_idx][col] += 1
            col += 1
            
            # M5 = (a+b)h
            U[a_idx][col] += 1
            U[b_idx][col] += 1
            V[h_idx][col] += 1
            W[c00_idx][col] -= 1
            W[c01_idx][col] += 1
            col += 1
            
            # M6 = (c-a)(e+f)
            U[c_idx][col] += 1
            U[a_idx][col] -= 1
            V[e_idx][col] += 1
            V[f_idx][col] += 1
            W[c11_idx][col] += 1
            col += 1
            
            # M7 = (b-d)(g+h)
            U[b_idx][col] += 1
            U[d_idx][col] -= 1
            V[g_idx][col] += 1
            V[h_idx][col] += 1
            W[c00_idx][col] += 1
            col += 1

        def add_schoolbook_221_block(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            
            e_idx = j0 * p + k0
            g_idx = (j0 + 1) * p + k0
            
            c00_idx = k0 * n + i0
            c10_idx = k0 * n + (i0 + 1)
            
            U[a_idx][col] += 1
            V[e_idx][col] += 1
            W[c00_idx][col] += 1
            col += 1
            
            U[b_idx][col] += 1
            V[g_idx][col] += 1
            W[c00_idx][col] += 1
            col += 1
            
            U[c_idx][col] += 1
            V[e_idx][col] += 1
            W[c10_idx][col] += 1
            col += 1
            
            U[d_idx][col] += 1
            V[g_idx][col] += 1
            W[c10_idx][col] += 1
            col += 1

        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                add_strassen_block(0, j_start, k_start)
            add_schoolbook_221_block(0, j_start, 6)
            
        return {"U": U, "V": V, "W": W}

    # Kronecker-product structured search for lower rank
    def try_kronecker_search(max_rank=49, time_limit=90):
        """
        Try to find a decomposition with rank < 50 using a Kronecker-product
        structured ansatz. We model the multiplication as a sum of rank-1 terms
        derived from tensor products of smaller component tensors.
        """
        start_time = time.time()
        
        # The tensor T[i, k, j] = 1 for all i, j, k in the multiplication tensor
        # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
        # T = sum_r U_r ⊗ V_r ⊗ W_r (outer products)
        
        # Use a structured search: try different Kronecker decompositions
        # Split m=4 into 2+2, and p=7 into 2+2+2+1
        # Try to find shared factors between blocks
        
        # Approach: Randomized search with structured initialization
        # Initialize with the 50-rank solution and try to reduce rank
        
        base_sol = generate_base_solution()
        U_base = np.array(base_sol['U'], dtype=float)
        V_base = np.array(base_sol['V'], dtype=float)
        W_base = np.array(base_sol['W'], dtype=float)
        
        # Try ranks from max_rank down to 45
        for target_rank in range(max_rank, 44, -1):
            if time.time() - start_time > time_limit:
                break
            
            # Try multiple random restarts
            for restart in range(50):
                if time.time() - start_time > time_limit:
                    break
                
                # Initialize with random factors
                U = np.random.randn(n * m, target_rank) * 0.5
                V = np.random.randn(m * p, target_rank) * 0.5
                W = np.random.randn(p * n, target_rank) * 0.5
                
                # Gradient descent to minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
                # T[i, k, j] = 1 for all i, k, j
                # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
                
                # Compute the tensor T
                T = np.ones((n, p, m))
                
                # Gradient descent
                lr = 0.01
                for step in range(200):
                    if time.time() - start_time > time_limit:
                        break
                    
                    # Compute current tensor
                    T_current = np.zeros((n, p, m))
                    for r in range(target_rank):
                        T_current += np.einsum('i,j,k->ijk', 
                                               U[:, r].reshape(n, m), 
                                               V[:, r].reshape(m, p), 
                                               W[:, r].reshape(p, n))
                    
                    # Error
                    err = T_current - T
                    err_norm = np.linalg.norm(err)
                    
                    if err_norm < 1e-6:
                        # Found a valid decomposition
                        # Round to nearest half-integer
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        
                        # Verify
                        T_check = np.zeros((n, p, m))
                        for r in range(target_rank):
                            T_check += np.einsum('i,j,k->ijk', 
                                                 U_rounded[:, r].reshape(n, m), 
                                                 V_rounded[:, r].reshape(m, p), 
                                                 W_rounded[:, r].reshape(p, n))
                        
                        if np.allclose(T_check, T, atol=1e-6):
                            return {
                                "U": U_rounded.tolist(),
                                "V": V_rounded.tolist(),
                                "W": W_rounded.tolist()
                            }
                    
                    # Gradient of loss with respect to U, V, W
                    # Loss = sum_{i,k,j} (T_current[i,k,j] - T[i,k,j])^2
                    # dLoss/dU[i*m+j, r] = 2 * sum_{i,k,j} err[i,k,j] * V[j*p+k, r] * W[k*n+i, r]
                    
                    grad_U = np.zeros_like(U)
                    grad_V = np.zeros_like(V)
                    grad_W = np.zeros_like(W)
                    
                    for i in range(n):
                        for k in range(p):
                            for j in range(m):
                                e = err[i, k, j]
                                for r in range(target_rank):
                                    grad_U[i * m + j, r] += 2 * e * V[j * p + k, r] * W[k * n + i, r]
                                    grad_V[j * p + k, r] += 2 * e * U[i * m + j, r] * W[k * n + i, r]
                                    grad_W[k * n + i, r] += 2 * e * U[i * m + j, r] * V[j * p + k, r]
                    
                    # Update
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
        
        return None

    # Try the Kronecker search
    better_sol = try_kronecker_search(max_rank=49, time_limit=90)
    
    if better_sol is not None:
        return better_sol
    
    # Fall back to base solution
    return generate_base_solution()
