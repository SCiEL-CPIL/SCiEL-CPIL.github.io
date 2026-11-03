import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Target: Find a decomposition for M_{2,3,3} with rank < 18
    # Then use it 3 times for M_{2,3,9}
    
    # We will try to find a rank R decomposition for M_{2,3,3}
    # R can be 14, 15, 16, 17
    # If found, total rank is 3*R
    
    # Solver for M_{2,3,3}
    def solve_m233(rank_target, max_iters=1000, timeout=10):
        # T_{i,j,k} = 1 for all i,j,k in M_{2,3,3}
        # Condition: sum_r W_{i,k}^r U_{i,j}^r V_{j,k}^r = delta_{j,j'}
        # This is equivalent to finding U, V, W such that
        # sum_r U^r \otimes V^r \otimes W^r approximates the tensor T where T_{i,j,k} is not just 1.
        # The tensor T for multiplication is defined by T_{i,j,k} = 1?
        # No, the tensor is M_{2,3,3}.
        # The entries of the tensor M_{n,m,p} are 1 at (i,j,k) ?
        # Let's check the schoolbook decomposition again.
        # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
        # This corresponds to the term A_{i,j} B_{j,k} contributing to C_{i,k}.
        # So the tensor T has T_{i,j,k} = 1?
        # If we contract U, V, W:
        # Sum_c U_{i,j}^c V_{j,k}^c W_{i,k}^c
        # In schoolbook, for a specific (i,j,k), there is one c where U=1, V=1, W=1.
        # So the sum is 1.
        # So yes, the tensor T_{i,j,k} = 1 for all i,j,k.
        
        # So we need to find U (2*3 x R), V (3*3 x R), W (2*3 x R)
        # such that sum_r U_{i,j}^r V_{j,k}^r W_{i,k}^r = 1 for all i,j,k.
        
        # This is a tensor decomposition problem.
        # We can use random initialization and gradient descent.
        
        start_time = time.time()
        np.random.seed(42)
        
        for attempt in range(10):
            U = np.random.randn(6, rank_target)
            V = np.random.randn(9, rank_target)
            W = np.random.randn(6, rank_target)
            
            # Normalize to avoid scale issues
            # Not strictly necessary if using least squares
            
            for it in range(max_iters):
                if time.time() - start_time > timeout:
                    return None
                
                # Compute current tensor approximation
                # T_approx_{i,j,k} = sum_r U_{i,j}^r V_{j,k}^r W_{i,k}^r
                # This is expensive to compute naively.
                # Use einsum?
                # T_approx = np.einsum('ir,jr,kr->ijk', U, V, W) ?
                # U is 6xR (flattened i,j). V is 9xR (flattened j,k). W is 6xR (flattened i,k).
                # Need to reshape.
                
                U_reshaped = U.reshape(2, 3, rank_target)
                V_reshaped = V.reshape(3, 3, rank_target)
                W_reshaped = W.reshape(2, 3, rank_target)
                
                T_approx = np.einsum('ijr,jkr,ikr->ijk', U_reshaped, V_reshaped, W_reshaped)
                
                # Target T is all ones
                T_target = np.ones((2, 3, 3))
                
                error = np.linalg.norm(T_approx - T_target)
                
                if error < 1e-6:
                    # Check if we can round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    U_r = U_rounded.reshape(2, 3, rank_target)
                    V_r = V_rounded.reshape(3, 3, rank_target)
                    W_r = W_rounded.reshape(2, 3, rank_target)
                    
                    T_check = np.einsum('ijr,jkr,ikr->ijk', U_r, V_r, W_r)
                    if np.allclose(T_check, T_target, atol=1e-9):
                        return U_rounded, V_rounded, W_rounded
                    
                    # If rounding failed, try to refine?
                    # For now, return None or continue
                    continue
                
                # Gradient descent step
                # Gradient of T_approx w.r.t U
                # d(T_approx)/dU_{ij}^r = V_{jk}^r W_{ik}^r (sum over k)
                # d(T_approx)/dU = np.einsum('jkr,ikr->ijr', V_reshaped, W_reshaped)
                # Similarly for V and W
                
                grad_U = np.einsum('jkr,ikr->ijr', V_reshaped, W_reshaped)
                grad_V = np.einsum('ijr,ikr->jkr', U_reshaped, W_reshaped)
                grad_W = np.einsum('ijr,jkr->ikr', U_reshaped, V_reshaped)
                
                # Error gradient
                # T_approx - T_target
                # grad_U_contrib = np.einsum('ijk,ikr->ijr', (T_approx - T_target), ?)
                # Actually, the derivative of sum (T_approx - T_target)^2 w.r.t U
                # is 2 * (T_approx - T_target) contracted with V, W.
                
                diff = T_approx - T_target
                
                # d/dU_{ij}^r sum_{i,j,k} (T_approx_{ijk} - 1)^2
                # = 2 sum_{i,j,k} (T_approx_{ijk} - 1) * d(T_approx_{ijk})/dU_{ij}^r
                # d(T_approx_{ijk})/dU_{ij'}^{r'} = delta_{ij,ij'} delta_{rr'} V_{j'k}^r W_{ik}^r ?
                # No, T_approx_{ijk} = sum_r U_{ij}^r V_{jk}^r W_{ik}^r.
                # d/dU_{ij}^r T_approx_{ijk} = V_{jk}^r W_{ik}^r.
                # So grad_U_{ij}^r = 2 sum_k (T_approx_{ijk} - 1) V_{jk}^r W_{ik}^r.
                
                grad_U_flat = 2 * np.einsum('ijk,jkr,ikr->ijr', diff, V_reshaped, W_reshaped)
                grad_V_flat = 2 * np.einsum('ijk,ijr,ikr->jkr', diff, U_reshaped, W_reshaped)
                grad_W_flat = 2 * np.einsum('ijk,ijr,jkr->ikr', diff, U_reshaped, V_reshaped)
                
                lr = 0.01
                U -= lr * grad_U_flat.reshape(6, rank_target)
                V -= lr * grad_V_flat.reshape(9, rank_target)
                W -= lr * grad_W_flat.reshape(6, rank_target)
                
        return None

    # Try to find decomposition for M_{2,3,3}
    # Start with lower ranks
    best_decomp = None
    for rank in [14, 15, 16, 17]:
        decomp = solve_m233(rank, max_iters=500, timeout=5)
        if decomp is not None:
            best_decomp = decomp
            break
            
    if best_decomp is not None:
        U_233, V_233, W_233 = best_decomp
        rank_233 = U_233.shape[1]
        total_rank = 3 * rank_233
        
        # Construct U, V, W for M_{2,3,9}
        # U is 6 x total_rank
        # V is 27 x total_rank
        # W is 18 x total_rank
        
        U = np.zeros((6, total_rank))
        V = np.zeros((27, total_rank))
        W = np.zeros((18, total_rank))
        
        # Copy 3 times
        for block in range(3):
            col_start = block * rank_233
            col_end = (block + 1) * rank_233
            
            # U is same for all blocks?
            # U_{i,j} depends on i,j.
            # In M_{2,3,9}, U_{i,j} is the same for all k blocks?
            # The term is A_{i,j} B_{j,k}.
            # If we use the same U for all blocks, then
            # sum_r W_{i,k}^r U_{i,j}^r V_{j,k}^r
            # For block b, k is in [3b, 3b+2].
            # We set V_{j, 3b+k'} = V_{233}[j, k']
            # We set W_{i, 3b+k'} = W_{233}[i, k']
            # We set U_{i, j} = U_{233}[i, j]
            
            U[:, col_start:col_end] = U_233
            
            # V is 27 x R. Flattened j (0..2), k (0..8).
            # V_{j, k} index in flat array: j*9 + k.
            # For block b, k_local = 0..2, k_global = 3b + k_local.
            # V_233 is 9 x R. Flattened j (0..2), k_local (0..2). Index j*3 + k_local.
            # We need to map V_233[j, k_local, r] to V[j, 3b+k_local, r].
            
            V_233_reshaped = V_233.reshape(3, 3, rank_233)
            for j in range(3):
                for k_local in range(3):
                    k_global = 3 * block + k_local
                    V[j * 9 + k_global, col_start:col_end] = V_233_reshaped[j, k_local, :]
                    
            # W is 18 x R. Flattened k (0..8), i (0..1).
            # W_{k, i} index in flat array: k*2 + i.
            # W_233 is 6 x R. Flattened k_local (0..2), i (0..1). Index k_local*2 + i.
            
            W_233_reshaped = W_233.reshape(3, 2, rank_233)
            for k_local in range(3):
                k_global = 3 * block + k_local
                for i in range(2):
                    W[k_global * 2 + i, col_start:col_end] = W_233_reshaped[k_local, i, :]
                    
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
