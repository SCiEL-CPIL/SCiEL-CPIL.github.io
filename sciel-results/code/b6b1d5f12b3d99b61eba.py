"""Matrix multiplication 2x2x10: rank-minimal exact tensor decomposition.

Strategy: Attempt to find a rank-30 decomposition by solving the tensor
decomposition equations numerically, falling back to the rank-35 Strassen
baseline if no lower-rank solution is found.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 10]
_DEADLINE = 120.0


def _build_strassen_baseline():
    """Build the safe rank-35 Strassen baseline."""
    n, m, p = 2, 2, 10
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_decomp = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    for b in range(num_blocks):
        base_col = b * 7
        for t in range(7):
            col = base_col + t
            u_side, v_side, w_side = strassen_decomp[t]
            for u_row, u_coeff in u_side:
                U[u_row][col] = u_coeff
            for v_koff, v_coeff in v_side:
                j = v_koff // 2
                k = 2 * b + v_koff % 2
                V[j * p + k][col] = v_coeff
            for w_koff, w_coeff in w_side:
                i = w_koff // 2
                k = 2 * b + w_koff % 2
                W[k * n + i][col] = w_coeff
    
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with the safe baseline
    U, V, W, rank = _build_strassen_baseline()
    
    # Try to find a lower-rank decomposition
    # The tensor T[i,j,k] = 1 if (i,j) pair contributes to output k
    # T is 4 x 20 x 20
    # We want U (4xr), V (20xr), W (20xr) such that T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    
    # For rank r, we have 4r + 20r + 20r = 44r unknowns
    # And 4*20*20 = 1600 equations
    # So r >= 1600/44 ≈ 36.4, so r >= 37? That can't be right.
    # Wait, the tensor is sparse. Let me count non-zero entries.
    
    # T[i,j,k] = 1 if output C[i,k] depends on A[i//2, i%2] and B[j//10, j%10]
    # Actually, let me think about this differently.
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[A_idx, B_idx, C_idx] = 1 where:
    #   A_idx = i*m + j (for A[i,j])
    #   B_idx = j*p + k (for B[j,k])  
    #   C_idx = k*n + i (for C[i,k])
    # And the condition is that A[i,j]*B[j,k] contributes to C[i,k]
    
    # So for each (i,j,k): T[i*m+j, j*p+k, k*n+i] = 1
    # That's 2*2*10 = 40 non-zero entries.
    
    # With 40 equations and 44r unknowns, for r=10 we have 440 unknowns vs 40 equations.
    # So it's very underdetermined. The question is whether a solution exists with
    # the additional constraint that the decomposition is exact.
    
    # Let me try to find a rank-30 solution using least squares.
    
    try:
        for target_rank in [30, 31, 32, 33, 34]:
            if time.time() - start_time > _DEADLINE - 30:
                break
            
            # Build the tensor
            T = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i*m+j, j*p+k, k*n+i] = 1.0
            
            # Try to find U, V, W with rank target_rank
            # This is a non-convex optimization problem.
            # Use alternating least squares.
            
            rng = np.random.default_rng(42 + target_rank)
            U_init = rng.normal(size=(n*m, target_rank))
            V_init = rng.normal(size=(m*p, target_rank))
            W_init = rng.normal(size=(p*n, target_rank))
            
            # Normalize
            U_init /= np.linalg.norm(U_init)
            V_init /= np.linalg.norm(V_init)
            W_init /= np.linalg.norm(W_init)
            
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            converged = False
            for iteration in range(100):
                if time.time() - start_time > _DEADLINE - 30:
                    break
                
                # Update U: for each r, U[:,r] should satisfy
                # sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
                # Fix V and W, solve for U.
                # For each r: U[:,r] * (V[:,r] outer W[:,r]) = T
                # This is a linear system.
                
                # Vectorize: reshape T to (n*m, m*p*p*n)
                T_vec = T.reshape(n*m, -1)
                
                # For each r, the contribution is U[:,r] * (V[:,r] * W[:,r])^T
                # where V[:,r] * W[:,r] is the outer product reshaped.
                
                # Build the system: for each r, A_r = V[:,r] outer W[:,r] reshaped to (m*p, p*n)
                # Then T = sum_r U[:,r] * A_r.flatten()
                
                # This is a linear system in U.
                # Let's build it properly.
                
                # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
                # For fixed i, j, k: T[i,j,k] = sum_r U[i,r] * (V[j,r] * W[k,r])
                
                # So for fixed i: T[i, :, :] = sum_r U[i,r] * (V[:,r] * W[:,r])
                # where V[:,r] * W[:,r] is the outer product.
                
                # Let B_r = V[:,r] outer W[:,r], shape (m*p, p*n)
                # Then T[i, :, :] = sum_r U[i,r] * B_r
                # Vectorizing: T[i, :].flatten() = sum_r U[i,r] * B_r.flatten()
                
                # So for each i: T[i].flatten() = U[i, :] @ [B_0.flatten(), B_1.flatten(), ...]
                
                # Build the matrix M where M[j,k] = sum_r V[j,r] * W[k,r]
                # No wait, that's not right.
                
                # Let me think again. For fixed i:
                # T[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
                # Let's define for each r: C_r[j,k] = V[j,r] * W[k,r]
                # Then T[i,j,k] = sum_r U[i,r] * C_r[j,k]
                
                # So T[i] = sum_r U[i,r] * C_r (as a matrix)
                # Vectorizing: T[i].flatten() = sum_r U[i,r] * C_r.flatten()
                
                # So for each i, we have: T[i].flatten() = U[i,:] @ C_flattened
                # where C_flattened is a (target_rank, m*p*p*n) matrix with row r = C_r.flatten()
                
                C_mat = np.einsum('jr,kr->rjk', V_curr, W_curr)  # (r, m*p, p*n)
                C_flat = C_mat.reshape(target_rank, -1)  # (r, m*p*p*n)
                
                # Solve U[i,:] @ C_flat.T = T[i].flatten()
                # U[i,:] = T[i].flatten() @ C_flat.T @ (C_flat @ C_flat.T)^{-1}
                
                try:
                    U_new = T_vec @ C_flat.T @ np.linalg.pinv(C_flat @ C_flat.T)
                    U_curr = U_new
                except:
                    pass
                
                # Similarly update V
                C_mat = np.einsum('ir,kr->rik', U_curr, W_curr)  # (r, n*m, p*n)
                C_flat = C_mat.reshape(target_rank, -1)
                V_new = T.transpose(1, 0, 2).reshape(m*p, -1) @ C_flat.T @ np.linalg.pinv(C_flat @ C_flat.T)
                V_curr = V_new.reshape(m*p, target_rank)
                
                # Similarly update W
                C_mat = np.einsum('ir,jr->rij', U_curr, V_curr)  # (r, n*m, m*p)
                C_flat = C_mat.reshape(target_rank, -1)
                W_new = T.transpose(2, 0, 1).reshape(p*n, -1) @ C_flat.T @ np.linalg.pinv(C_flat @ C_flat.T)
                W_curr = W_new.reshape(p*n, target_rank)
                
                # Check convergence
                T_recon = np.einsum('ir,jr,kr->ijk', U_curr, V_curr, W_curr)
                err = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
                
                if err < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Found a lower-rank decomposition!
                rank = target_rank
                U = U_curr.tolist()
                V = V_curr.tolist()
                W = W_curr.tolist()
                break
    
    except Exception:
        pass
    
    # Verify the result
    try:
        rng = np.random.default_rng(42)
        A_test = rng.integers(-5, 6, size=(n, m))
        B_test = rng.integers(-5, 6, size=(m, p))
        C_expected = A_test @ B_test
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        C_computed = np.zeros(p * n)
        for r in range(rank):
            uA = np.dot(U_arr[:, r], A_flat)
            vB = np.dot(V_arr[:, r], B_flat)
            C_computed += uA * vB * W_arr[:, r]
        
        C_expected_reorder = np.zeros(p * n)
        for i in range(n):
            for k in range(p):
                C_expected_reorder[k * n + i] = C_expected[i, k]
        
        if not np.allclose(C_computed, C_expected_reorder, atol=1e-9):
            # Fall back to baseline
            U, V, W, rank = _build_strassen_baseline()
    except Exception:
        U, V, W, rank = _build_strassen_baseline()
    
    return {"U": U, "V": V, "W": W}
