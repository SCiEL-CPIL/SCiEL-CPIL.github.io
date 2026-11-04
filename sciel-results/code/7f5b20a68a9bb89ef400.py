"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build initial rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(i_start, j_start, k_start, offset):
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 1
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] -= 1
        
        col = offset + 2
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] -= 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        
        col = offset + 4
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        
        col = offset + 5
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+0)][col] -= 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1
        
        col = offset + 6
        U[(i_start+0)*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] -= 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1

    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 2, 7)
    add_strassen_block(0, 0, 4, 14)
    add_strassen_block(0, 2, 0, 21)
    add_strassen_block(0, 2, 2, 28)
    add_strassen_block(0, 2, 4, 35)
    
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify the decomposition is correct
    def verify_decomp(U_arr, V_arr, W_arr):
        """Check that sum_k U[:,k] * V[:,k] * W[:,k] equals the matmul tensor."""
        # T[i,j,k] = sum_k U[i,k]*V[j,k]*W[k',i] where k' is output index
        # Actually: C[i,k_out] = sum_j A[i,j]*B[j,k_out]
        # Tensor: T[i,j,k] = 1 if this is the matmul tensor
        # Decomposition: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # But W is indexed by (p*n) = k*n + i in the trace convention
        # So W index: W[k_out * n + i, r]
        # Let's verify: for each (i, j, k_out), check sum_r U[i*m+j, r] * V[j*p+k_out, r] * W[k_out*n+i, r] == 1
        dim_in = n * m  # 8
        dim_mid = m * p  # 24
        dim_out = p * n  # 12
        rank_test = U_arr.shape[1]
        
        # Build the full tensor from decomposition
        T_decomp = np.zeros((n, m, p))
        for r in range(rank_test):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_decomp[i,j,k] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
        
        # The matmul tensor should be all 1s (since C[i,k] = sum_j A[i,j]*B[j,k])
        T_expected = np.ones((n, m, p))
        return np.allclose(T_decomp, T_expected, atol=1e-10)
    
    if not verify_decomp(U_np, V_np, W_np):
        # Fallback: return the original decomposition
        return {"U": U, "V": V, "W": W}
    
    # Peeling algorithm: try to find rank-1 terms that can be removed
    # A rank-1 term r can be removed if there exist vectors u', v', w' such that
    # sum_{r' != r} U[:,r']*V[:,r']*W[:,r'] = sum_{r' != r, r' != r_new} ... 
    # More practically: try to express column r as a linear combination of others
    # in the tensor product space.
    
    # The tangent space approach: a term r is redundant if the rank-1 tensor
    # U[:,r] ⊗ V[:,r] ⊗ W[:,r] lies in the span of the other rank-1 tensors.
    
    # We'll use a simpler approach: try removing each term and check if the
    # remaining decomposition still works (with possible rescaling).
    
    best_rank = rank
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    
    # Try peeling: for each column, check if removing it and adjusting others works
    # This is expensive, so we limit iterations
    max_peel_attempts = 50
    peel_count = 0
    
    current_rank = rank
    U_cur = U_np.copy()
    V_cur = V_np.copy()
    W_cur = W_np.copy()
    
    for attempt in range(max_peel_attempts):
        if time.time() > deadline - 10:
            break
        
        found_pell = False
        
        for r in range(current_rank):
            if time.time() > deadline - 10:
                break
            
            # Try removing column r
            # Check if the rank-1 tensor U[:,r]⊗V[:,r]⊗W[:,r] is in the span
            # of the remaining tensors.
            
            # Build matrix of remaining rank-1 tensors flattened
            # Each rank-1 tensor has size n*m*p = 48
            remaining = [rr for rr in range(current_rank) if rr != r]
            
            if len(remaining) < 1:
                break
            
            # Flatten each remaining rank-1 tensor
            # T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Size: n*m*p = 2*4*6 = 48
            
            # Build the target tensor (the one we want to remove)
            target = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target[idx] = U_cur[i*m+j, r] * V_cur[j*p+k, r] * W_cur[k*n+i, r]
                        idx += 1
            
            # Build matrix of remaining tensors
            n_remaining = len(remaining)
            M = np.zeros((n*m*p, n_remaining))
            for col_idx, rr in enumerate(remaining):
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            M[idx, col_idx] = U_cur[i*m+j, rr] * V_cur[j*p+k, rr] * W_cur[k*n+i, rr]
                            idx += 1
            
            # Check if target is in the column space of M
            # Solve M @ x = target, check residual
            try:
                # Use least squares
                x, residuals, rank_M, sv = np.linalg.lstsq(M, target, rcond=1e-10)
                residual = np.linalg.norm(target - M @ x)
                
                # If residual is very small, the term is redundant
                norm_target = np.linalg.norm(target)
                if norm_target > 1e-15 and residual / norm_target < 1e-8:
                    # Remove this term
                    U_cur = np.delete(U_cur, r, axis=1)
                    V_cur = np.delete(V_cur, r, axis=1)
                    W_cur = np.delete(W_cur, r, axis=1)
                    current_rank -= 1
                    found_pell = True
                    peel_count += 1
                    break
            except Exception:
                continue
        
        if not found_pell:
            break
    
    # Check if we improved
    if current_rank < best_rank:
        best_rank = current_rank
        best_U, best_V, best_W = U_cur, V_cur, W_cur
    
    # Verify the best decomposition
    if verify_decomp(best_U, best_V, best_W):
        # Convert back to lists with integer/half-integer entries if possible
        def to_half_integer_list(arr):
            result = []
            for row in arr:
                new_row = []
                for val in row:
                    # Check if close to half-integer
                    half_int = round(val * 2) / 2
                    if abs(val - half_int) < 1e-10:
                        new_row.append(half_int if half_int != int(half_int) else int(half_int))
                    else:
                        new_row.append(val)
                result.append(new_row)
            return result
        
        return {"U": to_half_integer_list(best_U), 
                "V": to_half_integer_list(best_V), 
                "W": to_half_integer_list(best_W)}
    else:
        # Fallback to original
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
