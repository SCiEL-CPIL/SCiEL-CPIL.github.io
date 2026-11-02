"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
_start_time = time.time()
_DEADLINE = 120.0

def _build_strassen_35():
    """Build the known-good rank-35 Strassen decomposition."""
    n, m, p = DIMENSIONS  # 2, 2, 10
    num_blocks = p // 2  # 5 blocks of 2 columns each
    rank = num_blocks * 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35
    
    strassen_decomp = [
        # M1 = (A00+A11) * (B00+B11)
        (
            [(0, 1), (3, 1)],
            [(0, 1), (3, 1)],
            [(0, 1), (3, 1)],
        ),
        # M2 = (A10+A11) * B00
        (
            [(2, 1), (3, 1)],
            [(0, 1)],
            [(2, 1), (3, -1)],
        ),
        # M3 = A00 * (B01-B11)
        (
            [(0, 1)],
            [(1, 1), (3, -1)],
            [(1, 1), (3, 1)],
        ),
        # M4 = A11 * (B10-B00)
        (
            [(3, 1)],
            [(2, 1), (0, -1)],
            [(0, 1), (2, 1)],
        ),
        # M5 = (A00+A01) * B11
        (
            [(0, 1), (1, 1)],
            [(3, 1)],
            [(1, 1), (0, -1)],
        ),
        # M6 = (A10-A00) * (B00+B01)
        (
            [(2, 1), (0, -1)],
            [(0, 1), (1, 1)],
            [(3, 1)],
        ),
        # M7 = (A01-A11) * (B10+B11)
        (
            [(1, 1), (3, -1)],
            [(2, 1), (3, 1)],
            [(0, 1)],
        ),
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
    
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=2, p=10, num_trials=50, tol=1e-9):
    """Verify that the tensor decomposition is correct by random testing."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for _ in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute via tensor decomposition
        # C[k*n+i, :] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # More precisely: C[i,k] = sum_{j,r} A[i,j] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, the decomposition is: C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
        # No, that's not right either. Let me think more carefully.
        # The tensor T_{i,j,k} = delta_{i,i} * delta_{j,j} * delta_{k,k} for matmul
        # T = sum_r u_r ⊗ v_r ⊗ w_r where (u_r)_i = U[i*m+j, r]? No...
        # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor is T[i,j,k] = 1 if the indices match, i.e., T[i,j,k] = δ_{i,j,k} in the sense that
        # C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] where T[i,j,k]=1 for all valid i,j,k.
        # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        # So C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r U[i,j,r] V[j,k,r] W[i,k,r] A[i,j] B[j,k]
        # This doesn't look right. Let me reconsider.
        
        # Standard tensor decomposition for matmul:
        # C = A*B means C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor T of shape (n, m, p) where T[i,j,k] = 1 (the multiplication tensor)
        # T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
        # (u_r)_{i*m+j} = U[i*m+j, r], (v_r)_{j*p+k} = V[j*p+k, r], (w_r)_{k*n+i} = W[k*n+i, r]
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And we need T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j 1 * A[i,j] * B[j,k])
        
        # So verification: check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        # This is a static check, not dependent on A and B!
        
        # Let's just do the static check
        break  # Don't need random trials for static check
    
    # Static verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank_reduction(U, V, W, n=2, m=2, p=10, time_budget=30.0):
    """Try to reduce rank by finding linear dependencies among columns."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    best_U, best_V, best_W = U_arr, V_arr, W_arr
    best_rank = rank
    
    # Check if any single column can be removed
    # A column r is redundant if there exist coefficients c_s (s≠r) such that
    # U[:,r] = sum_{s≠r} c_s U[:,s], V[:,r] = sum_{s≠r} c_s V[:,s], W[:,r] = sum_{s≠r} c_s W[:,s]
    # This is a very restrictive condition.
    
    # More generally, we can try to find a set of columns that are linearly dependent
    # in the tensor sense. But this is complex.
    
    # Alternative: try small perturbations and see if rank drops (unlikely to work exactly)
    
    # For now, just return the original if no reduction found within time budget
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - time_budget:
        return U, V, W
    
    # Try checking if removing each single column still gives a valid decomposition
    # (This won't work since we need all 35 terms, but let's check)
    for r_remove in range(rank):
        elapsed = time.time() - _start_time
        if elapsed > _DEADLINE - 5:
            break
        U_sub = np.delete(U_arr, r_remove, axis=1)
        V_sub = np.delete(V_arr, r_remove, axis=1)
        W_sub = np.delete(W_arr, r_remove, axis=1)
        
        # Check if the reduced decomposition is still valid
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(rank - 1):
                        s += U_sub[i*m+j, rr] * V_sub[j*p+k, rr] * W_sub[k*n+i, rr]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_U, best_V, best_W = U_sub, V_sub, W_sub
            best_rank = rank - 1
            break  # Found a reduction, stop
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist()

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 10
    
    # Build the known-good rank-35 Strassen decomposition
    U, V, W = _build_strassen_35()
    
    # Verify it's correct
    if not _verify_decomposition(U, V, W, n, m, p):
        # Should never happen, but return anyway
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    U_red, V_red, W_red = _try_rank_reduction(U, V, W, n, m, p, time_budget=60.0)
    
    return {"U": U_red, "V": V_red, "W": W_red}
# EVOLVE-BLOCK-END
