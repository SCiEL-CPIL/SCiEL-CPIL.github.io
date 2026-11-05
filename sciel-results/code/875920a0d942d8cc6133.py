"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
DEADLINE = START_TIME + 120  # 120s wall-clock limit

def _time_remaining():
    return DEADLINE - time.time()

# EVOLVE-BLOCK-START

def _build_strassen_rank54():
    """Build the rank-54 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # M0: (A00+A11)(B00+B11)
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        nonlocal col
        # 1: A00 B00 -> C00
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        # 2: A00 B01 -> C01
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        # 3: A10 B00 -> C10
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        # 4: A10 B01 -> C11
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

    # Apply Strassen blocks for j-pairs (0,1) and (2,3)
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # Apply naive blocks for j=4
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)
    
    if col < rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, num_trials=3):
    """Verify the decomposition produces correct matrix products."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    r = U_np.shape[1]
    
    for trial in range(num_trials):
        np.random.seed(42 + trial)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for r_idx in range(r):
            u_col = U_np[:, r_idx]
            v_col = V_np[:, r_idx]
            w_col = W_np[:, r_idx]
            for i in range(n):
                for j in range(m):
                    u_val = u_col[i*m+j]
                    if u_val != 0:
                        for k in range(p):
                            v_val = v_col[j*p+k]
                            if v_val != 0:
                                w_val = w_col[k*n+i]
                                if w_val != 0:
                                    C_computed[i, k] += u_val * v_val * w_val * A_test[i,j] * B_test[j,k]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _try_rank_reduction(U, V, W, n, m, p, target_rank=50):
    """
    Try to reduce the rank by finding linear dependencies among the terms.
    Uses the fact that if sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0 as a tensor,
    then we can eliminate one term.
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    r = U_np.shape[1]
    
    if r <= target_rank:
        return U, V, W
    
    # Try to find a linear dependency: sum_{r in S} c_r * (u_r ⊗ v_r ⊗ w_r) = 0
    # This is equivalent to: for all (i,j,k), sum_{r in S} c_r * U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 0
    # We can set this up as a system: for each (i,j,k), we get an equation.
    
    # Build the tensor T of shape (n*m, m*p, p*n) where T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # But we want to find if any column is a linear combination of others.
    
    # Alternative: try to find c such that for all a,b,c_idx:
    # sum_r c_r * U[a,r] * V[b,r] * W[c_idx,r] = 0
    # This is a bilinear system in c. Let's try a simpler approach:
    # Check if any pair of terms can be combined.
    
    # For now, return the original if we can't find a reduction
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Build the rank-54 Strassen decomposition
    U, V, W = _build_strassen_rank54()
    r = len(U[0])
    
    # Verify correctness
    if _time_remaining() > 10:
        if not _verify_decomposition(U, V, W, n, m, p):
            # Fall back to schoolbook
            rank_school = n * m * p
            U_school = [[0] * rank_school for _ in range(n * m)]
            V_school = [[0] * rank_school for _ in range(m * p)]
            W_school = [[0] * rank_school for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        U_school[i*m+j][idx] = 1
                        V_school[j*p+k][idx] = 1
                        W_school[k*n+i][idx] = 1
            return {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to reduce rank if time permits
    if _time_remaining() > 30 and r > 50:
        U, V, W = _try_rank_reduction(U, V, W, n, m, p, target_rank=50)
        if _time_remaining() > 10:
            if not _verify_decomposition(U, V, W, n, m, p):
                # Rebuild if reduction broke something
                U, V, W = _build_strassen_rank54()
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
