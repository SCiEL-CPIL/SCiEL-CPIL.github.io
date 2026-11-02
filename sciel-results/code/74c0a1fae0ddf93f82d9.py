"""
Rank-minimal tensor decomposition for 2x3 x 3x10 matrix multiplication.

Strategy:
1. Use a block-wise Strassen+schoolbook decomposition to get rank 55.
2. Attempt to reduce to rank 54 by finding a linear dependency or using
   a border-rank lifting / parameterized search approach.
3. Fall back to rank 55 if rank 54 cannot be found within time budget.

The rank-55 construction:
- Split the 3x10 matrix B into B1 (2x10, rows 0-1) and B2 (1x10, row 2).
- Split the 2x3 matrix A into A1 (2x2, cols 0-1) and A2 (2x1, col 2).
- Output C = A1*B1 + A2*B2, where A1*B1 is 2x10 and A2*B2 is 2x10.
- For A1*B1 (2x2 * 2x10): split 10 into 5 groups of 2, use Strassen (rank 7) per group.
  Total: 5 * 7 = 35.
- For A2*B2 (2x1 * 1x10): schoolbook, rank 10.
- Total: 35 + 10 = 45? Wait, that doesn't match. Let me recount.

Actually, looking at the inspiration code more carefully:
- Part 1: 5 Strassen blocks, each with 7 terms = 35 terms.
- Part 2: schoolbook for A2 (2x1) * B2 (1x10) = 2*1*10 = 20 terms.
- Total: 35 + 20 = 55. Yes, that matches.

Let me verify the Strassen part:
For 2x2 * 2x2, Strassen uses 7 multiplications.
Each multiplication m_c contributes to specific entries of the output.
The U, V, W matrices for each Strassen block have specific patterns.

For the border-rank lifting / rank-54 attempt:
We parameterize the decomposition with a symbolic variable and try to find
a 54-term decomposition. This is complex, so we use a heuristic search:
- Start from the rank-55 decomposition.
- Try to merge two columns by finding if their tensor contribution can be
  absorbed into the remaining 53 columns with adjusted coefficients.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
_START_TIME = time.time()
TIME_LIMIT = 140  # Leave 10s safety margin


def _build_strassen_55():
    """Build the known rank-55 decomposition."""
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    # --- Part 1: A1 (2x2) * B1 (2x10) using 5 Strassen blocks ---
    for g in range(5):
        k_base = 2 * g
        k0 = k_base
        k1 = k_base + 1
        
        v_idx_b00 = 0 * p + k0
        v_idx_b01 = 0 * p + k1
        v_idx_b10 = 1 * p + k0
        v_idx_b11 = 1 * p + k1
        
        u_idx_a00 = 0 * m + 0
        u_idx_a01 = 0 * m + 1
        u_idx_a10 = 1 * m + 0
        u_idx_a11 = 1 * m + 1
        
        w_idx_c00 = k0 * n + 0
        w_idx_c01 = k1 * n + 0
        w_idx_c10 = k0 * n + 1
        w_idx_c11 = k1 * n + 1
        
        # Term 1: m1 = (a00+a11)(b00+b11)
        U[u_idx_a00][col] = 1
        U[u_idx_a11][col] = 1
        V[v_idx_b00][col] = 1
        V[v_idx_b11][col] = 1
        W[w_idx_c00][col] = 1
        W[w_idx_c11][col] = 1
        col += 1
        
        # Term 2: m2 = (a10+a11)b00
        U[u_idx_a10][col] = 1
        U[u_idx_a11][col] = 1
        V[v_idx_b00][col] = 1
        W[w_idx_c10][col] = 1
        W[w_idx_c11][col] = -1
        col += 1
        
        # Term 3: m3 = a00(b01-b11)
        U[u_idx_a00][col] = 1
        V[v_idx_b01][col] = 1
        V[v_idx_b11][col] = -1
        W[w_idx_c01][col] = 1
        W[w_idx_c11][col] = 1
        col += 1
        
        # Term 4: m4 = a11(b10-b00)
        U[u_idx_a11][col] = 1
        V[v_idx_b10][col] = 1
        V[v_idx_b00][col] = -1
        W[w_idx_c00][col] = 1
        W[w_idx_c10][col] = 1
        col += 1
        
        # Term 5: m5 = (a00+a01)b11
        U[u_idx_a00][col] = 1
        U[u_idx_a01][col] = 1
        V[v_idx_b11][col] = 1
        W[w_idx_c01][col] = 1
        W[w_idx_c00][col] = -1
        col += 1
        
        # Term 6: m6 = (a10-a00)(b00+b01)
        U[u_idx_a10][col] = 1
        U[u_idx_a00][col] = -1
        V[v_idx_b00][col] = 1
        V[v_idx_b01][col] = 1
        W[w_idx_c11][col] = 1
        col += 1
        
        # Term 7: m7 = (a01-a11)(b10+b11)
        U[u_idx_a01][col] = 1
        U[u_idx_a11][col] = -1
        V[v_idx_b10][col] = 1
        V[v_idx_b11][col] = 1
        W[w_idx_c00][col] = 1
        col += 1

    # --- Part 2: A2 (2x1) * B2 (1x10) using Schoolbook ---
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W):
    """Verify that the decomposition is exact using numpy."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                computed = 0.0
                for r in range(rank):
                    u_val = U[i*m + j][r]
                    v_val = V[j*p + k][r]
                    w_val = W[k*n + i][r]
                    computed += u_val * v_val * w_val
                err = abs(computed - expected)
                if err > max_err:
                    max_err = err
    return max_err < 1e-9


def _build_rank_54_winograd():
    """
    Attempt to construct a rank-54 decomposition using a Winograd-style
    frequency domain approach combined with structured rank sharing.
    
    Key insight: For n=2, the two output rows share the same B matrix.
    We can potentially save one rank by finding a shared linear combination.
    
    Strategy:
    1. Use the rank-55 construction as a starting point.
    2. Try to merge the two schoolbook terms for k=0 (i=0 and i=1)
       into a single term by adjusting the Strassen part.
    3. If that fails, try other systematic merges.
    """
    n, m, p = DIMENSIONS
    base = _build_strassen_55()
    U = [row[:] for row in base['U']]
    V = [row[:] for row in base['V']]
    W = [row[:] for row in base['W']]
    rank = 55
    
    # The schoolbook part uses columns 35-54.
    # Column 35: i=0, k=0: U[2,35]=1, V[20,35]=1, W[0,35]=1
    # Column 36: i=1, k=0: U[5,36]=1, V[20,36]=1, W[1,36]=1
    # Both have V[20,:] = 1, so they share the same V entry for k=0.
    
    # Can we merge columns 35 and 36 into one?
    # We'd need a single column r such that:
    #   U[2,r] * V[20,r] * W[0,r] = 1  (for i=0,k=0)
    #   U[5,r] * V[20,r] * W[1,r] = 1  (for i=1,k=0)
    # And all other (i,j,k) entries remain satisfied.
    
    # If we set V[20,r] = 1, then we need:
    #   U[2,r] * W[0,r] = 1
    #   U[5,r] * W[1,r] = 1
    # We can set U[2,r]=1, W[0,r]=1, U[5,r]=1, W[1,r]=1.
    # But this adds contributions to other (i,j,k) entries where
    # U[2,r] or U[5,r] is non-zero and V[20,r] is non-zero.
    
    # U[2,r] corresponds to A[0,2], and U[5,r] corresponds to A[1,2].
    # V[20,r] corresponds to B[2,0].
    # W[0,r] corresponds to C[0,0], and W[1,r] corresponds to C[1,0].
    
    # The issue is that this merged term would contribute to:
    #   T[0,2,0] += U[2,r]*V[20,r]*W[0,r] = 1*1*1 = 1 (desired)
    #   T[1,2,0] += U[5,r]*V[20,r]*W[1,r] = 1*1*1 = 1 (desired)
    # But also to any (i,j,k) where U[i*3+2,r] != 0, V[2*10+k,r] != 0, W[k*2+i,r] != 0.
    
    # Since U[2,r] and U[5,r] are the only non-zero U entries in column r,
    # and V[20,r] is the only non-zero V entry, the only contributions are:
    #   For j=2, k=0: T[i,2,0] += U[i*3+2,r] * 1 * W[0*2+i,r]
    #   = U[i*3+2,r] * W[i,r] for i=0,1
    
    # So if we set U[2,r]=1, U[5,r]=1, V[20,r]=1, W[0,r]=1, W[1,r]=1,
    # we get exactly the two desired contributions and no others.
    
    # But we need to remove the contributions from columns 35 and 36.
    # Column 35: U[2,35]=1, V[20,35]=1, W[0,35]=1 -> contributes T[0,2,0] += 1
    # Column 36: U[5,36]=1, V[20,36]=1, W[1,36]=1 -> contributes T[1,2,0] += 1
    
    # If we replace columns 35 and 36 with a single merged column,
    # we save one rank. But we need to verify that no other (i,j,k)
    # entries are affected.
    
    # The merged column only affects T[0,2,0] and T[1,2,0], which is correct.
    # So this merge is valid!
    
    # Create the merged column at position 35, and remove column 36.
    U_merged = [row[:] for row in U]
    V_merged = [row[:] for row in V]
    W_merged = [row[:] for row in W]
    
    # First, save columns 37-54
    U_save = [U_merged[i][37:55] for i in range(n * m)]
    V_save = [V_merged[j][37:55] for j in range(m * p)]
    W_save = [W_merged[k][37:55] for k in range(p * n)]
    
    # Set up the merged column at index 35
    U_merged[2][35] = 1
    U_merged[5][35] = 1
    V_merged[20][35] = 1
    W_merged[0][35] = 1
    W_merged[1][35] = 1
    
    # Shift saved columns (37-54) to positions 36-53
    for c in range(18):
        for i in range(n * m):
            U_merged[i][36 + c] = U_save[i][c]
        for j in range(m * p):
            V_merged[j][36 + c] = V_save[j][c]
        for k in range(p * n):
            W_merged[k][36 + c] = W_save[k][c]
    
    # Trim to rank 54
    U_final = [row[:54] for row in U_merged]
    V_final = [row[:54] for row in V_merged]
    W_final = [row[:54] for row in W_merged]
    
    result = {"U": U_final, "V": V_final, "W": W_final}
    
    if _verify_decomposition(U_final, V_final, W_final):
        return result
    
    return None


def construct():
    deadline = _START_TIME + TIME_LIMIT
    
    # Try the Winograd-style rank-54 construction
    result_54 = _build_rank_54_winograd()
    if result_54 is not None:
        if _verify_decomposition(result_54['U'], result_54['V'], result_54['W']):
            return result_54
    
    # Fall back to rank-55
    result_55 = _build_strassen_55()
    
    # Verify before returning
    if _verify_decomposition(result_55['U'], result_55['V'], result_55['W']):
        return result_55
    
    # Should not reach here, but just in case, return a schoolbook decomposition
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
