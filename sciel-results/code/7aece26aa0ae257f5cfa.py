"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # safety margin before 150s kill
    
    num_U, num_V, num_W = n * m, m * p, p * n  # 20, 30, 24
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((num_U, rank_school), dtype=int)
    V_school = np.zeros((num_V, rank_school), dtype=int)
    W_school = np.zeros((num_W, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_factors = (U_school, V_school, W_school)
    best_rank = rank_school

    # Use Strassen's algorithm for 2x2 blocks to reduce rank.
    # Partition:
    # I: [0,1] -> I1, [2,3] -> I2
    # J: [0,1] -> J1, [2,3] -> J2, [4] -> J3
    # K: [0,1] -> K1, [2,3] -> K2, [4,5] -> K3
    
    # For each pair of I-block and K-block, we have a 2x5 * 5x2 multiplication.
    # This can be decomposed into:
    # - 2x2 * 2x2 (for J1, J2) -> Rank 7 each (Strassen)
    # - 2x1 * 1x2 (for J3)     -> Rank 4 (Schoolbook, since Strassen doesn't apply to 1D)
    # Total per (I,K) block pair: 7 + 7 + 4 = 18.
    # There are 2 I-blocks and 3 K-blocks, so 6 pairs.
    # Total Rank = 6 * 18 = 108.
    
    rank_structured = 108
    U_struct = np.zeros((num_U, rank_structured), dtype=int)
    V_struct = np.zeros((num_V, rank_structured), dtype=int)
    W_struct = np.zeros((num_W, rank_structured), dtype=int)
    
    def add_strassen_2x2(i_off, j_off, k_off, start_col):
        """
        Adds Strassen decomposition for 2x2 * 2x2.
        A: rows i_off..i_off+1, cols j_off..j_off+1
        B: rows j_off..j_off+1, cols k_off..k_off+1
        C: rows i_off..i_off+1, cols k_off..k_off+1
        Returns 7.
        """
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        # Strassen terms (rank 7)
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map these to U, V, W columns.
        # Each M is a product of a linear combo of A entries and a linear combo of B entries.
        # The W part collects the contributions to C entries.
        
        # Let's define the 7 terms explicitly.
        # Term t corresponds to U vector u_t, V vector v_t, W vector w_t.
        # The contribution is sum_t (u_t . A) * (v_t . B) * (w_t . C)
        # But we are building the factorization T = sum_r U_r V_r W_r.
        # So for each term r, we set U[:, r], V[:, r], W[:, r].
        
        # Term 1: M1
        # A part: A11 + A22
        # B part: B11 + B22
        # C part: C11 + C22 (since C11 has +M1, C22 has +M1)
        # Wait, C11 = M1 + ..., C22 = M1 + ...
        # So W has 1 at C11 and 1 at C22.
        
        # Let's fill the columns.
        
        # Term 1
        col = start_col
        U_struct[u_idx(i_off, j_off), col] += 1
        U_struct[u_idx(i_off+1, j_off+1), col] += 1
        V_struct[v_idx(j_off, k_off), col] += 1
        V_struct[v_idx(j_off+1, k_off+1), col] += 1
        W_struct[w_idx(k_off, i_off), col] += 1      # C11
        W_struct[w_idx(k_off+1, i_off+1), col] += 1  # C22
        col += 1
        
        # Term 2: M2 = (A21+A22)B11
        # A part: A21 + A22
        # B part: B11
        # C part: C21 (+M2), C22 (-M2)
        U_struct[u_idx(i_off+1, j_off), col] += 1
        U_struct[u_idx(i_off+1, j_off+1), col] += 1
        V_struct[v_idx(j_off, k_off), col] += 1
        W_struct[w_idx(k_off, i_off+1), col] += 1    # C21
        W_struct[w_idx(k_off+1, i_off+1), col] -= 1  # C22
        col += 1
        
        # Term 3: M3 = A11(B21-B22)
        # A part: A11
        # B part: B21 - B22
        # C part: C12 (+M3), C22 (+M3)
        U_struct[u_idx(i_off, j_off), col] += 1
        V_struct[v_idx(j_off+1, k_off), col] += 1
        V_struct[v_idx(j_off+1, k_off+1), col] -= 1
        W_struct[w_idx(k_off+1, i_off), col] += 1    # C12
        W_struct[w_idx(k_off+1, i_off+1), col] += 1  # C22
        col += 1
        
        # Term 4: M4 = A22(B12-B11)
        # A part: A22
        # B part: B12 - B11
        # C part: C11 (+M4), C21 (+M4)
        U_struct[u_idx(i_off+1, j_off+1), col] += 1
        V_struct[v_idx(j_off, k_off+1), col] += 1
        V_struct[v_idx(j_off, k_off), col] -= 1
        W_struct[w_idx(k_off, i_off), col] += 1      # C11
        W_struct[w_idx(k_off, i_off+1), col] += 1    # C21
        col += 1
        
        # Term 5: M5 = (A11+A12)B22
        # A part: A11 + A12
        # B part: B22
        # C part: C11 (-M5), C12 (+M5)
        U_struct[u_idx(i_off, j_off), col] += 1
        U_struct[u_idx(i_off, j_off+1), col] += 1
        V_struct[v_idx(j_off+1, k_off+1), col] += 1
        W_struct[w_idx(k_off, i_off), col] -= 1      # C11
        W_struct[w_idx(k_off+1, i_off), col] += 1    # C12
        col += 1
        
        # Term 6: M6 = (A21-A11)(B11+B12)
        # A part: A21 - A11
        # B part: B11 + B12
        # C part: C22 (+M6)
        U_struct[u_idx(i_off+1, j_off), col] += 1
        U_struct[u_idx(i_off, j_off), col] -= 1
        V_struct[v_idx(j_off, k_off), col] += 1
        V_struct[v_idx(j_off, k_off+1), col] += 1
        W_struct[w_idx(k_off+1, i_off+1), col] += 1  # C22
        col += 1
        
        # Term 7: M7 = (A12-A22)(B21+B22)
        # A part: A12 - A22
        # B part: B21 + B22
        # C part: C11 (+M7)
        U_struct[u_idx(i_off, j_off+1), col] += 1
        U_struct[u_idx(i_off+1, j_off+1), col] -= 1
        V_struct[v_idx(j_off+1, k_off), col] += 1
        V_struct[v_idx(j_off+1, k_off+1), col] += 1
        W_struct[w_idx(k_off, i_off), col] += 1      # C11
        
        return 7

    def add_2x1x1x2(i_off, j_off, k_off, start_col):
        """
        Adds schoolbook decomposition for 2x1 * 1x2.
        A: rows i_off..i_off+1, col j_off
        B: row j_off, cols k_off..k_off+1
        C: rows i_off..i_off+1, cols k_off..k_off+1
        Returns 4.
        """
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        # C11 = A1 B1
        # C12 = A1 B2
        # C21 = A2 B1
        # C22 = A2 B2
        
        col = start_col
        # Term 1: A1 B1 -> C11
        U_struct[u_idx(i_off, j_off), col] += 1
        V_struct[v_idx(j_off, k_off), col] += 1
        W_struct[w_idx(k_off, i_off), col] += 1
        col += 1
        
        # Term 2: A1 B2 -> C12
        U_struct[u_idx(i_off, j_off), col] += 1
        V_struct[v_idx(j_off, k_off+1), col] += 1
        W_struct[w_idx(k_off+1, i_off), col] += 1
        col += 1
        
        # Term 3: A2 B1 -> C21
        U_struct[u_idx(i_off+1, j_off), col] += 1
        V_struct[v_idx(j_off, k_off), col] += 1
        W_struct[w_idx(k_off, i_off+1), col] += 1
        col += 1
        
        # Term 4: A2 B2 -> C22
        U_struct[u_idx(i_off+1, j_off), col] += 1
        V_struct[v_idx(j_off, k_off+1), col] += 1
        W_struct[w_idx(k_off+1, i_off+1), col] += 1
        col += 1
        
        return 4

    # Build the structured decomposition
    col = 0
    i_blocks = [(0, 1), (2, 3)]
    j_blocks = [(0, 1), (2, 3), (4, 4)]  # Last one is 2x1, so we handle it specially
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    for i_off, i_end in i_blocks:
        for k_off, k_end in k_blocks:
            # J1: 2x2
            col += add_strassen_2x2(i_off, 0, k_off, col)
            # J2: 2x2
            col += add_strassen_2x2(i_off, 2, k_off, col)
            # J3: 2x1 * 1x2
            col += add_2x1x1x2(i_off, 4, k_off, col)
            
    # Verify the structured decomposition
    # Check if col == rank_structured
    if col != rank_structured:
        # Fallback to schoolbook if something went wrong with the structure
        pass
    else:
        # Verify correctness
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Should be 1 for all i,j,k
        U_check = U_struct[:, :col]
        V_check = V_struct[:, :col]
        W_check = W_struct[:, :col]
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.dot(U_check[i*m+j, :], V_check[j*p+k, :]), W_check[k*n+i, :])
                    if val != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_factors = (U_check, V_check, W_check)
            best_rank = col
        else:
            # Debug print if needed, but we must return something
            # print("Structured decomposition failed verification")
            pass

    U, V, W = best_factors
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
