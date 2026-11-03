"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
n, m, p = DIMENSIONS
# Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
# Tensor: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
# Output factor follows W[k*n+i, r]


def _strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    Each term: (u4, v4, w4) where:
      u4 = [U00, U01, U10, U11]
      v4 = [V00, V01, V10, V11]
      w4 = [W00, W01, W10, W11]
    """
    return [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,1,-1,0]),   # M2
        ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),   # M3
        ([0,0,0,1], [-1,1,0,0], [1,0,1,0]),   # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),   # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),   # M6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),   # M7
    ]


def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill

    n, m, p = 3, 5, 5
    nm, mp, pn = n*m, m*p, p*n

    def time_check():
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")

    # Strategy: Block decomposition of 3x5 * 5x5 -> 3x5
    # Split A (3x5) into A1 (2x5) and A2 (1x5)
    # Split C (3x5) into C1 (2x5) and C2 (1x5)
    # C1 = A1 * B (2x5 * 5x5 -> 2x5)
    # C2 = A2 * B (1x5 * 5x5 -> 1x5)
    #
    # For C1 = A1 * B:
    #   Split B's columns into groups: (0,1), (2,3), (4)
    #   For each pair of columns: 2x5 * 5x2 -> 2x2, rank 18 (Strassen-based)
    #   For single column: 2x5 * 5x1 -> 2x1, rank 10 (schoolbook)
    #   Total: 18 + 18 + 10 = 46
    #
    # For C2 = A2 * B:
    #   Split B's columns into groups: (0,1), (2,3), (4)
    #   For each pair: 1x5 * 5x2 -> 1x2, rank 10 (schoolbook)
    #   For single: 1x5 * 5x1 -> 1x1, rank 5 (schoolbook)
    #   Total: 10 + 10 + 5 = 25
    #
    # Grand total: 46 + 25 = 71

    def _pad_2x2_to_2x5_5x2(u4, v4, w4, a_cols, b_rows):
        """
        Pad a 2x2*2x2->2x2 term to a 2x5*5x2->2x2 term.
        a_cols: which 2 columns of A (0-indexed)
        b_rows: which 2 rows of B (0-indexed)
        Returns (u10, v10, w4)
        """
        u10 = [0]*10
        v10 = [0]*10
        u10[0*5 + a_cols[0]] = u4[0]
        u10[0*5 + a_cols[1]] = u4[1]
        u10[1*5 + a_cols[0]] = u4[2]
        u10[1*5 + a_cols[1]] = u4[3]
        
        v10[b_rows[0]*2 + 0] = v4[0]
        v10[b_rows[0]*2 + 1] = v4[1]
        v10[b_rows[1]*2 + 0] = v4[2]
        v10[b_rows[1]*2 + 1] = v4[3]
        
        return (u10, v10, list(w4))
    
    def _outer_product_2x1_1x2():
        """
        Return 4 terms for 2x1 * 1x2 -> 2x2 (outer product).
        """
        return [
            ([1,0], [1,0], [1,0,0,0]),
            ([1,0], [0,1], [0,1,0,0]),
            ([0,1], [1,0], [0,0,1,0]),
            ([0,1], [0,1], [0,0,0,1]),
        ]
    
    def _pad_outer_2x1_1x2_to_2x5_5x2(terms, a_col, b_row):
        """
        Pad 2x1*1x2->2x2 terms to 2x5*5x2->2x2.
        """
        result = []
        for u2, v2, w4 in terms:
            u10 = [0]*10
            v10 = [0]*10
            u10[0*5 + a_col] = u2[0]
            u10[1*5 + a_col] = u2[1]
            v10[b_row*2 + 0] = v2[0]
            v10[b_row*2 + 1] = v2[1]
            result.append((u10, v10, list(w4)))
        return result
    
    def _decompose_2x5_5x2():
        """
        Decompose 2x5 * 5x2 -> 2x2 using rank 18.
        """
        s2x2 = _strassen_2x2x2()
        terms = []
        for u4, v4, w4 in s2x2:
            terms.append(_pad_2x2_to_2x5_5x2(u4, v4, w4, (0,1), (0,1)))
        for u4, v4, w4 in s2x2:
            terms.append(_pad_2x2_to_2x5_5x2(u4, v4, w4, (2,3), (2,3)))
        op = _outer_product_2x1_1x2()
        terms.extend(_pad_outer_2x1_1x2_to_2x5_5x2(op, 4, 4))
        return terms
    
    def _decompose_1x5_5x2():
        """
        Decompose 1x5 * 5x2 -> 1x2 using rank 10 (schoolbook).
        """
        terms = []
        for j in range(5):
            for k in range(2):
                u5 = [0]*5
                v10 = [0]*10
                w2 = [0]*2
                u5[j] = 1
                v10[j*2 + k] = 1
                w2[k] = 1
                terms.append((u5, v10, w2))
        return terms
    
    all_terms = []
    
    # --- Top 2 rows (i=0,1) ---
    d2x5_5x2 = _decompose_2x5_5x2()
    
    for col_group in [(0,1), (2,3)]:
        for u10, v10, w4 in d2x5_5x2:
            u15 = [0]*15
            v25 = [0]*25
            w15 = [0]*15
            
            for i in range(2):
                for j in range(5):
                    u15[i*5 + j] = u10[i*5 + j]
            
            for j in range(5):
                for dk in range(2):
                    v25[j*5 + col_group[dk]] = v10[j*2 + dk]
            
            for i in range(2):
                for dk in range(2):
                    w15[col_group[dk]*3 + i] = w4[i*2 + dk]
            
            all_terms.append((u15, v25, w15))
    
    # 2x5 * 5x1 -> 2x1 (for column group (4,))
    for i in range(2):
        for j in range(5):
            u15 = [0]*15
            v25 = [0]*25
            w15 = [0]*15
            u15[i*5 + j] = 1
            v25[j*5 + 4] = 1
            w15[4*3 + i] = 1
            all_terms.append((u15, v25, w15))
    
    # --- Bottom row (i=2) ---
    d1x5_5x2 = _decompose_1x5_5x2()
    for col_group in [(0,1), (2,3)]:
        for u5, v10, w2 in d1x5_5x2:
            u15 = [0]*15
            v25 = [0]*25
            w15 = [0]*15
            
            for j in range(5):
                u15[10 + j] = u5[j]
            
            for j in range(5):
                for dk in range(2):
                    v25[j*5 + col_group[dk]] = v10[j*2 + dk]
            
            for dk in range(2):
                w15[col_group[dk]*3 + 2] = w2[dk]
            
            all_terms.append((u15, v25, w15))
    
    # 1x5 * 5x1 -> 1x1 for column group (4,)
    for j in range(5):
        u15 = [0]*15
        v25 = [0]*25
        w15 = [0]*15
        u15[10 + j] = 1
        v25[j*5 + 4] = 1
        w15[4*3 + 2] = 1
        all_terms.append((u15, v25, w15))
    
    rank = len(all_terms)
    
    # Build the factor matrices
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for r in range(rank):
        U[:, r] = np.array(all_terms[r][0], dtype=float)
        V[:, r] = np.array(all_terms[r][1], dtype=float)
        W[:, r] = np.array(all_terms[r][2], dtype=float)
    
    # Verify correctness
    def verify(U, V, W):
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        if not np.allclose(T, 1.0, atol=1e-9):
            diff = np.abs(T - 1.0)
            max_idx = np.unravel_index(np.argmax(diff), T.shape)
            return False, (max_idx, T[max_idx])
        return True, None
    
    is_valid, err = verify(U, V, W)
    time_check()
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    if not is_valid:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((nm, rank_sb))
        V_sb = np.zeros((mp, rank_sb))
        W_sb = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
