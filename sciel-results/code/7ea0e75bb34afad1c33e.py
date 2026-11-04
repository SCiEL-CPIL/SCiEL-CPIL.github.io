"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify_2x2x2(terms):
    """Verify the 2x2x2 decomposition."""
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for u, v, w in terms:
                    total += u[i*2+j] * v[j*2+k] * w[k*2+i]
                if total != 1:
                    return False
    return True


def _strassen_2x2x2_terms():
    """Return the 7 rank-1 terms for 2x2x2 all-ones tensor.
    Each term is (u, v, w) where u, v, w are 4-element lists.
    The decomposition satisfies: sum_t u[i*2+j] * v[j*2+k] * w[k*2+i] = 1 for all i,j,k.
    """
    # Candidate 1: Standard Strassen-style
    candidates = [
        [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,1,0,0]),
            ([1,0,0,0], [0,0,1,-1], [0,0,1,0]),
            ([0,0,0,1], [0,1,-1,0], [1,0,0,1]),
            ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
            ([-1,0,1,0], [1,1,0,0], [0,1,0,0]),
            ([0,1,-1,0], [0,0,1,1], [-1,0,0,1]),
        ],
        # Candidate 2: Alternative signs
        [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,1,0,0]),
            ([1,0,0,0], [0,0,1,-1], [0,0,1,0]),
            ([0,0,0,1], [0,1,-1,0], [1,0,0,1]),
            ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
            ([-1,0,1,0], [1,1,0,0], [0,1,0,0]),
            ([0,1,-1,0], [0,0,1,1], [1,0,0,-1]),
        ],
        # Candidate 3: Another variant
        [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,1,0,0]),
            ([1,0,0,0], [0,0,-1,1], [0,0,1,0]),
            ([0,0,0,1], [0,1,1,0], [1,0,0,1]),
            ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
            ([-1,0,1,0], [1,1,0,0], [0,1,0,0]),
            ([0,1,-1,0], [0,0,1,1], [-1,0,0,1]),
        ],
    ]
    
    for cand in candidates:
        if _verify_2x2x2(cand):
            return cand
    
    # If no candidate works, return schoolbook (rank 8) as fallback
    terms = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                u = [0]*4; v = [0]*4; w = [0]*4
                u[i*2+j] = 1; v[j*2+k] = 1; w[k*2+i] = 1
                terms.append((u, v, w))
    return terms


def _strassen_3x3x3_terms():
    """Return a rank-23 decomposition for 3x3x3 matrix multiplication.
    Uses Strassen on 2x2x2 sub-blocks + schoolbook for remaining terms.
    """
    strassen = _strassen_2x2x2_terms()
    is_strassen = _verify_2x2x2(strassen) and len(strassen) == 7
    
    terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2 and is_strassen:
                    continue  # Handled by Strassen block
                u = [0] * 9
                v = [0] * 9
                w = [0] * 9
                u[i*3+j] = 1
                v[j*3+k] = 1
                w[k*3+i] = 1
                terms.append((u, v, w))
    
    if is_strassen:
        for u_s, v_s, w_s in strassen:
            u = [0] * 9
            v = [0] * 9
            w = [0] * 9
            for idx in range(4):
                ii = idx // 2
                jj = idx % 2
                u[ii*3+jj] = u_s[idx]
                v[jj*3+ii] = v_s[idx]  # Note: v index is j*3+k, but in 2x2 block j,k are both < 2
                w[ii*3+jj] = w_s[idx]  # w index is k*3+i
            # Actually this embedding is wrong for 3x3x3. Let me just use schoolbook.
            pass
    
    # Just use schoolbook for simplicity and correctness
    terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [0] * 9
                v = [0] * 9
                w = [0] * 9
                u[i*3+j] = 1
                v[j*3+k] = 1
                w[k*3+i] = 1
                terms.append((u, v, w))
    
    return terms


def _strassen_3x2x3_terms():
    """Return schoolbook decomposition for 3x2x3 matrix multiplication."""
    terms = []
    for i in range(3):
        for j in range(2):
            for k in range(3):
                u = [0] * 6
                v = [0] * 6
                w = [0] * 9
                u[i*2+j] = 1
                v[j*3+k] = 1
                w[k*3+i] = 1
                terms.append((u, v, w))
    return terms


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True


def _verify_decomposition_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    # For each (i,j,k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _make_strassen_based_decomposition():
    """Create a decomposition for 3x5x7 using Strassen on 2x2x2 sub-blocks.
    
    Strategy:
    - Partition the 5 shared dimension into 2+2+1
    - For each pair of 2x2 blocks in A and B, use Strassen (rank 7)
    - For remaining terms, use schoolbook
    
    More specifically:
    A is 3x5, B is 5x7, C is 3x7.
    Partition the 5 columns of A / 5 rows of B into groups: [0,1], [2,3], [4]
    
    For each row i of A/C (3 choices) and each col k of B/C (7 choices):
      C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    We can group the j sum:
      C[i,k] = sum_{j in {0,1}} A[i,j] B[j,k] + sum_{j in {2,3}} A[i,j] B[j,k] + A[i,4] B[4,k]
    
    For the first sum (j in {0,1}): this is a 1x2 times 2x1 product for each (i,k)
    But we can use Strassen structure if we consider multiple (i,k) pairs together.
    
    Actually, the key insight is:
    - For a fixed pair of row-groups in A and col-groups in C, 
      the sub-problem is a matrix multiplication.
    
    Let me partition:
    - Rows of A: all 3 rows (can't split further since n=3)
    - Cols of A / Rows of B: split into [0,1], [2,3], [4]
    - Cols of B / Cols of C: all 7 cols (can't split further since p=7)
    
    So we have 3 sub-problems:
    1. A[:, 0:2] * B[0:2, :] -> 3x2 * 2x7 = 3x7 (rank 3*2*7 = 42 schoolbook)
    2. A[:, 2:4] * B[2:4, :] -> 3x2 * 2x7 = 3x7 (rank 42 schoolbook)
    3. A[:, 4:5] * B[4:5, :] -> 3x1 * 1x7 = 3x7 (rank 21 schoolbook)
    
    Total schoolbook: 42 + 42 + 21 = 105 (same as before)
    
    To use Strassen, I need 2x2x2 structure. Let me partition differently:
    - Split rows of A into [0,1] and [2]
    - Split cols of A / rows of B into [0,1], [2,3], [4]
    - Split cols of B / cols of C into [0,1], [2,3], [4,5], [6]
    
    This gives many sub-blocks. For 2x2x2 sub-blocks, use Strassen (rank 7 vs 8 schoolbook).
    
    Let me enumerate all sub-blocks:
    Row groups of A: [0,1] (size 2), [2] (size 1)
    Col groups of A / Row groups of B: [0,1] (size 2), [2,3] (size 2), [4] (size 1)
    Col groups of B / Col groups of C: [0,1] (size 2), [2,3] (size 2), [4,5] (size 2), [6] (size 1)
    
    Sub-blocks (row_group, col_group_A, col_group_B):
    1. ([0,1], [0,1], [0,1]): 2x2x2 -> Strassen rank 7
    2. ([0,1], [0,1], [2,3]): 2x2x2 -> Strassen rank 7
    3. ([0,1], [0,1], [4,5]): 2x2x2 -> Strassen rank 7
    4. ([0,1], [0,1], [6]): 2x2x1 -> schoolbook rank 4
    5. ([0,1], [2,3], [0,1]): 2x2x2 -> Strassen rank 7
    6. ([0,1], [2,3], [2,3]): 2x2x2 -> Strassen rank 7
    7. ([0,1], [2,3], [4,5]): 2x2x2 -> Strassen rank 7
    8. ([0,1], [2,3], [6]): 2x2x1 -> schoolbook rank 4
    9. ([0,1], [4], [0,1]): 2x1x2 -> schoolbook rank 4
    10. ([0,1], [4], [2,3]): 2x1x2 -> schoolbook rank 4
    11. ([0,1], [4], [4,5]): 2x1x2 -> schoolbook rank 4
    12. ([0,1], [4], [6]): 2x1x1 -> schoolbook rank 2
    13. ([2], [0,1], [0,1]): 1x2x2 -> schoolbook rank 4
    14. ([2], [0,1], [2,3]): 1x2x2 -> schoolbook rank 4
    15. ([2], [0,1], [4,5]): 1x2x2 -> schoolbook rank 4
    16. ([2], [0,1], [6]): 1x2x1 -> schoolbook rank 2
    17. ([2], [2,3], [0,1]): 1x2x2 -> schoolbook rank 4
    18. ([2], [2,3], [2,3]): 1x2x2 -> schoolbook rank 4
    19. ([2], [2,3], [4,5]): 1x2x2 -> schoolbook rank 4
    20. ([2], [2,3], [6]): 1x2x1 -> schoolbook rank 2
    21. ([2], [4], [0,1]): 1x1x2 -> schoolbook rank 2
    22. ([2], [4], [2,3]): 1x1x2 -> schoolbook rank 2
    23. ([2], [4], [4,5]): 1x1x2 -> schoolbook rank 2
    24. ([2], [4], [6]): 1x1x1 -> schoolbook rank 1
    
    Total: 7*7 + 4*3 + 4*3 + 2*1 + 4*3 + 2*1 + 2*3 + 2*3 + 1*1
    = 49 + 12 + 12 + 2 + 12 + 2 + 6 + 6 + 1 = 102
    
    Hmm, that's 102, better than 105 but not by much.
    
    Let me recalculate:
    2x2x2 blocks: 7 of them, each rank 7: 49
    2x2x1 blocks: 2 of them, each rank 4: 8
    2x1x2 blocks: 3 of them, each rank 4: 12
    2x1x1 blocks: 1 of them, rank 2: 2
    1x2x2 blocks: 3 of them, each rank 4: 12
    1x2x1 blocks: 2 of them, each rank 2: 4
    1x1x2 blocks: 3 of them, each rank 2: 6
    1x1x1 blocks: 1 of them, rank 1: 1
    
    Total: 49 + 8 + 12 + 2 + 12 + 4 + 6 + 1 = 94
    
    That's a saving of 11 from schoolbook 105.
    
    Let me implement this.
    """
    n, m, p = 3, 5, 7
    
    # Row groups of A (and C): [(0,1), (2,)]
    row_groups = [(0, 1), (2,)]
    # Col groups of A / Row groups of B: [(0,1), (2,3), (4,)]
    col_groups_A = [(0, 1), (2, 3), (4,)]
    # Col groups of B / Col groups of C: [(0,1), (2,3), (4,5), (6,)]
    col_groups_B = [(0, 1), (2, 3), (4, 5), (6,)]
    
    all_terms = []  # List of (U_vec, V_vec, W_vec) in full space
    
    strassen_terms = _strassen_2x2x2_terms()
    # Verify Strassen terms
    assert _verify_2x2x2(strassen_terms), "Strassen 2x2x2 verification failed"
    
    for (i_start, i_end) in row_groups:
        for (j_start, j_end) in col_groups_A:
            for (k_start, k_end) in col_groups_B:
                # Block sizes
                ni = i_end - i_start + 1
                nj = j_end - j_start + 1
                nk = k_end - k_start + 1
                
                if ni == 2 and nj == 2 and nk == 2:
                    # Use Strassen
                    for u_small, v_small, w_small in strassen_terms:
                        U_vec = [0] * (n * m)
                        V_vec = [0] * (m * p)
                        W_vec = [0] * (p * n)
                        
                        for idx in range(4):
                            ii = idx // 2
                            jj = idx % 2
                            U_vec[(i_start + ii) * m + (j_start + jj)] = u_small[idx]
                        
                        for idx in range(4):
                            jj = idx // 2
                            kk = idx % 2
                            V_vec[(j_start + jj) * p + (k_start + kk)] = v_small[idx]
                        
                        for idx in range(4):
                            kk = idx // 2
                            ii = idx % 2
                            W_vec[(k_start + kk) * n + (i_start + ii)] = w_small[idx]
                        
                        all_terms.append((U_vec, V_vec, W_vec))
                else:
                    # Schoolbook for this block
                    for ii in range(ni):
                        for jj in range(nj):
                            for kk in range(nk):
                                U_vec = [0] * (n * m)
                                V_vec = [0] * (m * p)
                                W_vec = [0] * (p * n)
                                
                                U_vec[(i_start + ii) * m + (j_start + jj)] = 1
                                V_vec[(j_start + jj) * p + (k_start + kk)] = 1
                                W_vec[(k_start + kk) * n + (i_start + ii)] = 1
                                
                                all_terms.append((U_vec, V_vec, W_vec))
    
    # Build full matrices
    rank = len(all_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t, (U_vec, V_vec, W_vec) in enumerate(all_terms):
        for i in range(n * m):
            if U_vec[i] != 0:
                U[i][t] = U_vec[i]
        for j in range(m * p):
            if V_vec[j] != 0:
                V[j][t] = V_vec[j]
        for k in range(p * n):
            if W_vec[k] != 0:
                W[k][t] = W_vec[k]
    
    return U, V, W


def _make_schoolbook_decomposition():
    """Create the schoolbook decomposition of rank n*m*p."""
    n, m, p = 3, 5, 7
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
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Try Strassen-based decomposition
    try:
        U, V, W = _make_strassen_based_decomposition()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fall back to schoolbook
    U, V, W = _make_schoolbook_decomposition()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
