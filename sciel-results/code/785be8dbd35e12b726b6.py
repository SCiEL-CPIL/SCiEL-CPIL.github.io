import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Try rank 28 decomposition first, fall back to rank 56 if verification fails
    rank_28 = _construct_rank28(n, m, p, deadline)
    if rank_28 is not None:
        return rank_28
    
    # Fallback to rank 56
    return _construct_rank56(n, m, p, deadline)

def _construct_rank28(n, m, p, deadline):
    """
    Rank 28 decomposition for 2x4 x 4x8 matrix multiplication.
    
    Strategy: Split the 4x8 matrix B into two 4x4 halves: B = [B_left, B_right]
    where B_left = B[:, 0:4] and B_right = B[:, 4:8].
    
    For each 2x2x2 Strassen block, we can share terms between the two halves
    when the A coefficients are the same.
    
    The 2x4 A matrix is split into 2x2 blocks: A = [A_00, A_01] where
    A_00 = A[:, 0:2] and A_01 = A[:, 2:4].
    
    For the left half (k=0..3), we use Strassen on (A_00, A_01) x B_left
    For the right half (k=4..7), we use Strassen on (A_00, A_01) x B_right
    
    Since the A coefficients are shared, we can potentially merge some terms.
    """
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen terms for 2x2x2
    # Each term has U, V, W coefficients
    # U: {(i, j): coeff} for A[i,j]
    # V: {(j, k): coeff} for B[j,k]
    # W: {(k, i): coeff} for C[i,k] (trace convention W[k*n+i])
    
    strassen_terms = [
        # M0 = (A00+A11)(B00+B11)
        {'U': {(0,0):1, (1,1):1}, 'V': {(0,0):1, (1,1):1}, 'W': {(0,0):1, (1,1):1}},
        # M1 = (A10+A11)B00
        {'U': {(1,0):1, (1,1):1}, 'V': {(0,0):1}, 'W': {(0,1):1, (1,1):-1}},
        # M2 = A00(B01-B11)
        {'U': {(0,0):1}, 'V': {(0,1):1, (1,1):-1}, 'W': {(1,0):1, (1,1):1}},
        # M3 = A11(B10-B00)
        {'U': {(1,1):1}, 'V': {(1,0):1, (0,0):-1}, 'W': {(0,0):1, (0,1):1}},
        # M4 = (A00+A01)B11
        {'U': {(0,0):1, (0,1):1}, 'V': {(1,1):1}, 'W': {(0,0):-1, (1,0):1}},
        # M5 = (A10-A00)(B00+B01)
        {'U': {(1,0):1, (0,0):-1}, 'V': {(0,0):1, (0,1):1}, 'W': {(1,1):1}},
        # M6 = (A01-A11)(B10+B11)
        {'U': {(0,1):1, (1,1):-1}, 'V': {(1,0):1, (1,1):1}, 'W': {(0,0):1}},
    ]
    
    col = 0
    
    # For each of the 4 j-pairs (j=0,1 and j=2,3) and each of the 4 k-pairs (k=0,1 and k=2,3)
    # We need to apply Strassen. But we can share A coefficients.
    
    # Actually, let's think about this differently.
    # A is 2x4, so we have 2x2 blocks: A[:, 0:2] and A[:, 2:4]
    # B is 4x8, so we have 2x2 blocks in both dimensions:
    #   B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
    #   B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6], B[2:4, 6:8]
    
    # For each combination of A-block (2 options) and B-block (4 options),
    # we apply Strassen. That's 8 blocks x 7 terms = 56 terms.
    
    # To get rank 28, we need to share terms. The key is that for a given
    # A-block, the U coefficients are the same across different B-blocks.
    # So we can potentially merge V and W coefficients.
    
    # Let's try a different approach: use 4 independent Strassen decompositions
    # but share the A terms.
    
    # Actually, the simplest way to get rank 28 is to use 4 Strassen blocks
    # (one for each 2x2x2 sub-problem) and then combine them.
    
    # Let me reconsider: we have 2x4 x 4x8. We can view this as:
    # - Split A into 2 blocks of 2x2: A_left = A[:, 0:2], A_right = A[:, 2:4]
    # - Split B into 4 blocks of 2x2: B[0:2, 0:2], B[0:2, 2:4], B[2:4, 0:2], B[2:4, 2:4]
    #   Wait, B is 4x8, so we have 2x2 blocks: 2 rows x 4 cols = 8 blocks of 2x2
    
    # Actually, B is 4x8, so it has (4/2) * (8/2) = 2 * 4 = 8 blocks of 2x2.
    # A is 2x4, so it has (2/2) * (4/2) = 1 * 2 = 2 blocks of 2x2.
    
    # For each A-block (2 options) and each B-block (8 options), we could apply
    # Strassen, giving 16 blocks x 7 = 112 terms. That's too many.
    
    # The key insight: for a fixed A-block, the U coefficients are the same
    # across all B-blocks that use the same row indices. So we can share U.
    
    # Let me just implement the straightforward rank 56 first and verify it works,
    # then try to optimize.
    
    # For now, return None to use the fallback
    return None

def _construct_rank56(n, m, p, deadline):
    """Rank 56 decomposition using 8 independent Strassen blocks."""
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        {'U': {(0,0):1, (1,1):1}, 'V': {(0,0):1, (1,1):1}, 'W': {(0,0):1, (1,1):1}},
        {'U': {(1,0):1, (1,1):1}, 'V': {(0,0):1}, 'W': {(0,1):1, (1,1):-1}},
        {'U': {(0,0):1}, 'V': {(0,1):1, (1,1):-1}, 'W': {(1,0):1, (1,1):1}},
        {'U': {(1,1):1}, 'V': {(1,0):1, (0,0):-1}, 'W': {(0,0):1, (0,1):1}},
        {'U': {(0,0):1, (0,1):1}, 'V': {(1,1):1}, 'W': {(0,0):-1, (1,0):1}},
        {'U': {(1,0):1, (0,0):-1}, 'V': {(0,0):1, (0,1):1}, 'W': {(1,1):1}},
        {'U': {(0,1):1, (1,1):-1}, 'V': {(1,0):1, (1,1):1}, 'W': {(0,0):1}},
    ]
    
    col = 0
    for j_high in range(2):  # j = 0,1 or 2,3
        for k_high in range(4):  # k = 0,1 or 2,3 or 4,5 or 6,7
            for t_idx in range(7):
                term = strassen_terms[t_idx]
                
                for (i, j_low), coeff in term['U'].items():
                    j = j_high * 2 + j_low
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                for (j_low, k_low), coeff in term['V'].items():
                    j = j_high * 2 + j_low
                    k = k_high * 2 + k_low
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                for (k_low, i), coeff in term['W'].items():
                    k = k_high * 2 + k_low
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                    
                col += 1
    
    # Verify the decomposition
    if time.time() < deadline:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        np.random.seed(42)
        test_ok = True
        for _ in range(100):
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            val = np.sum(U_arr[u_idx, :] * V_arr[v_idx, :] * W_arr[w_idx, :])
            if abs(val - 1.0) > 1e-9:
                test_ok = False
                break
        
        if not test_ok:
            # Fallback: return naive rank 64
            rank_naive = n * m * p
            U_naive = [[0] * rank_naive for _ in range(n * m)]
            V_naive = [[0] * rank_naive for _ in range(m * p)]
            W_naive = [[0] * rank_naive for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        U_naive[i * m + j][r] = 1
                        V_naive[j * p + k][r] = 1
                        W_naive[k * n + i][r] = 1
            return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
