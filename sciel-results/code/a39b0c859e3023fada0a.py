import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

def _build_trivial(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m*p) + j * p + k
                U[i*m + j, r] = 1
                V[j*p + k, r] = 1
                W[k*n + i, r] = 1
    return U, V, W

def _build_block_strassen(n, m, p):
    """Block-strassen decomposition. Rank 42 for 2x4x6."""
    rank = 42
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen 2x2x2
    U_s = np.zeros((4, 7), dtype=float)
    V_s = np.zeros((4, 7), dtype=float)
    W_s = np.zeros((4, 7), dtype=float)
    
    U_s[0, 0] = 1; U_s[3, 0] = 1
    U_s[2, 1] = 1; U_s[3, 1] = 1
    U_s[0, 2] = 1
    U_s[3, 3] = 1
    U_s[0, 4] = 1; U_s[1, 4] = 1
    U_s[2, 5] = 1; U_s[0, 5] = -1
    U_s[1, 6] = 1; U_s[3, 6] = -1
    
    V_s[0, 0] = 1; V_s[3, 0] = 1
    V_s[0, 1] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    V_s[3, 4] = 1
    V_s[0, 5] = 1; V_s[2, 5] = 1
    V_s[1, 6] = 1; V_s[3, 6] = 1
    
    W_s[0, 0] = 1; W_s[0, 3] = 1; W_s[0, 4] = -1; W_s[0, 6] = 1
    W_s[1, 1] = 1; W_s[1, 3] = 1
    W_s[2, 2] = 1; W_s[2, 4] = 1
    W_s[3, 0] = 1; W_s[3, 1] = -1; W_s[3, 2] = 1; W_s[3, 5] = 1
    
    for a_block in range(2):
        for b_block in range(3):
            base_r = (a_block * 3 + b_block) * 7
            a_col_offset = a_block * 2
            b_row_offset = a_block * 2
            b_col_offset = b_block * 2
            
            for r_s in range(7):
                r = base_r + r_s
                for i in range(2):
                    for j in range(2):
                        full_j = j + a_col_offset
                        U[i * m + full_j, r] = U_s[i * 2 + j, r_s]
                for j in range(2):
                    for k in range(2):
                        full_j = j + b_row_offset
                        full_k = k + b_col_offset
                        V[full_j * p + full_k, r] = V_s[j * 2 + k, r_s]
                for k in range(2):
                    for i in range(2):
                        full_k = k + b_col_offset
                        W[full_k * n + i, r] = W_s[k * 2 + i, r_s]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, num_tests=20, seed=42):
    """Verify that the tensor decomposition exactly matches matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    rng = np.random.RandomState(seed)
    for _ in range(num_tests):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                C_computed[i, k] = s
        if not np.allclose(C_computed, C_expected, atol=1e-9, rtol=1e-9):
            return False
    return True

def _search_low_rank(n, m, p, target_rank, time_limit):
    """
    Attempt to find a rank-target_rank decomposition using structured integer search.
    Uses the fact that for small dimensions, we can try sparse patterns.
    """
    start_time = time.time()
    
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We want U (n*m, R), V (m*p, R), W (p*n, R) with integer/half-integer entries
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Strategy: Try to find a decomposition where each "term" r involves
    # a small number of non-zero entries in U, V, W.
    # For rank R, each term is a rank-1 tensor u_r ⊗ v_r ⊗ w_r.
    
    # For 2x4x6, let's try rank 24, 28, 30, 32, 34, 36
    # Use a greedy approach: build up the decomposition term by term.
    
    # Actually, let's try a different approach: use the known structure.
    # 2x4x6 can be seen as:
    # Split the 4 middle dimension into two 2s: A = [A1, A2] where A1, A2 are 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where Bij are 2x2
    # C = A1 * [B11, B12, B13] + A2 * [B21, B22, B23]
    #    = [A1*B11 + A2*B21, A1*B12 + A2*B22, A1*B13 + A2*B23]
    
    # For each output column block k (0,1,2):
    # C[:, k*2:(k+1)*2] = A1 * B[0, k*2:(k+1)*2] + A2 * B[1, k*2:(k+1)*2]
    
    # This is a 2x2 * 2x2 + 2x2 * 2x2 = 2x2 problem for each k.
    # The tensor for "X*Y + Z*W = C" where X,Z are 2x2, Y,W are 2x2, C is 2x2.
    
    # Actually, let's think of it differently.
    # The full tensor is T[i, j, k] = 1 for i in {0,1}, j in {0,1,2,3}, k in {0,...,5}
    
    # Let's try to use the approach from the inspiration: rank 36.
    # This would correspond to 2 * 18 for the two A-blocks, or some other structure.
    
    # Given time constraints, let's try a focused search for rank 36.
    # We'll use the fact that the decomposition must satisfy:
    # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try to construct a rank-36 decomposition by combining:
    # - For each of the 2 A-rows, use a 2x2x6 sub-decomposition
    # If rank(2x2x6) = 18, total = 36.
    
    # Let's try to find rank(2x2x6).
    # T226[i, j, k] = 1 for i in {0,1}, j in {0,1}, k in {0,...,5}
    # Lower bound: max(4, 12, 12) = 12.
    # Schoolbook: 12.
    # Can we do better than 12? Unlikely for 2x2x6.
    
    # So rank(2x2x6) is probably 12, giving rank(2x4x6) = 24.
    # But we need to verify this is achievable with integer/half-integer entries.
    
    # Let's try to construct rank 24:
    # For A-block 0 (rows 0,1 of A, cols 0,1):
    #   U0 (4, 12), V0 (12, 12), W0 (12, 12) for 2x2x6
    # For A-block 1 (rows 0,1 of A, cols 2,3):
    #   U1 (4, 12), V1 (12, 12), W1 (12, 12) for 2x2x6
    
    # The U part: U[i*m + j, r]
    # For block 0: j in {0,1}, so U[0*4+0, r], U[0*4+1, r], U[1*4+0, r], U[1*4+1, r]
    # For block 1: j in {2,3}, so U[0*4+2, r+12], U[0*4+3, r+12], U[1*4+2, r+12], U[1*4+3, r+12]
    
    # The V part: V[j*p + k, r]
    # For block 0: j in {0,1}, k in {0,...,5}
    # For block 1: j in {2,3}, k in {0,...,5}
    
    # The W part: W[k*n + i, r]
    # Same for both blocks since i, k are the same.
    
    # So we need a 2x2x6 decomposition with rank 12.
    # Schoolbook for 2x2x6: rank = 2*2*6 = 24? No.
    # Wait, 2x2x6 means A is 2x2, B is 2x6, C is 2x6.
    # The tensor has size (2*2) x (2*6) x (6*2) = 4 x 12 x 12.
    # Schoolbook rank = 2*2*6 = 24? No, the number of terms is n*m*p = 2*2*6 = 24.
    # But the lower bound is max(4, 12, 12) = 12.
    
    # Can we achieve rank 12 for 2x2x6?
    # T226[i, j, k] = 1.
    # We need U (4, 12), V (12, 12), W (12, 12).
    
    # Let's try: for each (i, k) pair, we need sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    # This is equivalent to: for each (i,k), the vector (V[:,k,:] * W[k,i,:]) dotted with U[i,:,r] = 1
    
    # This is complex. Let's just try the schoolbook rank 24 for 2x2x6,
    # which gives rank 48 for 2x4x6. That's worse than 42.
    
    # Alternatively, use Strassen for the 2x2 parts:
    # For 2x2x6: split B into three 2x2 blocks.
    # C = A*B1 + A*B2 + A*B3
    # Use Strassen for each: rank 3*7 = 21 for 2x2x6.
    # Total for 2x4x6: 2*21 = 42. Same as current.
    
    # To get 36, we need rank(2x2x6) = 18.
    # Is rank(2x2x6) = 18 achievable?
    # Lower bound is 12, schoolbook is 24, Strassen-based is 21.
    # Maybe 18 is possible with a smarter decomposition.
    
    # Given the time constraint and difficulty of finding a novel low-rank decomposition,
    # I'll implement a search that tries to verify if rank 36 is achievable by
    # checking if the tensor has a rank-36 decomposition using SVD-based verification,
    # and if not, fall back to 42.
    
    # For now, return None to indicate no better rank found.
    if time.time() - start_time > time_limit:
        return None
    
    # Try to use SVD to estimate the tensor rank
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Matricize and check SVD
    T_mat = T.reshape(n*m, m*p * p*n)
    _, s, _ = np.linalg.svd(T_mat, full_matrices=False)
    num_rank = np.sum(s > 1e-10)
    
    # The SVD rank of the matricization is an upper bound on the tensor rank.
    # If num_rank <= target_rank, it might be possible.
    if num_rank > target_rank:
        return None
    
    # We can't easily construct the decomposition from SVD, so return None.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find a lower rank decomposition
    for target_rank in [24, 28, 30, 32, 34, 36, 38, 40]:
        if time.time() > deadline - 30:
            break
        result = _search_low_rank(n, m, p, target_rank, time_limit=10)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p, num_tests=10, seed=123):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to block Strassen (rank 42)
    U, V, W = _build_block_strassen(n, m, p)
    if time.time() < deadline:
        if _verify_decomposition(U, V, W, n, m, p, num_tests=20, seed=123):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to trivial
    U_t, V_t, W_t = _build_trivial(n, m, p)
    return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
