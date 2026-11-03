"""Tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = 120.0

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all (i,j,k)
    """
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """
    Return the 7 rank-1 terms for 2x2 * 2x2 matrix multiplication.
    Each term is (u, v, w) where:
      u is a 4-vector for A indices (A00, A01, A10, A11)
      v is a 4-vector for B indices (B00, B01, B10, B11)
      w is a 4-vector for C indices (C00, C01, C10, C11)
    """
    # Strassen's 7 multiplications:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)(B00)
    # M3 = (A00)(B10 + B11)
    # M4 = (A00 - A11)(B11)
    # M5 = (A01 + A11)(B00 - B10)
    # M6 = (A00 - A10)(B01 + B11)
    # M7 = (A10 - A00)(B00 - B11)
    
    terms = []
    # M1
    terms.append((np.array([1, 0, 0, 1], dtype=np.int64),
                  np.array([1, 0, 0, 1], dtype=np.int64),
                  np.array([1, 0, 0, 1], dtype=np.int64)))
    # M2
    terms.append((np.array([0, 0, 1, 1], dtype=np.int64),
                  np.array([1, 0, 0, 0], dtype=np.int64),
                  np.array([0, 0, 1, 0], dtype=np.int64)))
    # M3
    terms.append((np.array([1, 0, 0, 0], dtype=np.int64),
                  np.array([0, 0, 1, 1], dtype=np.int64),
                  np.array([0, 1, 0, 0], dtype=np.int64)))
    # M4
    terms.append((np.array([1, 0, 0, -1], dtype=np.int64),
                  np.array([0, 0, 0, 1], dtype=np.int64),
                  np.array([0, 1, 0, 0], dtype=np.int64)))
    # M5
    terms.append((np.array([0, 1, 0, 1], dtype=np.int64),
                  np.array([1, 0, -1, 0], dtype=np.int64),
                  np.array([1, 0, 0, 0], dtype=np.int64)))
    # M6
    terms.append((np.array([1, 0, -1, 0], dtype=np.int64),
                  np.array([0, 1, 0, 1], dtype=np.int64),
                  np.array([0, 1, 0, 0], dtype=np.int64)))
    # M7
    terms.append((np.array([0, 0, 1, -1], dtype=np.int64),
                  np.array([1, 0, 0, -1], dtype=np.int64),
                  np.array([1, 0, 0, 0], dtype=np.int64)))
    
    return terms


def _construct_strassen_based(n, m, p):
    """
    Construct a decomposition using Strassen's algorithm for 2x2x2 blocks.
    
    For 3x3 * 3x8:
    - Partition A (3x3) into 2x2 block (rows 0,1; cols 0,1) + remaining
    - Partition B (3x8) into 2x2 blocks (rows 0,1; cols 0,1) + remaining
    
    Strategy: Use Strassen for the (0,1)x(0,1) block computation, and schoolbook
    for the remaining cross terms.
    
    A = [[A00, A01, A02],
         [A10, A11, A12],
         [A20, A21, A22]]
    
    B = [[B00, B01, B02, B03, B04, B05, B06, B07],
         [B10, B11, B12, B13, B14, B15, B16, B17],
         [B20, B21, B22, B23, B24, B25, B26, B27]]
    
    C = A * B (3x8)
    
    C[0, k] = A00*B0k + A01*B1k + A02*B2k
    C[1, k] = A10*B0k + A11*B1k + A12*B2k
    C[2, k] = A20*B0k + A21*B1k + A22*B2k
    
    For k in 0,1: we can use Strassen on the (A00,A01,A10,A11) x (B00,B01,B10,B11) block.
    For k in 2..7: we use schoolbook for each k separately.
    
    Actually, let's be more careful. The Strassen decomposition applies to:
    C_block = A_block * B_block where A_block is 2x2 and B_block is 2x2.
    
    But here B has 8 columns. Let's partition B's columns into groups.
    
    Let's use Strassen for columns 0,1 of B and schoolbook for the rest.
    
    C[0:2, 0:2] comes from A[0:2, :] * B[:, 0:2]
    But A[0:2, :] is 2x3, not 2x2.
    
    Let's think differently. We'll decompose the full 3x3 * 3x8 as:
    - 2x2 * 2x2 Strassen for the (rows 0,1 of A, cols 0,1 of B) contribution to (rows 0,1 of C, cols 0,1 of C)
    - Cross terms involving row 2 of A or cols 2+ of B
    
    This gets complicated. Let's try a simpler approach:
    
    Use Strassen for the (0,1)x(0,1) sub-block and schoolbook for everything else.
    
    Actually, the cleanest way: think of 3x3 * 3x8 as a sum of rank-1 terms.
    Schoolbook gives 72 terms. Strassen can reduce the 2x2 * 2x2 part from 4 to 7... 
    wait, Strassen reduces 2x2*2x2 from 4 multiplications to 7? No, Strassen uses 7 
    multiplications for 2x2*2x2, which is MORE than 4. That's wrong.
    
    Strassen's algorithm uses 7 multiplications instead of 8 for 2x2*2x2 matrix multiplication.
    So it saves 1 multiplication per 2x2*2x2 block.
    
    For 3x3 * 3x8:
    - We can extract a 2x2 * 2x2 block and use Strassen (7 terms instead of 4... no)
    
    Wait, I'm confusing myself. In Strassen, 2x2 * 2x2 requires 7 scalar multiplications 
    instead of 8. So it saves 1.
    
    For 3x3 * 3x8 with schoolbook rank 72, if we can identify one 2x2 * 2x2 subproblem,
    we save 1 term, getting rank 71.
    
    But we can do better by identifying multiple disjoint 2x2 * 2x2 blocks.
    
    Let's try: partition A into 2x2 (rows 0,1, cols 0,1) + cross terms.
    Partition B into 2x2 (rows 0,1, cols 0,1) + cross terms.
    
    The product C = A*B can be written as:
    C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:3] * B[2:3, 0:2]
    
    Hmm, this is getting messy. Let me just implement a direct construction.
    
    For now, let's use the following approach:
    - Use 7 Strassen terms for the (0,1)x(0,1) * (0,1)x(0,1) block
    - Use schoolbook for all other contributions
    - Total rank = 72 - 4 + 7 = 75... that's worse!
    
    I made an error. Strassen uses 7 terms for what would be 4 terms in schoolbook for 2x2*2x2.
    So it's actually WORSE in terms of tensor rank!
    
    Wait no. In standard matrix multiplication of 2x2 * 2x2, schoolbook uses 8 multiplications.
    Strassen uses 7. So Strassen saves 1 multiplication.
    
    The 4 I mentioned was wrong. Let me redo:
    - 2x2 * 2x2 schoolbook: 8 terms (rank 8)
    - 2x2 * 2x2 Strassen: 7 terms (rank 7)
    
    So for 3x3 * 3x8, if we isolate a 2x2 * 2x2 subproblem, we replace 8 schoolbook terms
    with 7 Strassen terms, saving 1. Total rank: 72 - 8 + 7 = 71.
    
    Can we find multiple disjoint 2x2 * 2x2 blocks?
    
    A is 3x3, B is 3x8.
    We can partition A's rows into {0,1} and {2}, and A's cols into {0,1} and {2}.
    We can partition B's rows into {0,1} and {2}, and B's cols into pairs.
    
    For each pair of B columns (say cols 2t, 2t+1), we have a 3x2 block.
    The (0,1)x(0,1) * (0,1)x(2t, 2t+1) part is a 2x2 * 2x2 subproblem.
    
    There are 4 such column pairs (0,1), (2,3), (4,5), (6,7).
    Each gives a 2x2 * 2x2 subproblem that can be done in 7 instead of 8.
    
    So we save 4 terms total: rank = 72 - 4*1 = 68.
    
    Let's construct this properly.
    """
    n, m, p = 3, 3, 8
    
    # We'll build the decomposition term by term.
    # Start with schoolbook terms, then replace 2x2*2x2 blocks with Strassen terms.
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    strassen_terms = _strassen_2x2x2()
    
    # For each column pair (2t, 2t+1) in B, use Strassen for the (rows 0,1 of A) x 
    # (cols 0,1 of A) * (rows 0,1 of B) x (cols 2t, 2t+1 of B) block.
    
    for t in range(4):
        k1 = 2 * t
        k2 = 2 * t + 1
        
        # Strassen terms for this 2x2 * 2x2 block
        for (u_s, v_s, w_s) in strassen_terms:
            # u_s is for A[0:2, 0:2] -> indices (A00, A01, A10, A11)
            # v_s is for B[0:2, k1:k2+1] -> indices (B0k1, B0k2, B1k1, B1k2)
            # w_s is for C[0:2, k1:k2+1] -> indices (C0k1, C0k2, C1k1, C1k2)
            
            u_vec = np.zeros(n * m, dtype=np.int64)
            v_vec = np.zeros(m * p, dtype=np.int64)
            w_vec = np.zeros(p * n, dtype=np.int64)
            
            # Map Strassen indices to full matrix indices
            # A: (i, j) where i in {0,1}, j in {0,1}
            a_map = [(0, 0), (0, 1), (1, 0), (1, 1)]
            for idx, (ai, aj) in enumerate(a_map):
                u_vec[ai * m + aj] = u_s[idx]
            
            # B: (j, k) where j in {0,1}, k in {k1, k2}
            b_map = [(0, k1), (0, k2), (1, k1), (1, k2)]
            for idx, (bi, bk) in enumerate(b_map):
                v_vec[bi * p + bk] = v_s[idx]
            
            # C: (k, i) where k in {k1, k2}, i in {0,1}
            c_map = [(k1, 0), (k1, 1), (k2, 0), (k2, 1)]
            for idx, (ck, ci) in enumerate(c_map):
                w_vec[ck * n + ci] = w_s[idx]
            
            U_terms.append(u_vec)
            V_terms.append(v_vec)
            W_terms.append(w_vec)
    
    # Add schoolbook terms for the remaining parts:
    # 1. A[0:2, 2] * B[:, k] for all k (contribution to C[0:2, k])
    # 2. A[2, :] * B[:, k] for all k (contribution to C[2, k])
    # 3. A[0:2, 0:2] * B[2, k] for all k (contribution to C[0:2, k])
    #    Wait, this is already covered? No, the Strassen terms only cover B[0:2, k].
    
    # Let me re-examine. The full product is:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # 
    # For i in {0,1}, k in {0,1,2,...,7}:
    # C[i, k] = A[i, 0]*B[0, k] + A[i, 1]*B[1, k] + A[i, 2]*B[2, k]
    # 
    # The Strassen terms handle: A[i, j] * B[j, k] for i,j in {0,1} and k in column pairs.
    # So we need to add:
    # - A[i, 2] * B[2, k] for i in {0,1}, all k
    # - A[2, j] * B[j, k] for j in {0,1,2}, all k
    
    # Add terms for A[0:2, 2] * B[2, k]
    for i in range(2):
        for k in range(8):
            u_vec = np.zeros(n * m, dtype=np.int64)
            v_vec = np.zeros(m * p, dtype=np.int64)
            w_vec = np.zeros(p * n, dtype=np.int64)
            u_vec[i * m + 2] = 1
            v_vec[2 * p + k] = 1
            w_vec[k * n + i] = 1
            U_terms.append(u_vec)
            V_terms.append(v_vec)
            W_terms.append(w_vec)
    
    # Add terms for A[2, :] * B[:, k]
    for j in range(3):
        for k in range(8):
            u_vec = np.zeros(n * m, dtype=np.int64)
            v_vec = np.zeros(m * p, dtype=np.int64)
            w_vec = np.zeros(p * n, dtype=np.int64)
            u_vec[2 * m + j] = 1
            v_vec[j * p + k] = 1
            w_vec[k * n + 2] = 1
            U_terms.append(u_vec)
            V_terms.append(v_vec)
            W_terms.append(w_vec)
    
    # Total terms: 4*7 + 16 + 24 = 28 + 16 + 24 = 68
    rank = len(U_terms)
    U = np.column_stack(U_terms) if U_terms else np.zeros((n*m, 0), dtype=np.int64)
    V = np.column_stack(V_terms) if V_terms else np.zeros((m*p, 0), dtype=np.int64)
    W = np.column_stack(W_terms) if W_terms else np.zeros((p*n, 0), dtype=np.int64)
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try Strassen-based construction first
    U, V, W = _construct_strassen_based(n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        best_rank = U.shape[1]
        best_U, best_V, best_W = U, V, W
    else:
        # Fall back to schoolbook
        best_U, best_V, best_W = _schoolbook(n, m, p)
        best_rank = best_U.shape[1]
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
