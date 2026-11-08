import time
import numpy as np

_DIMS = (3, 4, 6)
_DEADLINE = 120.0
_T0 = time.time()

def _deadline_ok():
    return (time.time() - _T0) < (_DEADLINE - 5.0)

def _build_strassen_decomposition():
    """
    Constructs the rank-66 decomposition for 3x4 * 4x6 matrix multiplication.
    Uses block decomposition: 3 blocks of 3x4x2.
    Each 3x4x2 block:
      - 1x4x2 part: schoolbook rank 8.
      - 2x4x2 part: two 2x2x2 Strassen blocks, rank 7 each -> 14.
    Total per block: 22. Total: 66.
    """
    n, m, p = _DIMS
    terms = []
    
    # Helper to add a rank-1 term
    # a_terms: list of (i, j, coeff)
    # b_terms: list of (j, k, coeff)
    # w_terms: list of (k, i, coeff)
    def add_term(a_terms, b_terms, w_terms):
        terms.append((a_terms, b_terms, w_terms))

    # Strassen 2x2x2 terms for sub-blocks
    # Indices relative to sub-block: i in {0,1}, j in {0,1}, k in {0,1}
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)(B00) -> C10, C11
    # M3 = (A00)(B01-B11) -> C01, C11
    # M4 = (A11)(B10-B00) -> C00, C10
    # M5 = (A00+A01)(B11) -> C00, C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00
    
    def add_strassen_2x2x2(i_off, j_off, k_off):
        # i_off, j_off, k_off are offsets into the full n, m, p dimensions
        # Local indices: i, j, k in 0..1
        # Global: I = i_off + i, J = j_off + j, K = k_off + k
        
        # M1
        add_term(
            [(i_off, j_off, 1), (i_off+1, j_off+1, 1)],
            [(j_off, k_off, 1), (j_off+1, k_off+1, 1)],
            [(k_off, i_off, 1), (k_off+1, i_off+1, 1)]
        )
        # M2
        add_term(
            [(i_off+1, j_off, 1), (i_off+1, j_off+1, 1)],
            [(j_off, k_off, 1)],
            [(k_off, i_off+1, 1), (k_off+1, i_off+1, -1)]
        )
        # M3
        add_term(
            [(i_off, j_off, 1)],
            [(j_off, k_off+1, 1), (j_off+1, k_off+1, -1)],
            [(k_off+1, i_off, 1), (k_off+1, i_off+1, 1)]
        )
        # M4
        add_term(
            [(i_off+1, j_off+1, 1)],
            [(j_off+1, k_off, 1), (j_off, k_off, -1)],
            [(k_off, i_off, 1), (k_off, i_off+1, 1)]
        )
        # M5
        add_term(
            [(i_off, j_off, 1), (i_off, j_off+1, 1)],
            [(j_off+1, k_off+1, 1)],
            [(k_off, i_off, -1), (k_off+1, i_off, 1)]
        )
        # M6
        add_term(
            [(i_off+1, j_off, 1), (i_off, j_off, -1)],
            [(j_off, k_off, 1), (j_off, k_off+1, 1)],
            [(k_off+1, i_off+1, 1)]
        )
        # M7
        add_term(
            [(i_off, j_off+1, 1), (i_off+1, j_off+1, -1)],
            [(j_off+1, k_off, 1), (j_off+1, k_off+1, 1)],
            [(k_off, i_off, 1)]
        )

    def add_schoolbook_1x4x2(i_off, j_off, k_off):
        # n=1, m=4, p=2
        # i is fixed to i_off
        for j in range(4):
            for k in range(2):
                add_term(
                    [(i_off, j_off+j, 1)],
                    [(j_off+j, k_off+k, 1)],
                    [(k_off+k, i_off, 1)]
                )

    # Iterate over the 3 blocks of size 3x4x2
    # p is split into 2+2+2
    for block_idx in range(3):
        k_off = block_idx * 2
        
        # Part 1: i=0 (top row), j=0..3, k=k_off..k_off+1
        # This is a 1x4x2 block
        add_schoolbook_1x4x2(0, 0, k_off)
        
        # Part 2: i=1,2 (bottom 2 rows), j=0..3, k=k_off..k_off+1
        # This is a 2x4x2 block, split into two 2x2x2 blocks on j
        # j=0,1 and j=2,3
        add_strassen_2x2x2(1, 0, k_off)
        add_strassen_2x2x2(1, 2, k_off)

    return terms

def _verify_decomposition(U, V, W, rank):
    """
    Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    Note: The tensor T[i,j,k] is 1 if the multiplication is defined, but here 
    we are decomposing the identity tensor for the bilinear map?
    No, the matrix multiplication tensor T has T[i,j,k] = 1.
    The decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    So for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1.
    """
    n, m, p = _DIMS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape for easier broadcasting if needed, or just loop
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = _DIMS
    terms = _build_strassen_decomposition()
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (a_terms, b_terms, out_terms) in enumerate(terms):
        for (i, j, c) in a_terms:
            U[i * m + j][r] += c
        for (j, k, c) in b_terms:
            V[j * p + k][r] += c
        for (k, i, c) in out_terms:
            W[k * n + i][r] += c
    
    if _verify_decomposition(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: Schoolbook
    return _schoolbook()

def _schoolbook():
    n, m, p = _DIMS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}
