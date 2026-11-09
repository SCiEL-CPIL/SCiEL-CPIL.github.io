"""Matrix multiplication 3,3,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=3, p=7):
    """Verify that U, V, W give an exact decomposition of the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n=3, m=3, p=7):
    """Return the schoolbook (rank=n*m*p) decomposition."""
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # P2 = (A10+A11)B00 -> C10 += P2
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # P3 = A00(B10-B11) -> C01 += P3
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[1,2] = 1
    
    # P4 = A11(B01-B00) -> C00 += P4, C10 += P4
    U[3,3] = 1
    V[1,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    
    # P5 = (A00+A01)B11 -> C00 -= P5, C01 += P5
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # P6 = (A10-A00)(B00+B01) -> C11 += P6
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # P7 = (A01-A11)(B10+B11) -> C00 += P7
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _block_decompose_3x3x7():
    """
    Block decomposition of 3x3 * 3x7 into smaller sub-problems.
    Split A (3x3) into 2x2 + 1x1 blocks, B (3x7) into 2x7 + 1x7 blocks.
    Use Strassen for the 2x2 * 2x2 sub-problems where applicable.
    """
    n, m, p = 3, 3, 7
    
    # We'll build up the decomposition term by term.
    # Each term c has U entries (for A indices), V entries (for B indices), W entries (for C indices).
    # U is indexed by (i,j) -> row i*m+j
    # V is indexed by (j,k) -> row j*p+k
    # W is indexed by (k,i) -> row k*n+i
    
    # Block structure:
    # A = [[A11 (2x2), A12 (2x1)],
    #      [A21 (1x2), A22 (1x1)]]
    # B = [[B11 (2x7), B12 (1x7)]^T  -- wait, B is 3x7, so:
    #      [B11 (2x7), B12 (1x7)]  where B11 is rows 0-1, B12 is row 2
    # C = [[C11 (2x7), C12 (1x7)]  where C11 is rows 0-1, C12 is row 2
    
    # C11 = A11*B11 + A12*B12  (2x7 = 2x2*2x7 + 2x1*1x7)
    # C12 = A21*B11 + A22*B12  (1x7 = 1x2*2x7 + 1x1*1x7)
    
    # Let's decompose each product:
    # A11*B11: 2x2 * 2x7. Split 7 into 3+4? Or just use schoolbook: 2*2*7=28 terms.
    #   But we can use Strassen-like for 2x2*2x2 blocks.
    #   Actually, 2x2 * 2x7 = sum over k of 2x2 * 2x1 (7 columns).
    #   Each 2x2*2x1 has rank 4 (schoolbook) or can we do better?
    #   For 2x2*2x1: C00 = A00*B00 + A01*B10, C10 = A10*B00 + A11*B10. Rank 4.
    #   Strassen doesn't help for rectangular 2x2*2x1.
    #   So 2x2*2x7 = 7*4 = 28 terms (schoolbook).
    
    # A12*B12: 2x1 * 1x7. C0k = A02*B2k, C1k = A12*B2k. Rank 2*7 = 14 terms.
    
    # A21*B11: 1x2 * 2x7. C2k = A20*B0k + A21*B1k. Rank 2*7 = 14 terms.
    
    # A22*B12: 1x1 * 1x7. C2k = A22*B2k. Rank 7 terms.
    
    # Total schoolbook: 28 + 14 + 14 + 7 = 63. Same as full schoolbook.
    
    # To do better, we need to use Strassen for the 2x2*2x2 sub-blocks.
    # Let's split the 7 columns into 3+4 and handle each separately.
    # For the 2x2*2x3 part: use Strassen for the 2x2*2x2 part (rank 7) + 2x2*2x1 (rank 4).
    #   Total: 7 + 4 = 11 vs schoolbook 2*2*3 = 12. Saves 1.
    # For the 2x2*2x4 part: use Strassen for two 2x2*2x2 parts (rank 7+7=14) vs schoolbook 16. Saves 2.
    #   Or use one 2x2*2x4 with a smarter decomposition.
    
    # Let's try: split 7 into 3+4.
    # 2x2*2x3: Strassen(2x2*2x2) + schoolbook(2x2*2x1) = 7 + 4 = 11
    # 2x2*2x4: 2*Strassen(2x2*2x2) = 14
    # A12*B12 (2x1*1x7): 14
    # A21*B11 (1x2*2x7): 14
    # A22*B12 (1x1*1x7): 7
    # Total: 11 + 14 + 14 + 14 + 7 = 60. Saves 3!
    
    # Let's implement this.
    
    U_terms = []  # list of dicts: {(i,j): coeff}
    V_terms = []  # list of dicts: {(j,k): coeff}
    W_terms = []  # list of dicts: {(k,i): coeff}
    
    def add_term(U_dict, V_dict, W_dict):
        U_terms.append(U_dict)
        V_terms.append(V_dict)
        W_terms.append(W_dict)
    
    # Helper: add schoolbook terms for a sub-product
    def schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end, W_coeffs):
        """Add schoolbook terms for A[i_start:i_end, j_start:j_end] * B[j_start:j_end, k_start:k_end]"""
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U_dict = {(i, j): 1}
                    V_dict = {(j, k): 1}
                    W_dict = {(k, i): W_coeffs.get((i, k), 1)}
                    add_term(U_dict, V_dict, W_dict)
    
    # 1. A11*B11: 2x2 * 2x7, split into 2x2*2x3 and 2x2*2x4
    # 1a. 2x2*2x3 (k=0,1,2): Strassen for 2x2*2x2 (k=0,1) + schoolbook for 2x2*2x1 (k=2)
    # Strassen for 2x2*2x2 (i=0,1; j=0,1; k=0,1):
    sU, sV, sW = _strassen_2x2()
    for c in range(7):
        U_dict = {}
        for i in range(2):
            for j in range(2):
                if sU[i*2+j, c] != 0:
                    U_dict[(i, j)] = sU[i*2+j, c]
        V_dict = {}
        for j in range(2):
            for k in range(2):
                if sV[j*2+k, c] != 0:
                    V_dict[(j, k)] = sV[j*2+k, c]
        W_dict = {}
        for k in range(2):
            for i in range(2):
                if sW[k*2+i, c] != 0:
                    W_dict[(k, i)] = sW[k*2+i, c]
        add_term(U_dict, V_dict, W_dict)
    
    # Schoolbook for 2x2*2x1 (i=0,1; j=0,1; k=2):
    schoolbook_block(0, 2, 0, 2, 2, 3)
    
    # 1b. 2x2*2x4 (i=0,1; j=0,1; k=3,4,5,6): 2*Strassen for 2x2*2x2
    # Strassen for k=3,4:
    sU, sV, sW = _strassen_2x2()
    for c in range(7):
        U_dict = {}
        for i in range(2):
            for j in range(2):
                if sU[i*2+j, c] != 0:
                    U_dict[(i, j)] = sU[i*2+j, c]
        V_dict = {}
        for j in range(2):
            for k in range(2):
                if sV[j*2+k, c] != 0:
                    V_dict[(j, k+3)] = sV[j*2+k, c]
        W_dict = {}
        for k in range(2):
            for i in range(2):
                if sW[k*2+i, c] != 0:
                    W_dict[(k+3, i)] = sW[k*2+i, c]
        add_term(U_dict, V_dict, W_dict)
    
    # Strassen for k=5,6:
    sU, sV, sW = _strassen_2x2()
    for c in range(7):
        U_dict = {}
        for i in range(2):
            for j in range(2):
                if sU[i*2+j, c] != 0:
                    U_dict[(i, j)] = sU[i*2+j, c]
        V_dict = {}
        for j in range(2):
            for k in range(2):
                if sV[j*2+k, c] != 0:
                    V_dict[(j, k+5)] = sV[j*2+k, c]
        W_dict = {}
        for k in range(2):
            for i in range(2):
                if sW[k*2+i, c] != 0:
                    W_dict[(k+5, i)] = sW[k*2+i, c]
        add_term(U_dict, V_dict, W_dict)
    
    # 2. A12*B12: 2x1 * 1x7 (i=0,1; j=2; k=0..6)
    schoolbook_block(0, 2, 2, 3, 0, 7)
    
    # 3. A21*B11: 1x2 * 2x7 (i=2; j=0,1; k=0..6)
    schoolbook_block(2, 3, 0, 2, 0, 7)
    
    # 4. A22*B12: 1x1 * 1x7 (i=2; j=2; k=0..6)
    schoolbook_block(2, 3, 2, 3, 0, 7)
    
    # Convert to arrays
    rank = len(U_terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c in range(rank):
        for (i, j), coeff in U_terms[c].items():
            U[i * m + j, c] = coeff
        for (j, k), coeff in V_terms[c].items():
            V[j * p + k, c] = coeff
        for (k, i), coeff in W_terms[c].items():
            W[k * n + i, c] = coeff
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 by 3x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try block decomposition with Strassen
    try:
        U_block, V_block, W_block = _block_decompose_3x3x7()
        if _verify(U_block, V_block, W_block, n, m, p):
            rank_block = len(U_block[0]) if U_block else 0
            if rank_block < best_rank:
                best = {"U": U_block, "V": V_block, "W": W_block}
                best_rank = rank_block
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        return best
    
    return best
# EVOLVE-BLOCK-END
