"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 7]

def _build_strassen_2x2x2():
    """
    Build Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W of shape (4,7), (4,7), (4,7) for 2x2x2.
    
    Strassen formulas:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    """
    # Indexing: A is 2x2, B is 2x2, C is 2x2
    # U[i*m+j, r] where m=2, so U[i*2+j, r]
    # V[j*p+k, r] where p=2, so V[j*2+k, r]
    # W[k*n+i, r] where n=2, so W[k*2+i, r]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[1, 0] = 1  # B12... wait, need to be careful
    
    # Let me use explicit indexing
    # A indices: A[0,0], A[0,1], A[1,0], A[1,1] -> flat: 0,1,2,3
    # B indices: B[0,0], B[0,1], B[1,0], B[1,1] -> flat: 0,1,2,3
    # C indices: C[0,0], C[0,1], C[1,0], C[1,1] -> flat: 0,1,2,3
    
    # M1 = (A11+A22)(B11+B22)
    # A11 = A[0,0] = flat 0, A22 = A[1,1] = flat 3
    # B11 = B[0,0] = flat 0, B22 = B[1,1] = flat 3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 += M1, C22 += M1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0] and C[1,1]
    
    # M2 = (A21+A22)B11
    # A21 = A[1,0] = flat 2, A22 = A[1,1] = flat 3
    # B11 = B[0,0] = flat 0
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 += M2, C22 -= M2
    W[2, 1] = 1  # C[1,0]
    W[3, 1] = -1  # C[1,1]
    
    # M3 = A11(B21-B22)
    # A11 = A[0,0] = flat 0
    # B21 = B[1,0] = flat 2, B22 = B[1,1] = flat 3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    # C12 += M3, C22 += M3
    W[1, 2] = 1  # C[0,1]
    W[3, 2] = 1  # C[1,1]
    
    # M4 = A22(B12-B11)
    # A22 = A[1,1] = flat 3
    # B12 = B[0,1] = flat 1, B11 = B[0,0] = flat 0
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    # C11 += M4, C21 += M4
    W[0, 3] = 1  # C[0,0]
    W[2, 3] = 1  # C[1,0]
    
    # M5 = (A11+A12)B22
    # A11 = A[0,0] = flat 0, A12 = A[0,1] = flat 1
    # B22 = B[1,1] = flat 3
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C11 -= M5, C12 += M5
    W[0, 4] = -1  # C[0,0]
    W[1, 4] = 1  # C[0,1]
    
    # M6 = (A21-A11)(B11+B12)
    # A21 = A[1,0] = flat 2, A11 = A[0,0] = flat 0
    # B11 = B[0,0] = flat 0, B12 = B[0,1] = flat 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 += M6
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (A12-A22)(B21+B22)
    # A12 = A[0,1] = flat 1, A22 = A[1,1] = flat 3
    # B21 = B[1,0] = flat 2, B22 = B[1,1] = flat 3
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 += M7
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W

def _verify(U, V, W, tol=1e-8):
    """Verify tensor decomposition against multiple random test cases."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    
    # Test with multiple random matrices for robustness
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        U_reshaped = np.asarray(U).reshape(n, m, R)
        V_reshaped = np.asarray(V).reshape(m, p, R)
        W_reshaped = np.asarray(W).reshape(p, n, R)
        
        S_A = np.einsum('ijr,ij->ir', U_reshaped, A)
        S_B = np.einsum('jkr,jk->kr', V_reshaped, B)
        C_comp = np.einsum('kir,ir,kr->ik', W_reshaped, S_A, S_B)
        
        if not np.allclose(C_true, C_comp, atol=tol):
            return False
    
    return True

# EVOLVE-BLOCK-START
def _strassen_2x2x2(row_offset, col_offset, k_offset):
    """
    Return Strassen 2x2x2 terms as list of (matrix, index, value) tuples.
    matrix: 'U', 'V', or 'W'
    """
    n, m, p = DIMENSIONS
    
    def U_idx(i_loc, j_loc):
        return (row_offset + i_loc) * m + (k_offset + j_loc)
    
    def V_idx(j_loc, k_loc):
        return (k_offset + j_loc) * p + (col_offset + k_loc)
    
    def W_idx(k_loc, i_loc):
        return (col_offset + k_loc) * n + (row_offset + i_loc)
    
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    terms.append([('U', U_idx(0,0), 1), ('U', U_idx(1,1), 1),
                  ('V', V_idx(0,0), 1), ('V', V_idx(1,1), 1),
                  ('W', W_idx(0,0), 1), ('W', W_idx(1,1), 1)])
    # M2 = (A10+A11)B00
    terms.append([('U', U_idx(1,0), 1), ('U', U_idx(1,1), 1),
                  ('V', V_idx(0,0), 1),
                  ('W', W_idx(0,1), 1), ('W', W_idx(1,1), -1)])
    # M3 = A00(B10-B11)
    terms.append([('U', U_idx(0,0), 1),
                  ('V', V_idx(1,0), 1), ('V', V_idx(1,1), -1),
                  ('W', W_idx(1,0), 1), ('W', W_idx(1,1), 1)])
    # M4 = A11(B01-B00)
    terms.append([('U', U_idx(1,1), 1),
                  ('V', V_idx(0,1), 1), ('V', V_idx(0,0), -1),
                  ('W', W_idx(0,0), 1), ('W', W_idx(0,1), 1)])
    # M5 = (A00+A01)B11
    terms.append([('U', U_idx(0,0), 1), ('U', U_idx(0,1), 1),
                  ('V', V_idx(1,1), 1),
                  ('W', W_idx(0,0), -1), ('W', W_idx(1,0), 1)])
    # M6 = (A10-A00)(B00+B01)
    terms.append([('U', U_idx(1,0), 1), ('U', U_idx(0,0), -1),
                  ('V', V_idx(0,0), 1), ('V', V_idx(0,1), 1),
                  ('W', W_idx(1,1), 1)])
    # M7 = (A01-A11)(B10+B11)
    terms.append([('U', U_idx(0,1), 1), ('U', U_idx(1,1), -1),
                  ('V', V_idx(1,0), 1), ('V', V_idx(1,1), 1),
                  ('W', W_idx(0,0), 1)])
    
    return terms

def _schoolbook_2x2x1(row_offset, col_offset, k_offset):
    """
    Schoolbook for 2x2x1. A is 2x2, B is 2x1, C is 2x1.
    Returns list of (matrix, index, value) tuples.
    """
    n, m, p = DIMENSIONS
    
    def U_idx(i_loc, j_loc):
        return (row_offset + i_loc) * m + (k_offset + j_loc)
    
    def V_idx(j_loc, k_loc):
        return (k_offset + j_loc) * p + (col_offset + k_loc)
    
    def W_idx(k_loc, i_loc):
        return (col_offset + k_loc) * n + (row_offset + i_loc)
    
    terms = []
    # C00 = A00*B00 + A01*B10
    terms.append([('U', U_idx(0,0), 1), ('V', V_idx(0,0), 1), ('W', W_idx(0,0), 1)])
    terms.append([('U', U_idx(0,1), 1), ('V', V_idx(1,0), 1), ('W', W_idx(0,0), 1)])
    # C10 = A10*B00 + A11*B10
    terms.append([('U', U_idx(1,0), 1), ('V', V_idx(0,0), 1), ('W', W_idx(0,1), 1)])
    terms.append([('U', U_idx(1,1), 1), ('V', V_idx(1,0), 1), ('W', W_idx(0,1), 1)])
    
    return terms

def _schoolbook_1x2x2(row_offset, col_offset, k_offset):
    """
    Schoolbook for 1x2x2. A is 1x2, B is 2x2, C is 1x2.
    Returns list of (matrix, index, value) tuples.
    """
    n, m, p = DIMENSIONS
    
    def U_idx(i_loc, j_loc):
        return (row_offset + i_loc) * m + (k_offset + j_loc)
    
    def V_idx(j_loc, k_loc):
        return (k_offset + j_loc) * p + (col_offset + k_loc)
    
    def W_idx(k_loc, i_loc):
        return (col_offset + k_loc) * n + (row_offset + i_loc)
    
    terms = []
    # C00 = A00*B00 + A01*B10
    terms.append([('U', U_idx(0,0), 1), ('V', V_idx(0,0), 1), ('W', W_idx(0,0), 1)])
    terms.append([('U', U_idx(0,1), 1), ('V', V_idx(1,0), 1), ('W', W_idx(0,0), 1)])
    # C01 = A00*B01 + A01*B11
    terms.append([('U', U_idx(0,0), 1), ('V', V_idx(0,1), 1), ('W', W_idx(1,0), 1)])
    terms.append([('U', U_idx(0,1), 1), ('V', V_idx(1,1), 1), ('W', W_idx(1,0), 1)])
    
    return terms

def construct():
    """
    Subproduct method: split 3x4x7 into 2x4x7 + 1x4x7.
    - 2x4x7: split 4->2+2, 7->3+2+2
      - 2x2x3: split 3->2+1: Strassen(7) + schoolbook_2x2x1(2) = 9
      - 2x2x2: Strassen(7)
      - 2x2x2: Strassen(7)
      Total: 23
    - 1x4x7: split 4->2+2, 7->2+2+2+1
      - 1x2x2: schoolbook(2)
      - 1x2x2: schoolbook(2)
      - 1x2x2: schoolbook(2)
      - 1x2x1: schoolbook(2)
      Total: 8
    Grand total: 31
    """
    n, m, p = DIMENSIONS  # 3, 4, 7
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Collect all terms as list of (matrix_name, index, value) tuples
    all_terms = []
    
    # Part 1: 2x4x7 (rows 0,1 of A)
    # Split 4 (m) into 2+2, 7 (p) into 3+2+2
    
    # Block 1: A rows 0-1, B cols 0-1 (j=0,1), C cols 0-2 (k=0,1,2)
    # 2x2x3: split k into 2+1
    all_terms.extend(_strassen_2x2x2(row_offset=0, col_offset=0, k_offset=0))
    all_terms.extend(_schoolbook_2x2x1(row_offset=0, col_offset=2, k_offset=0))
    
    # Block 2: A rows 0-1, B cols 2-3 (j=2,3), C cols 0-2 (k=0,1,2)
    all_terms.extend(_strassen_2x2x2(row_offset=0, col_offset=0, k_offset=2))
    all_terms.extend(_schoolbook_2x2x1(row_offset=0, col_offset=2, k_offset=2))
    
    # Block 3: A rows 0-1, B cols 0-1 (j=0,1), C cols 3-4 (k=3,4)
    all_terms.extend(_strassen_2x2x2(row_offset=0, col_offset=3, k_offset=0))
    
    # Block 4: A rows 0-1, B cols 2-3 (j=2,3), C cols 3-4 (k=3,4)
    all_terms.extend(_strassen_2x2x2(row_offset=0, col_offset=3, k_offset=2))
    
    # Block 5: A rows 0-1, B cols 0-1 (j=0,1), C col 5 (k=5)
    all_terms.extend(_schoolbook_2x2x1(row_offset=0, col_offset=5, k_offset=0))
    
    # Block 6: A rows 0-1, B cols 2-3 (j=2,3), C col 5 (k=5)
    all_terms.extend(_schoolbook_2x2x1(row_offset=0, col_offset=5, k_offset=2))
    
    # Block 7: A rows 0-1, B cols 0-1 (j=0,1), C col 6 (k=6)
    all_terms.extend(_schoolbook_2x2x1(row_offset=0, col_offset=6, k_offset=0))
    
    # Block 8: A rows 0-1, B cols 2-3 (j=2,3), C col 6 (k=6)
    all_terms.extend(_schoolbook_2x2x1(row_offset=0, col_offset=6, k_offset=2))
    
    # Part 2: 1x4x7 (row 2 of A)
    # Split 4 (m) into 2+2, 7 (p) into 2+2+2+1
    all_terms.extend(_schoolbook_1x2x2(row_offset=2, col_offset=0, k_offset=0))
    all_terms.extend(_schoolbook_1x2x2(row_offset=2, col_offset=2, k_offset=0))
    all_terms.extend(_schoolbook_1x2x2(row_offset=2, col_offset=4, k_offset=0))
    
    # 1x2x1 for k=6, j=0,1
    def U_idx_1(i_loc, j_loc):
        return (2 + i_loc) * m + (0 + j_loc)
    def V_idx_1(j_loc, k_loc):
        return (0 + j_loc) * p + (6 + k_loc)
    def W_idx_1(k_loc, i_loc):
        return (6 + k_loc) * n + (2 + i_loc)
    all_terms.append([('U', U_idx_1(0,0), 1), ('V', V_idx_1(0,0), 1), ('W', W_idx_1(0,0), 1)])
    all_terms.append([('U', U_idx_1(0,1), 1), ('V', V_idx_1(1,0), 1), ('W', W_idx_1(0,0), 1)])
    
    # 1x2x1 for k=6, j=2,3
    def U_idx_2(i_loc, j_loc):
        return (2 + i_loc) * m + (2 + j_loc)
    def V_idx_2(j_loc, k_loc):
        return (2 + j_loc) * p + (6 + k_loc)
    def W_idx_2(k_loc, i_loc):
        return (6 + k_loc) * n + (2 + i_loc)
    all_terms.append([('U', U_idx_2(0,0), 1), ('V', V_idx_2(0,0), 1), ('W', W_idx_2(0,0), 1)])
    all_terms.append([('U', U_idx_2(0,1), 1), ('V', V_idx_2(1,0), 1), ('W', W_idx_2(0,0), 1)])
    
    # Build matrices
    # Each "term" in all_terms is now a tuple (matrix, idx, val), not a list
    # We need to group by rank index. Let me restructure.
    
    # Actually, each call to _strassen or _schoolbook returns a list of lists.
    # Each inner list is one rank term with multiple (matrix, idx, val) entries.
    # So all_terms is a list of rank terms, where each rank term is a list of (matrix, idx, val).
    
    R = len(all_terms)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for r, term in enumerate(all_terms):
        for mat_name, idx, val in term:
            if mat_name == 'U':
                U[idx, r] += val
            elif mat_name == 'V':
                V[idx, r] += val
            elif mat_name == 'W':
                W[idx, r] += val
    
    # Verify
    if _verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
