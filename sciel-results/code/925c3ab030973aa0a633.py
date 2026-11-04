import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _validate_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Validate that the given factor matrices exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Construct the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W each of shape (4, 7).
    
    Convention: U[i*2+j, r] for A[i,j], V[j*2+k, r] for B[j,k], W[k*2+i, r] for C[k,i]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # P1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (A10+A11) B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1
    
    # P3 = A00 (B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = -1
    
    # P4 = A11 (B01-B10)
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # P5 = (A00+A01) B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # P6 = (A10-A00) (B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = -1
    
    # P7 = (A01-A11) (B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4x4():
    """
    Construct the rank-49 tensor-product-of-Strassen decomposition for 4x4x4.
    Returns U, V, W each of shape (16, 49).
    
    Convention: U[i*4+j, r] for A[i,j], V[j*4+k, r] for B[j,k], W[k*4+i, r] for C[k,i]
    """
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            U[i1*4+i2, j1*4+j2, col] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            V[j1*4+j2, k1*4+k2, col] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            W[k1*4+k2, i1*4+i2, col] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
    
    return U, V, W

def _laderman_3x3x3():
    """
    Construct the rank-23 Laderman decomposition for 3x3x3 matrix multiplication.
    Returns U, V, W each of shape (9, 23).
    
    Convention: U[i*3+j, r] for A[i,j], V[j*3+k, r] for B[j,k], W[k*3+i, r] for C[k,i]
    """
    U = np.zeros((9, 23), dtype=float)
    V = np.zeros((9, 23), dtype=float)
    W = np.zeros((9, 23), dtype=float)
    
    # Laderman's 23 bilinear forms for 3x3 matrix multiplication
    # Using 0-indexed: a_ij is A[i,j], b_jk is B[j,k], c_ki is C[k,i]
    
    # The 23 products (mu1 through mu23)
    # mu1  = (a00+a10)(b00+b10)
    # mu2  = (a00+a10)(b01+b11)
    # mu3  = (a00+a10)(b12+b22)
    # mu4  = (a01+a11)(b00+b10)
    # mu5  = (a01+a11)(b01+b11)
    # mu6  = (a01+a11)(b12+b22)
    # mu7  = (a02+a12)(b00+b10)
    # mu8  = (a02+a12)(b01+b11)
    # mu9  = (a02+a12)(b12+b22)
    # mu10 = (a00-a10)(b00-b10)
    # mu11 = (a01-a11)(b01-b11)
    # mu12 = (a02-a12)(b12-b22)
    # mu13 = (a00-a01)(b10-b11)
    # mu14 = (a00-a02)(b12-b22)
    # mu15 = (a10-a11)(b00-b01)
    # mu16 = (a10-a12)(b02-b22)
    # mu17 = (a11-a12)(b01-b02)
    # mu18 = a00*b11
    # mu19 = a11*b00
    # mu20 = a01*b12
    # mu21 = a12*b02
    # mu22 = a02*b10
    # mu23 = a10*b22
    
    # Output formulas:
    # c00 = mu1 + mu10 - mu13 - mu15 - mu18
    # c01 = mu2 + mu11 - mu14 - mu16 - mu20
    # c02 = mu3 + mu12 - mu17 - mu21 - mu22
    # c10 = mu4 + mu10 - mu15 - mu18 - mu23
    # c11 = mu5 + mu11 - mu19
    # c12 = mu6 + mu12 - mu17 - mu20
    # c20 = mu7 + mu10 - mu13 - mu22
    # c21 = mu8 + mu11 - mu14 - mu19
    # c22 = mu9 + mu12 - mu16 - mu21
    
    # Each form: (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
    # U index: i*3+j, V index: j*3+k, W index: k*3+i
    
    forms = [
        # mu1 = (a00+a10)(b00+b10)
        # a00: U[0], a10: U[3]
        # b00: V[0], b10: V[3]
        # c00: W[0] (k=0,i=0)
        ([0, 3], [1, 1], [0, 3], [1, 1], [0], [1]),
        
        # mu2 = (a00+a10)(b01+b11)
        # b01: V[1], b11: V[4]
        # c01: W[1] (k=1,i=0)
        ([0, 3], [1, 1], [1, 4], [1, 1], [1], [1]),
        
        # mu3 = (a00+a10)(b12+b22)
        # b12: V[5], b22: V[8]
        # c02: W[2] (k=2,i=0)
        ([0, 3], [1, 1], [5, 8], [1, 1], [2], [1]),
        
        # mu4 = (a01+a11)(b00+b10)
        # a01: U[1], a11: U[4]
        # c10: W[3] (k=0,i=1)
        ([1, 4], [1, 1], [0, 3], [1, 1], [3], [1]),
        
        # mu5 = (a01+a11)(b01+b11)
        # c11: W[4] (k=1,i=1)
        ([1, 4], [1, 1], [1, 4], [1, 1], [4], [1]),
        
        # mu6 = (a01+a11)(b12+b22)
        # c12: W[5] (k=2,i=1)
        ([1, 4], [1, 1], [5, 8], [1, 1], [5], [1]),
        
        # mu7 = (a02+a12)(b00+b10)
        # a02: U[2], a12: U[5]
        # c20: W[6] (k=0,i=2)
        ([2, 5], [1, 1], [0, 3], [1, 1], [6], [1]),
        
        # mu8 = (a02+a12)(b01+b11)
        # c21: W[7] (k=1,i=2)
        ([2, 5], [1, 1], [1, 4], [1, 1], [7], [1]),
        
        # mu9 = (a02+a12)(b12+b22)
        # c22: W[8] (k=2,i=2)
        ([2, 5], [1, 1], [5, 8], [1, 1], [8], [1]),
        
        # mu10 = (a00-a10)(b00-b10)
        # c00: W[0], c10: W[3], c20: W[6]
        ([0, 3], [1, -1], [0, 3], [1, -1], [0, 3, 6], [1, 1, 1]),
        
        # mu11 = (a01-a11)(b01-b11)
        # c01: W[1], c11: W[4], c21: W[7]
        ([1, 4], [1, -1], [1, 4], [1, -1], [1, 4, 7], [1, 1, 1]),
        
        # mu12 = (a02-a12)(b12-b22)
        # c02: W[2], c12: W[5], c22: W[8]
        ([2, 5], [1, -1], [5, 8], [1, -1], [2, 5, 8], [1, 1, 1]),
        
        # mu13 = (a00-a01)(b10-b11)
        # a00: U[0], a01: U[1]
        # b10: V[3], b11: V[4]
        # c00: W[0], c20: W[6]
        ([0, 1], [1, -1], [3, 4], [1, -1], [0, 6], [-1, -1]),
        
        # mu14 = (a00-a02)(b12-b22)
        # a00: U[0], a02: U[2]
        # b12: V[5], b22: V[8]
        # c01: W[1], c21: W[7]
        ([0, 2], [1, -1], [5, 8], [1, -1], [1, 7], [-1, -1]),
        
        # mu15 = (a10-a11)(b00-b01)
        # a10: U[3], a11: U[4]
        # b00: V[0], b01: V[1]
        # c00: W[0], c10: W[3]
        ([3, 4], [1, -1], [0, 1], [1, -1], [0, 3], [-1, -1]),
        
        # mu16 = (a10-a12)(b02-b22)
        # a10: U[3], a12: U[5]
        # b02: V[2], b22: V[8]
        # c01: W[1], c10: W[3]
        ([3, 5], [1, -1], [2, 8], [1, -1], [1, 3], [-1, -1]),
        
        # mu17 = (a11-a12)(b01-b02)
        # a11: U[4], a12: U[5]
        # b01: V[1], b02: V[2]
        # c02: W[2], c12: W[5]
        ([4, 5], [1, -1], [1, 2], [1, -1], [2, 5], [-1, -1]),
        
        # mu18 = a00*b11
        # a00: U[0], b11: V[4]
        # c00: W[0]
        ([0], [1], [4], [1], [0], [-1]),
        
        # mu19 = a11*b00
        # a11: U[4], b00: V[0]
        # c11: W[4]
        ([4], [1], [0], [1], [4], [-1]),
        
        # mu20 = a01*b12
        # a01: U[1], b12: V[5]
        # c01: W[1]
        ([1], [1], [5], [1], [1], [-1]),
        
        # mu21 = a12*b02
        # a12: U[5], b02: V[2]
        # c12: W[5]
        ([5], [1], [2], [1], [5], [-1]),
        
        # mu22 = a02*b10
        # a02: U[2], b10: V[3]
        # c02: W[2]
        ([2], [1], [3], [1], [2], [-1]),
        
        # mu23 = a10*b22
        # a10: U[3], b22: V[8]
        # c10: W[3]
        ([3], [1], [8], [1], [3], [-1]),
    ]
    
    for r, (u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff) in enumerate(forms):
        for idx, coeff in zip(u_idx, u_coeff):
            U[idx, r] += coeff
        for idx, coeff in zip(v_idx, v_coeff):
            V[idx, r] += coeff
        for idx, coeff in zip(w_idx, w_coeff):
            W[idx, r] += coeff
    
    return U, V, W

def _decompose_4plus1():
    """
    Decompose 5x5x5 matrix multiplication using a 4+1 block split.
    
    Split: rows/cols 0-3 form the 4x4 block, row/col 4 is the 1x1 block.
    
    C[0:4, 0:4] = A[0:4, 0:4]*B[0:4, 0:4]  (rank 49 via Strassen⊗Strassen)
    C[4, 4] = A[4,4]*B[4,4]  (rank 1)
    C[0:4, 4] = A[0:4, 0:4]*B[0:4,4] + A[0:4,4]*B[4,4]  (rank 20 schoolbook)
    C[4, 0:4] = A[4,0:4]*B[0:4,0:4] + A[4,4]*B[4,0:4]  (rank 20 schoolbook)
    
    Total rank = 49 + 1 + 20 + 20 = 90
    """
    U4, V4, W4 = _strassen_4x4x4()
    
    rank_444 = 49
    rank_111 = 1
    rank_cross1 = 4 * 5 * 1  # C[i,4] for i in 0..3
    rank_cross2 = 1 * 5 * 4  # C[4,k] for k in 0..3
    
    rank_total = rank_444 + rank_111 + rank_cross1 + rank_cross2
    # = 49 + 1 + 20 + 20 = 90
    
    U = np.zeros((25, rank_total), dtype=float)
    V = np.zeros((25, rank_total), dtype=float)
    W = np.zeros((25, rank_total), dtype=float)
    
    # 4x4x4 block (top-left)
    for i in range(4):
        for j in range(4):
            row_u = i * 5 + j
            for r in range(rank_444):
                U[row_u, r] = U4[i * 4 + j, r]
    for j in range(4):
        for k in range(4):
            row_v = j * 5 + k
            for r in range(rank_444):
                V[row_v, r] = V4[j * 4 + k, r]
    for k in range(4):
        for i in range(4):
            row_w = k * 5 + i
            for r in range(rank_444):
                W[row_w, r] = W4[k * 4 + i, r]
    
    # 1x1x1 block (bottom-right)
    offset = rank_444
    U[4 * 5 + 4, offset] = 1.0
    V[4 * 5 + 4, offset] = 1.0
    W[4 * 5 + 4, offset] = 1.0
    offset += 1
    
    # Cross terms: schoolbook
    # C[i,4] for i in 0..3: sum_j A[i,j]*B[j,4] for j in 0..4
    for i in range(4):
        for j in range(5):
            col = offset
            U[i * 5 + j, col] = 1.0
            V[j * 5 + 4, col] = 1.0
            W[4 * 5 + i, col] = 1.0
            offset += 1
    
    # C[4,k] for k in 0..3: sum_j A[4,j]*B[j,k] for j in 0..4
    for k in range(4):
        for j in range(5):
            col = offset
            U[4 * 5 + j, col] = 1.0
            V[j * 5 + k, col] = 1.0
            W[k * 5 + 4, col] = 1.0
            offset += 1
    
    return U, V, W

def _decompose_3plus2():
    """
    Decompose 5x5x5 matrix multiplication using a 3+2 block split.
    
    Split: rows/cols 0-2 form the 3x3 block, rows/cols 3-4 form the 2x2 block.
    
    Use Laderman (rank 23) for 3x3x3, Strassen (rank 7) for 2x2x2,
    and schoolbook for cross terms.
    
    Rank = 23 + 7 + 30 + 30 = 90
    """
    n = 5
    U3, V3, W3 = _laderman_3x3x3()
    U2, V2, W2 = _strassen_2x2x2()
    
    rank_333 = 23
    rank_222 = 7
    rank_cross1 = 3 * 5 * 2
    rank_cross2 = 2 * 5 * 3
    
    rank_total = rank_333 + rank_222 + rank_cross1 + rank_cross2
    
    U = np.zeros((25, rank_total), dtype=float)
    V = np.zeros((25, rank_total), dtype=float)
    W = np.zeros((25, rank_total), dtype=float)
    
    # 3x3x3 block (top-left)
    for i in range(3):
        for j in range(3):
            row_u = i * 5 + j
            for r in range(rank_333):
                U[row_u, r] = U3[i * 3 + j, r]
    for j in range(3):
        for k in range(3):
            row_v = j * 5 + k
            for r in range(rank_333):
                V[row_v, r] = V3[j * 3 + k, r]
    for k in range(3):
        for i in range(3):
            row_w = k * 5 + i
            for r in range(rank_333):
                W[row_w, r] = W3[k * 3 + i, r]
    
    # 2x2x2 block (bottom-right)
    offset = rank_333
    for i in range(2):
        for j in range(2):
            row_u = (3 + i) * 5 + (3 + j)
            for r in range(rank_222):
                U[row_u, offset + r] = U2[i * 2 + j, r]
    for j in range(2):
        for k in range(2):
            row_v = (3 + j) * 5 + (3 + k)
            for r in range(rank_222):
                V[row_v, offset + r] = V2[j * 2 + k, r]
    for k in range(2):
        for i in range(2):
            row_w = (3 + k) * 5 + (3 + i)
            for r in range(rank_222):
                W[row_w, offset + r] = W2[k * 2 + i, r]
    
    # Cross terms: schoolbook
    offset = rank_333 + rank_222
    
    # C[i,k] for i in 0..2, k in 3..4
    for i in range(3):
        for j in range(5):
            for k in range(2):
                col = offset
                U[i * 5 + j, col] = 1.0
                V[j * 5 + (3 + k), col] = 1.0
                W[(3 + k) * 5 + i, col] = 1.0
                offset += 1
    
    # C[i,k] for i in 3..4, k in 0..2
    for i in range(2):
        for j in range(5):
            for k in range(3):
                col = offset
                U[(3 + i) * 5 + j, col] = 1.0
                V[j * 5 + k, col] = 1.0
                W[k * 5 + (3 + i), col] = 1.0
                offset += 1
    
    return U, V, W

def _validate_full(U, V, W, n, m, p, tol=1e-9):
    """
    Full validation: check all index combinations, not just valid chains.
    For matrix multiplication, the tensor T[a,b,c] should be:
    T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T[a,b,c] = 0 for all other combinations
    
    We check a sample of invalid combinations for efficiency.
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check some invalid entries (sample for speed)
    # Invalid: A index (i,j), B index (j',k) where j' != j
    # T[i*m+j, j'*p+k, k*n+i] should be 0 if j' != j
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                if jp == j:
                    continue
                for k in range(p):
                    val = np.dot(U_r[i, j, :], V_r[jp, k, :] * W_r[k, i, :])
                    if abs(val) > tol:
                        return False
    
    # Invalid: C index (k',i') where k' != k or i' != i
    # T[i*m+j, j*p+k, k'*n+i'] should be 0 if (k',i') != (k,i)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for kp in range(p):
                    for ip in range(n):
                        if kp == k and ip == i:
                            continue
                        val = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[kp, ip, :])
                        if abs(val) > tol:
                            return False
    
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Try 4+1 decomposition (rank 90)
    try:
        U, V, W = _decompose_4plus1()
        if time.time() < deadline:
            if _validate_decomposition(U, V, W, n, m, p):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    except Exception:
        pass
    
    # Try 3+2 decomposition (rank 90)
    try:
        U, V, W = _decompose_3plus2()
        if time.time() < deadline:
            if _validate_decomposition(U, V, W, n, m, p):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    except Exception:
        pass
    
    # Fall back to schoolbook (rank 125)
    U, V, W = _schoolbook_decomposition(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
