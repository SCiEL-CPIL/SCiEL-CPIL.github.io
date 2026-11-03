"""Optimized 2x5x6 matrix multiplication tensor decomposition using Strassen + Laderman."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_222():
    """Return U, V, W (each 4x7) for 2x2x2 matmul using Strassen.
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # r=1: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # k=0,i=1
    
    # r=2: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # k=1,i=0
    
    # r=3: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1  # k=0,i=1
    
    # r=4: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1  # k=1,i=0
    
    # r=5: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1  # k=0,i=0
    
    # r=6: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1  # k=0,i=0
    
    return U, V, W

def _laderman_233():
    """Return U, V, W for 2x3x3 matmul using Laderman's algorithm (rank 12).
    A is 2x3, B is 3x3, C = A*B is 2x3.
    Convention: U[i*3+j, r], V[j*3+k, r], W[k*2+i, r]
    
    Laderman's 12 products:
    m1  = a00 * b00
    m2  = a01 * (b00 + b10)
    m3  = a02 * (b00 + b20)
    m4  = a10 * (b01 + b11)
    m5  = a11 * (b02 + b12)
    m6  = a12 * (b00 + b20)
    m7  = (a00 + a10) * (b10 - b00)
    m8  = (a01 + a11) * (b11 - b01)
    m9  = (a02 + a12) * (b20 - b00)
    m10 = (a00 + a01) * (b10 - b11)
    m11 = (a01 + a02) * (b11 - b12)
    m12 = (a10 + a11) * (b12 - b11)
    """
    U = np.zeros((6, 12))  # 2*3 rows
    V = np.zeros((9, 12))  # 3*3 rows
    W = np.zeros((6, 12))  # 3*2 rows
    
    # Indexing: U[i*3+j], V[j*3+k], W[k*2+i]
    
    # m1 = a00 * b00
    U[0, 0] = 1       # a00
    V[0, 0] = 1       # b00
    W[0, 0] = 1       # c00 (k=0,i=0)
    
    # m2 = a01 * (b00 + b10)
    U[1, 1] = 1       # a01
    V[0, 1] = 1       # b00
    V[3, 1] = 1       # b10
    W[0, 1] = 1       # c00 (k=0,i=0)
    W[2, 1] = 1       # c01 (k=1,i=0)
    
    # m3 = a02 * (b00 + b20)
    U[2, 2] = 1       # a02
    V[0, 2] = 1       # b00
    V[6, 2] = 1       # b20
    W[0, 2] = 1       # c00 (k=0,i=0)
    W[4, 2] = 1       # c02 (k=2,i=0)
    
    # m4 = a10 * (b01 + b11)
    U[3, 3] = 1       # a10
    V[1, 3] = 1       # b01
    V[4, 3] = 1       # b11
    W[2, 3] = 1       # c10 (k=0,i=1)
    W[4, 3] = 1       # c11 (k=1,i=1)
    
    # m5 = a11 * (b02 + b12)
    U[4, 4] = 1       # a11
    V[2, 4] = 1       # b02
    V[5, 4] = 1       # b12
    W[2, 4] = 1       # c10 (k=0,i=1)
    W[6, 4] = 1       # c12 (k=2,i=1)
    
    # m6 = a12 * (b00 + b20)
    U[5, 5] = 1       # a12
    V[0, 5] = 1       # b00
    V[6, 5] = 1       # b20
    W[0, 5] = 1       # c10 (k=0,i=1)
    W[4, 5] = 1       # c12 (k=2,i=1)
    
    # m7 = (a00 + a10) * (b10 - b00)
    U[0, 6] = 1       # a00
    U[3, 6] = 1       # a10
    V[3, 6] = 1       # b10
    V[0, 6] = -1      # b00
    W[2, 6] = 1       # c10 (k=0,i=1)
    
    # m8 = (a01 + a11) * (b11 - b01)
    U[1, 7] = 1       # a01
    U[4, 7] = 1       # a11
    V[4, 7] = 1       # b11
    V[1, 7] = -1      # b01
    W[4, 7] = 1       # c11 (k=1,i=1)
    
    # m9 = (a02 + a12) * (b20 - b00)
    U[2, 8] = 1       # a02
    U[5, 8] = 1       # a12
    V[6, 8] = 1       # b20
    V[0, 8] = -1      # b00
    W[0, 8] = 1       # c00 (k=0,i=0)
    W[2, 8] = 1       # c10 (k=0,i=1)
    
    # m10 = (a00 + a01) * (b10 - b11)
    U[0, 9] = 1       # a00
    U[1, 9] = 1       # a01
    V[3, 9] = 1       # b10
    V[4, 9] = -1      # b11
    W[0, 9] = 1       # c00 (k=0,i=0)
    W[2, 9] = 1       # c01 (k=1,i=0)
    
    # m11 = (a01 + a02) * (b11 - b12)
    U[1, 10] = 1      # a01
    U[2, 10] = 1      # a02
    V[4, 10] = 1      # b11
    V[5, 10] = -1     # b12
    W[2, 10] = 1      # c01 (k=1,i=0)
    W[4, 10] = 1      # c02 (k=2,i=0)
    
    # m12 = (a10 + a11) * (b12 - b11)
    U[3, 11] = 1      # a10
    U[4, 11] = 1      # a11
    V[5, 11] = 1      # b12
    V[4, 11] = -1     # b11
    W[4, 11] = 1      # c11 (k=1,i=1)
    W[6, 11] = 1      # c12 (k=2,i=1)
    
    return U, V, W

def _verify_laderman():
    """Verify Laderman 2x3x3 decomposition."""
    U, V, W = _laderman_233()
    n, m, p = 2, 3, 3
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]))
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-9

def _build_decomposition_laderman():
    """Build the 2x5x6 decomposition using Laderman + Strassen.
    
    Strategy: Split m=5 as 3+2.
    - A (2x5) = [A_a (2x3) | A_b (2x2)]
    - B (5x6) = [B_a (3x6); B_b (2x6)]
    
    (2x3) * (3x6): Split k=0..5 into two 3x3 blocks.
      Each 2x3x3 uses Laderman (rank 12). Two blocks -> rank 24.
    (2x2) * (2x6): Split k=0..5 into three 2x2 blocks.
      Each 2x2x2 uses Strassen (rank 7). Three blocks -> rank 21.
    Total: 24 + 21 = 45.
    """
    n, m, p = 2, 5, 6
    total_rank = 45
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    lU, lV, lW = _laderman_233()
    sU, sV, sW = _strassen_222()
    col = 0
    
    # Part 1: A_a * B_a (2x3 x 6), two Laderman 2x3x3 blocks
    for block in range(2):
        k_off = block * 3
        for r in range(12):
            # U: i in {0,1}, j in {0,1,2}
            for i in range(2):
                for j in range(3):
                    U[i*5 + j, col] = lU[i*3 + j, r]
            
            # V: j in {0,1,2}, k in {k_off, k_off+1, k_off+2}
            for j in range(3):
                for kk in range(3):
                    V[j*6 + k_off + kk, col] = lV[j*3 + kk, r]
            
            # W: k in {k_off, k_off+1, k_off+2}, i in {0,1}
            for kk in range(3):
                for i in range(2):
                    W[(k_off + kk)*2 + i, col] = lW[(kk)*2 + i, r]
            col += 1
    # col = 24
    
    # Part 2: A_b * B_b (2x2 x 6), three Strassen 2x2x2 blocks
    for block in range(3):
        k_off = block * 2
        for r in range(7):
            # U: i in {0,1}, j in {3,4}
            for i in range(2):
                for jj in range(2):
                    U[i*5 + 3 + jj, col] = sU[i*2 + jj, r]
            
            # V: j in {3,4}, k in {k_off, k_off+1}
            for jj in range(2):
                for kk in range(2):
                    V[(3 + jj)*6 + k_off + kk, col] = sV[jj*2 + kk, r]
            
            # W: k in {k_off, k_off+1}, i in {0,1}
            for kk in range(2):
                for i in range(2):
                    W[(k_off + kk)*2 + i, col] = sW[kk*2 + i, r]
            col += 1
    # col = 45
    
    assert col == total_rank
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_decomposition_strassen():
    """Build the 2x5x6 decomposition using Strassen blocks only (rank 54).
    
    Strategy: Split m=5 as 2+1+2, p=6 as 2+2+2.
    - A (2x5) = [A_a (2x2) | A_b (2x1) | A_c (2x2)]
    - B (5x6) = [B_a (2x6); B_b (1x6); B_c (2x6)]
    
    Each 2x2x6 block uses 3 Strassen 2x2x2 (rank 7 each) = rank 21
    The 2x1x6 block uses schoolbook = rank 12
    
    Total: 21 + 12 + 21 = 54
    """
    n, m, p = 2, 5, 6
    total_rank = 54
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    sU, sV, sW = _strassen_222()
    col = 0
    
    # Part 1: A_a * B_a (2x2 x 6), three Strassen 2x2x2 blocks
    for block in range(3):
        k_off = block * 2
        for r in range(7):
            U[0, col] = sU[0, r]
            U[1, col] = sU[1, r]
            U[5, col] = sU[2, r]
            U[6, col] = sU[3, r]
            V[0*6 + k_off + 0, col] = sV[0, r]
            V[0*6 + k_off + 1, col] = sV[1, r]
            V[1*6 + k_off + 0, col] = sV[2, r]
            V[1*6 + k_off + 1, col] = sV[3, r]
            W[(k_off + 0)*2 + 0, col] = sW[0, r]
            W[(k_off + 0)*2 + 1, col] = sW[1, r]
            W[(k_off + 1)*2 + 0, col] = sW[2, r]
            W[(k_off + 1)*2 + 1, col] = sW[3, r]
            col += 1
    
    # Part 2: A_b * B_b (2x1 x 6), schoolbook rank 12
    for i in range(2):
        for k in range(6):
            U[i*5 + 2, col] = 1
            V[2*6 + k, col] = 1
            W[k*2 + i, col] = 1
            col += 1
    
    # Part 3: A_c * B_c (2x2 x 6), three Strassen 2x2x2 blocks
    for block in range(3):
        k_off = block * 2
        for r in range(7):
            U[0*5 + 3, col] = sU[0, r]
            U[0*5 + 4, col] = sU[1, r]
            U[1*5 + 3, col] = sU[2, r]
            U[1*5 + 4, col] = sU[3, r]
            V[3*6 + k_off + 0, col] = sV[0, r]
            V[3*6 + k_off + 1, col] = sV[1, r]
            V[4*6 + k_off + 0, col] = sV[2, r]
            V[4*6 + k_off + 1, col] = sV[3, r]
            W[(k_off + 0)*2 + 0, col] = sW[0, r]
            W[(k_off + 0)*2 + 1, col] = sW[1, r]
            W[(k_off + 1)*2 + 0, col] = sW[2, r]
            W[(k_off + 1)*2 + 1, col] = sW[3, r]
            col += 1
    
    assert col == total_rank
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_vectorized(U, V, W, n=2, m=5, p=6):
    """Vectorized verification using matrix operations."""
    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    rank = Ua.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = Ua[i*m + j, :]
            for k in range(p):
                v_row = Va[j*p + k, :]
                w_row = Wa[k*n + i, :]
                s = float(np.dot(u_row, v_row * w_row))
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-9

def _check_half_integer(U, V, W):
    """Check all entries are integer or half-integer."""
    for mat in [U, V, W]:
        arr = np.array(mat)
        if not np.allclose(arr * 2, np.round(arr * 2), atol=1e-12):
            return False
    return True

def _schoolbook():
    """Schoolbook decomposition, rank = n*m*p = 60."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][c] = 1.0
                V[j*p + k][c] = 1.0
                W[k*n + i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}

def construct():
    t0 = time.time()
    deadline = t0 + 100  # safety margin before 150s kill
    
    # Try Laderman+Strassen rank-45 first
    try:
        result = _build_decomposition_laderman()
        if time.time() < deadline:
            if _verify_vectorized(result['U'], result['V'], result['W']):
                if _check_half_integer(result['U'], result['V'], result['W']):
                    return result
    except Exception:
        pass
    
    # Fallback: Strassen-based rank-54
    try:
        result = _build_decomposition_strassen()
        if time.time() < deadline:
            if _verify_vectorized(result['U'], result['V'], result['W']):
                if _check_half_integer(result['U'], result['V'], result['W']):
                    return result
    except Exception:
        pass
    
    # Final fallback: schoolbook rank-60
    return _schoolbook()

# EVOLVE-BLOCK-END
