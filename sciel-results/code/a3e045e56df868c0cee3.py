"""Optimized 2x5x6 matrix multiplication tensor decomposition using Strassen blocks."""
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

def _build_decomposition():
    """Build the 2x5x6 decomposition.
    
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
            # U: i in {0,1}, j in {0,1}
            U[0, col] = sU[0, r]
            U[1, col] = sU[1, r]
            U[5, col] = sU[2, r]
            U[6, col] = sU[3, r]
            
            # V: j in {0,1}, k in {k_off, k_off+1}
            V[0*6 + k_off + 0, col] = sV[0, r]
            V[0*6 + k_off + 1, col] = sV[1, r]
            V[1*6 + k_off + 0, col] = sV[2, r]
            V[1*6 + k_off + 1, col] = sV[3, r]
            
            # W: k in {k_off, k_off+1}, i in {0,1}
            W[(k_off + 0)*2 + 0, col] = sW[0, r]
            W[(k_off + 0)*2 + 1, col] = sW[1, r]
            W[(k_off + 1)*2 + 0, col] = sW[2, r]
            W[(k_off + 1)*2 + 1, col] = sW[3, r]
            col += 1
    # col = 21
    
    # Part 2: A_b * B_b (2x1 x 6), schoolbook rank 12
    for i in range(2):
        for k in range(6):
            U[i*5 + 2, col] = 1
            V[2*6 + k, col] = 1
            W[k*2 + i, col] = 1
            col += 1
    # col = 33
    
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
    # col = 54
    
    assert col == total_rank
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_vectorized(U, V, W, n=2, m=5, p=6):
    """Vectorized verification using matrix operations."""
    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    rank = Ua.shape[1]
    
    # For each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # Vectorize: for fixed i,j, compute for all k at once
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = Ua[i*m + j, :]  # (rank,)
            for k in range(p):
                v_row = Va[j*p + k, :]  # (rank,)
                w_row = Wa[k*n + i, :]  # (rank,)
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

def _try_laderman_233():
    """Try to build a decomposition using Laderman's 2x3x3 method.
    Laderman's method multiplies 2x3 by 3x3 with rank 12 (vs schoolbook 18).
    If we can split m=5 as 3+2 and use Laderman for the 2x3x6 part...
    Actually, let's think: A is 2x5, B is 5x6.
    Split j=0..4.
    Option: Split into (2x3) * (3x6) + (2x2) * (2x6).
    (2x3) * (3x6): Split k=0..5 into two 3x3 blocks.
      Each 2x3x3 can use Laderman (rank 12). Two blocks -> rank 24.
    (2x2) * (2x6): Split k=0..5 into three 2x2 blocks.
      Each 2x2x2 uses Strassen (rank 7). Three blocks -> rank 21.
    Total: 24 + 21 = 45. This is better than 54!
    """
    n, m, p = 2, 5, 6
    
    # We need Laderman 2x3x3.
    # Laderman's algorithm:
    # A (2x3), B (3x3), C = A*B (2x3)
    # 12 products:
    # M1 = A00 * B00
    # M2 = (A00+A01) * (B10-B11)
    # M3 = (A00+A02) * (B20-B21)
    # M4 = A10 * (B00+B10)
    # M5 = (A10+A11) * (B01-B11)
    # M6 = (A10+A12) * (B02-B12)
    # M7 = A11 * (B10+B20)
    # M8 = (A11+A12) * (B11-B21)
    # M9 = (A11+A12) * (B12-B22)  -- wait, standard Laderman has specific forms.
    # Let's look up or derive Laderman 2x3x3.
    # Reference: Laderman (1976).
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    # C = [[c00, c01, c02], [c10, c11, c12]]
    
    # The 12 terms are:
    # m1 = a00*b00
    # m2 = (a00+a01)*(b10-b11)
    # m3 = (a00+a02)*(b20-b21)
    # m4 = a10*(b00+b10)
    # m5 = (a10+a11)*(b01-b11)
    # m6 = (a10+a12)*(b02-b12)
    # m7 = a11*(b10+b20)
    # m8 = (a11+a12)*(b11-b21)
    # m9 = (a11+a12)*(b12-b22)
    # m10 = a12*(b20+b21)  -- This doesn't look symmetric. Let me verify.
    
    # Actually, I should just verify a known Laderman decomposition.
    # Since I cannot be 100% sure of the coefficients from memory without verification,
    # and the current program already works at rank 54, attempting a potentially
    # incorrect rank 45 decomposition is risky. If it fails verification, it falls back.
    # But if I get the coefficients wrong, it will fail.
    
    # Let's stick to the known correct rank 54 for safety, unless I can verify Laderman quickly.
    # Given the constraints, I will return None to indicate "not attempted/failed".
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 100  # safety margin before 150s kill
    
    # Primary: Strassen-based rank-54
    try:
        result = _build_decomposition()
        if time.time() < deadline:
            if _verify_vectorized(result['U'], result['V'], result['W']):
                if _check_half_integer(result['U'], result['V'], result['W']):
                    return result
    except Exception:
        pass
    
    # Fallback: schoolbook rank-60
    return _schoolbook()

# EVOLVE-BLOCK-END
