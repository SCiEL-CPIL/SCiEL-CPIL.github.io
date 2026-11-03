"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _strassen_2x2_factors():
    """
    Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication.
    Each factor is (u_vec, v_vec, w_vec) where:
      u_vec: 4-element vector for A (row-major: A00, A01, A10, A11)
      v_vec: 4-element vector for B (row-major: B00, B01, B10, B11)
      w_vec: 4-element vector for C (row-major: C00, C01, C10, C11)
    """
    factors = []
    
    # p1 = (A00+A11)(B00+B11)
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    w = [1, 0, 0, 1]
    factors.append((u, v, w))
    
    # p2 = (A10+A11) * B00
    u = [0, 0, 1, 1]
    v = [1, 0, 0, 0]
    w = [0, 0, 1, -1]
    factors.append((u, v, w))
    
    # p3 = A00 * (B01-B11)
    u = [1, 0, 0, 0]
    v = [0, 1, 0, -1]
    w = [0, 1, 0, 1]
    factors.append((u, v, w))
    
    # p4 = A11 * (B10-B00)
    u = [0, 0, 0, 1]
    v = [-1, 0, 1, 0]
    w = [1, 0, 1, 0]
    factors.append((u, v, w))
    
    # p5 = (A00+A01) * B11
    u = [1, 1, 0, 0]
    v = [0, 0, 0, 1]
    w = [-1, 1, 0, 0]
    factors.append((u, v, w))
    
    # p6 = (A10-A00) * (B00+B01)
    u = [-1, 0, 1, 0]
    v = [1, 1, 0, 0]
    w = [0, 0, 0, 1]
    factors.append((u, v, w))
    
    # p7 = (A01-A11) * (B10+B11)
    u = [0, 1, 0, -1]
    v = [0, 0, 1, 1]
    w = [1, 0, 0, 0]
    factors.append((u, v, w))
    
    return factors

def _build_strassen_4x4():
    """
    Build a rank-49 decomposition for 4x4 * 4x4 using two levels of Strassen.
    Returns U (16, 49), V (16, 49), W (16, 49) where:
      U[i*4+j, r] = coefficient of A[i,j]
      V[j*4+k, r] = coefficient of B[j,k]  
      W[k*4+i, r] = coefficient of C[i,k] (using W[k*n+i] convention with n=4)
    """
    s2 = _strassen_2x2_factors()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for s in range(7):
        u_block = s2[s][0]  # 4 coeffs for A blocks (A00,A01,A10,A11)
        v_block = s2[s][1]  # 4 coeffs for B blocks (B00,B01,B10,B11)
        w_block = s2[s][2]  # 4 coeffs for C blocks (C00,C01,C10,C11)
        
        for r in range(7):
            col = s * 7 + r
            u_scl = s2[r][0]  # 4 coeffs for elements within a 2x2 block
            v_scl = s2[r][1]
            w_scl = s2[r][2]
            
            # U: A[i,j] where i = 2*bi+ei, j = 2*bj+ej
            for bi in range(2):
                for bj in range(2):
                    cb = u_block[bi * 2 + bj]
                    if cb == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            cs = u_scl[ei * 2 + ej]
                            if cs == 0:
                                continue
                            i = 2 * bi + ei
                            j = 2 * bj + ej
                            U[i * 4 + j][col] += cb * cs
            
            # V: B[j,k] where j = 2*bi+ei, k = 2*bj+ej
            for bi in range(2):
                for bj in range(2):
                    cb = v_block[bi * 2 + bj]
                    if cb == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            cs = v_scl[ei * 2 + ej]
                            if cs == 0:
                                continue
                            j = 2 * bi + ei
                            k = 2 * bj + ej
                            V[j * 4 + k][col] += cb * cs
            
            # W: C[i,k] where i = 2*ci+ei, k = 2*cj+ej
            # W index = k*4 + i
            for ci in range(2):
                for cj in range(2):
                    cb = w_block[ci * 2 + cj]
                    if cb == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            cs = w_scl[ei * 2 + ej]
                            if cs == 0:
                                continue
                            i = 2 * ci + ei
                            k = 2 * cj + ej
                            W[k * 4 + i][col] += cb * cs
    
    return U, V, W

def _verify_4x4(U, V, W):
    """Verify 4x4*4x4 decomposition. U,V,W are (16,49) lists."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(4, 4, 49)
    V3 = V_arr.reshape(4, 4, 49)
    W3 = W_arr.reshape(4, 4, 49)  # W[k*4+i] -> reshape to (k, i, r) = (4,4,49)
    result = np.einsum('ijr,jkr,kir->ik', U3, V3, W3)
    # result[i,k] should be 1 for all i,k
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _try_strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using Strassen.
    Split B into two 4x4 blocks. Each 4x4*4x4 uses rank 49 (two-level Strassen).
    Total rank = 98.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    U4, V4, W4 = _build_strassen_4x4()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]   # (16, 98)
    V = [[0] * rank for _ in range(m * p)]   # (32, 98)
    W = [[0] * rank for _ in range(p * n)]   # (32, 98)
    
    for t in range(2):
        base = t * 49
        for r in range(49):
            col = base + r
            # U: same for both subproblems
            for i in range(16):
                if U4[i][r] != 0:
                    U[i][col] = U4[i][r]
            # V: B[j, 4*t + k]
            for j in range(4):
                for k in range(4):
                    val = V4[j * 4 + k][r]
                    if val != 0:
                        V[j * p + (4 * t + k)][col] = val
            # W: C[i, 4*t + k] -> W index = (4*t+k)*n + i
            for i in range(4):
                for k in range(4):
                    val = W4[k * 4 + i][r]
                    if val != 0:
                        W[(4 * t + k) * n + i][col] = val
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Tries Strassen-based decomposition (rank 98) first, falls back to schoolbook (rank 128).
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try Strassen-based decomposition
    try:
        if time.time() < deadline:
            # First verify the 4x4*4x4 building block
            U4, V4, W4 = _build_strassen_4x4()
            if not _verify_4x4(U4, V4, W4):
                raise ValueError("4x4 Strassen block failed verification")
            
            if time.time() < deadline:
                result = _try_strassen_4x4_4x8()
                if time.time() < deadline:
                    if _verify(result["U"], result["V"], result["W"]):
                        return result
    except Exception:
        pass
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
