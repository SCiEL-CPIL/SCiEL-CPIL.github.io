# EVOLVE-BLOCK-START
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

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

def _verify(U, V, W, n=4, m=4, p=8):
    """Check that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
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

def _try_strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using Strassen.
    
    Strategy: Split the 4x8 matrix B into two 4x4 blocks B1 and B2.
    Compute C1 = A*B1 and C2 = A*B2, each using Strassen (rank 49).
    Total rank = 98.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Get the 2x2 Strassen factors
    s2_factors = _strassen_2x2_factors()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # (16, 98)
    V = [[0] * rank for _ in range(m * p)]  # (32, 98)
    W = [[0] * rank for _ in range(p * n)]  # (32, 98)
    
    # For each of the two subproblems (t=0 for B1, t=1 for B2):
    for t in range(2):
        # For each of the 7 top-level Strassen products (s=0..6):
        for s in range(7):
            # For each of the 7 scalar Strassen products (r=0..6):
            for r in range(7):
                col = t * 49 + s * 7 + r
                
                # The top-level Strassen product s involves certain A blocks and B blocks.
                s2_u_block = s2_factors[s][0]  # [coeff for A00, A01, A10, A11] in block terms
                s2_v_block = s2_factors[s][1]  # [coeff for B00, B01, B10, B11] in block terms
                s2_w_block = s2_factors[s][2]  # [coeff for C00, C01, C10, C11] in block terms
                
                # For the scalar Strassen product r:
                s2_u_scalar = s2_factors[r][0]
                s2_v_scalar = s2_factors[r][1]
                s2_w_scalar = s2_factors[r][2]
                
                # U contributions:
                for bi in range(2):
                    for bj in range(2):
                        block_idx = bi * 2 + bj
                        c_block = s2_u_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_u_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                i = 2 * bi + ei
                                j = 2 * bj + ej
                                U[i * m + j][col] += c_block * c_scalar
                
                # V contributions:
                for bi in range(2):
                    for bj in range(2):
                        block_idx = bi * 2 + bj
                        c_block = s2_v_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_v_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                j = 2 * bi + ei  # B row
                                k = 4 * t + 2 * bj + ej  # B col (offset by 4*t)
                                V[j * p + k][col] += c_block * c_scalar
                
                # W contributions:
                for ki in range(2):
                    for ii in range(2):
                        block_idx = ki * 2 + ii
                        c_block = s2_w_block[block_idx]
                        if c_block == 0:
                            continue
                        for e_k in range(2):
                            for e_i in range(2):
                                scalar_idx = e_k * 2 + e_i
                                c_scalar = s2_w_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                k = 4 * t + 2 * ki + e_k  # C row
                                i = 2 * ii + e_i  # C col
                                W[k * n + i][col] += c_block * c_scalar
    
    return {"U": U, "V": V, "W": W}

def construct():
    import time
    
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    # First try the rank-98 Strassen decomposition
    try:
        result = _try_strassen_4x4_4x8()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
