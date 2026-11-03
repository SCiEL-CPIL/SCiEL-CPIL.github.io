"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(result, 1.0, atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    # C00 += p1, C11 += p1
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    
    # p2 = (A10+A11)*B00
    # C10 += p2, C11 -= p2
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    
    # p3 = A00*(B01-B11)
    # C01 += p3, C11 += p3
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    
    # p4 = A11*(B10-B00)
    # C00 += p4, C10 += p4
    factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    
    # p5 = (A00+A01)*B11
    # C00 -= p5, C01 += p5
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    
    # p6 = (A10-A00)*(B00+B01)
    # C11 += p6
    factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    
    # p7 = (A01-A11)*(B10+B11)
    # C00 += p7
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    
    return factors

def _build_4x4x4_strassen():
    """
    Build rank-49 decomposition for 4x4 * 4x4.
    Uses Kronecker product of two 2x2x2 Strassen decompositions.
    Returns U (16,49), V (16,49), W (16,49).
    Index convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    s2 = _strassen_2x2_factors()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for s in range(7):
        u_block = s2[s][0]  # 4 coeffs for A blocks
        v_block = s2[s][1]  # 4 coeffs for B blocks
        w_block = s2[s][2]  # 4 coeffs for C blocks
        
        for r in range(7):
            col = s * 7 + r
            u_scalar = s2[r][0]  # 4 coeffs for elements within A block
            v_scalar = s2[r][1]  # 4 coeffs for elements within B block
            w_scalar = s2[r][2]  # 4 coeffs for elements within C block
            
            # U: A[i,j] where i = 2*bi+ei, j = 2*bj+ej
            for bi in range(2):
                for bj in range(2):
                    c_b = u_block[bi * 2 + bj]
                    if c_b == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            c_s = u_scalar[ei * 2 + ej]
                            if c_s == 0:
                                continue
                            i = 2 * bi + ei
                            j = 2 * bj + ej
                            U[i * 4 + j][col] += c_b * c_s
            
            # V: B[j,k] where j = 2*bi+ei, k = 2*bj+ej
            for bi in range(2):
                for bj in range(2):
                    c_b = v_block[bi * 2 + bj]
                    if c_b == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            c_s = v_scalar[ei * 2 + ej]
                            if c_s == 0:
                                continue
                            j = 2 * bi + ei
                            k = 2 * bj + ej
                            V[j * 4 + k][col] += c_b * c_s
            
            # W: C[k,i] where k = 2*bi+ei, i = 2*bj+ej
            for bi in range(2):
                for bj in range(2):
                    c_b = w_block[bi * 2 + bj]
                    if c_b == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            c_s = w_scalar[ei * 2 + ej]
                            if c_s == 0:
                                continue
                            k = 2 * bi + ei
                            i = 2 * bj + ej
                            W[k * 4 + i][col] += c_b * c_s
    
    return U, V, W

def _build_4x4x8_strassen():
    """
    Build rank-98 decomposition for 4x4 * 4x8.
    Splits B into two 4x4 blocks (columns 0-3 and 4-7).
    Each 4x4*4x4 uses rank-49 Strassen.
    Total rank = 98.
    
    U: (16, 98), V: (32, 98), W: (32, 98)
    Index convention: U[i*4+j, r], V[j*8+k, r], W[k*4+i, r]
    """
    n, m, p = 4, 4, 8
    
    U4, V4, W4 = _build_4x4x4_strassen()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]   # (16, 98)
    V = [[0] * rank for _ in range(m * p)]   # (32, 98)
    W = [[0] * rank for _ in range(p * n)]   # (32, 98)
    
    for t in range(2):  # t=0: B columns 0-3, t=1: B columns 4-7
        base_col = t * 49
        base_k = t * 4  # offset for B column index
        
        for r in range(49):
            col = base_col + r
            
            # U is the same for both blocks (A doesn't change)
            for i in range(4):
                for j in range(4):
                    val = U4[i * 4 + j][r]
                    if val != 0:
                        U[i * m + j][col] += val
            
            # V: B[j, k] where k = base_k + k_local
            for j in range(4):
                for k_local in range(4):
                    val = V4[j * 4 + k_local][r]
                    if val != 0:
                        k = base_k + k_local
                        V[j * p + k][col] += val
            
            # W: C[k, i] where k = base_k + k_local
            for k_local in range(4):
                for i in range(4):
                    val = W4[k_local * 4 + i][r]
                    if val != 0:
                        k = base_k + k_local
                        W[k * n + i][col] += val
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    # Try Strassen-based decomposition: rank 98
    try:
        strassen_result = _build_4x4x8_strassen()
        strassen_rank = len(strassen_result["U"][0])
        
        if strassen_rank < best_rank:
            if _verify(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
                best = strassen_result
                best_rank = strassen_rank
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best
    
    return best

# EVOLVE-BLOCK-END
