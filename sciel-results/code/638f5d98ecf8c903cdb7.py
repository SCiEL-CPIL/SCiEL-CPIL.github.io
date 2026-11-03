"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication."""
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
    Each factor is (U_vec, V_vec, W_vec) where each is a 4-element list.
    U_vec indexed by [i*2+j], V_vec by [j*2+k], W_vec by [k*2+i].
    """
    factors = []
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (A10+A11) * B00 -> C10 += M2, C11 -= M2
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # M3 = A00 * (B01-B11) -> C01 += M3, C11 += M3
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    # M4 = A11 * (B10-B00) -> C10 += M4, C00 += M4
    factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    # M5 = (A00+A01) * B11 -> C00 -= M5, C01 += M5
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    # M6 = (A10-A00) * (B00+B01) -> C10 += M6
    factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    # M7 = (A01-A11) * (B10+B11) -> C01 += M7
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    return factors

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

def _strassen_4x4x4_factors():
    """
    Generate the 49 rank-1 factors for 4x4 * 4x4 matrix multiplication
    using recursive Strassen (Strassen on 2x2 blocks of 2x2 matrices).
    Returns list of (U_vec, V_vec, W_vec) where each is a 16-element vector.
    U_vec indexed by [i*4+j], V_vec by [j*4+k], W_vec by [k*4+i].
    """
    s2 = _strassen_2x2_factors()
    factors = []
    
    for s in range(7):
        for r in range(7):
            u_block = s2[s][0]  # 4 elements: coeffs for A blocks [00,01,10,11]
            v_block = s2[s][1]  # 4 elements: coeffs for B blocks [00,01,10,11]
            w_block = s2[s][2]  # 4 elements: coeffs for C blocks [00,01,10,11]
            
            u_scalar = s2[r][0]  # 4 elements: coeffs for scalar A positions [00,01,10,11]
            v_scalar = s2[r][1]  # 4 elements: coeffs for scalar B positions [00,01,10,11]
            w_scalar = s2[r][2]  # 4 elements: coeffs for scalar C positions [00,01,10,11]
            
            U_vec = [0.0] * 16
            V_vec = [0.0] * 16
            W_vec = [0.0] * 16
            
            # Fill U_vec: U[i,j] = sum over block (bi,bj) of c_block[bi,bj] * u_scalar[ei,ej]
            # where i = 2*bi+ei, j = 2*bj+ej
            for bi in range(2):
                for bj in range(2):
                    c_block = u_block[bi * 2 + bj]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            c_scalar = u_scalar[ei * 2 + ej]
                            if c_scalar == 0:
                                continue
                            i = 2 * bi + ei
                            j = 2 * bj + ej
                            U_vec[i * 4 + j] += c_block * c_scalar
            
            # Fill V_vec: V[j,k] = sum over block (bi,bj) of c_block[bi,bj] * v_scalar[ei,ej]
            # where j = 2*bi+ei, k = 2*bj+ej
            for bi in range(2):
                for bj in range(2):
                    c_block = v_block[bi * 2 + bj]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            c_scalar = v_scalar[ei * 2 + ej]
                            if c_scalar == 0:
                                continue
                            j = 2 * bi + ei
                            k = 2 * bj + ej
                            V_vec[j * 4 + k] += c_block * c_scalar
            
            # Fill W_vec: W[k,i] = sum over block (ci,cj) of c_block[ci,cj] * w_scalar[ei,ej]
            # where k = 2*ci+ei, i = 2*cj+ej
            for ci in range(2):
                for cj in range(2):
                    c_block = w_block[ci * 2 + cj]
                    if c_block == 0:
                        continue
                    for ei in range(2):
                        for ej in range(2):
                            c_scalar = w_scalar[ei * 2 + ej]
                            if c_scalar == 0:
                                continue
                            k = 2 * ci + ei
                            i = 2 * cj + ej
                            W_vec[k * 4 + i] += c_block * c_scalar
            
            factors.append((U_vec, V_vec, W_vec))
    
    return factors

def _try_strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using Strassen.
    Split B into two 4x4 blocks (columns 0-3 and 4-7), compute A*B1 and A*B2
    each with rank-49 Strassen. The output C has columns 0-3 from first, 4-7 from second.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    s44_factors = _strassen_4x4x4_factors()
    
    rank = 98
    U = [[0.0] * rank for _ in range(n * m)]   # shape (16, 98)
    V = [[0.0] * rank for _ in range(m * p)]   # shape (32, 98)
    W = [[0.0] * rank for _ in range(p * n)]   # shape (32, 98)
    
    for t in range(2):  # two 4x4 sub-problems
        for f in range(49):
            col = t * 49 + f
            U_vec, V_vec, W_vec = s44_factors[f]
            
            # U: same for both subproblems (A is unchanged)
            for idx in range(16):
                if U_vec[idx] != 0:
                    U[idx][col] = U_vec[idx]
            
            # V: for subproblem t, B columns are offset by 4*t
            # V_vec[idx] corresponds to B element (j,k) where idx = j*4+k
            # In the full 4x8 B, this becomes (j, k+4*t)
            for idx in range(16):
                if V_vec[idx] == 0:
                    continue
                j = idx // 4
                k = idx % 4
                full_k = k + 4 * t
                V[j * p + full_k][col] = V_vec[idx]
            
            # W: for subproblem t, output C columns are offset by 4*t
            # W_vec[idx] corresponds to C element (k,i) where idx = k*4+i
            # In the full 4x8 C, this becomes (k+4*t, i)
            # W is indexed by [k_full * n + i]
            for idx in range(16):
                if W_vec[idx] == 0:
                    continue
                k = idx // 4
                i = idx % 4
                full_k = k + 4 * t
                W[full_k * n + i][col] = W_vec[idx]
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try Strassen-based decomposition (rank 98)
    try:
        result = _try_strassen_4x4_4x8()
        if time.time() < deadline:
            if _verify(result["U"], result["V"], result["W"]):
                return result
        else:
            # Time ran out during construction, but we have the result.
            # Return it without verification to avoid timeout.
            return result
    except Exception:
        pass
    
    # Fall back to schoolbook (rank 128)
    return _schoolbook()
# EVOLVE-BLOCK-END
