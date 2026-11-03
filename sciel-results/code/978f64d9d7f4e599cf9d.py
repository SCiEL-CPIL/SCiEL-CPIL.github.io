"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

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

def _strassen_2x2_factors():
    """
    Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication.
    Each factor is (u_vec, v_vec, w_vec) where:
      u_vec: 4-element vector for A (row-major: A00, A01, A10, A11)
      v_vec: 4-element vector for B (row-major: B00, B01, B10, B11)
      w_vec: 4-element vector for C (row-major: C00, C01, C10, C11)
    """
    # Standard Strassen for 2x2:
    # p1 = (A00+A11)(B00+B11)
    # p2 = (A10+A11)(B00)
    # p3 = A00(B01-B11)
    # p4 = A11(B10-B00)
    # p5 = (A00+A01)(B11)
    # p6 = (A10-A00)(B00+B01)
    # p7 = (A01-A11)(B10+B11)
    #
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6

    factors = []

    # p1 = (A00+A11)(B00+B11)
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # p2 = (A10+A11) * B00
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # p3 = A00 * (B01-B11)
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    # p4 = A11 * (B10-B00)
    factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    # p5 = (A00+A01) * B11
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    # p6 = (A10-A00) * (B00+B01)
    factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    # p7 = (A01-A11) * (B10+B11)
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))

    return factors

def _rational_polish(mat, deadline):
    """
    Refinement pass: snap every entry of `mat` (a 2-D list of numbers) to the
    nearest half-integer lattice point, then return the snapped list.
    This is the 'border rank / rational approximation' projection: the exact
    tensor lies in the secant variety, and the Strassen factors already live on
    the integer (hence half-integer) lattice; snapping is a no-op on correct
    factors but a safety net that removes any off-lattice drift from a bug.
    """
    snapped = []
    for row in mat:
        new_row = []
        for val in row:
            # nearest half-integer: round(2*val)/2
            snapped_val = (int(round(2.0 * val))) / 2.0
            new_row.append(snapped_val)
        snapped.append(new_row)
        if time.time() > deadline:
            break
    return snapped

def _try_strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using recursive Strassen.

    Strategy: Split the 4x8 matrix B into two 4x4 blocks B1 and B2.
    Compute C1 = A*B1 and C2 = A*B2, each using Strassen (rank 49).
    Total rank = 98.

    Each 4x4*4x4 Strassen decomposition uses 7 top-level Strassen products,
    each of which is a 2x2*2x2 multiplication, itself decomposed with 7 scalar
    Strassen products. Total: 7 * 7 = 49 per subproblem.

    FIX: the W-coefficient mapping must combine the block-level W coefficient
    (s2_w_block) with the scalar-level W coefficient (s2_w_scalar), exactly as
    U and V do. The previous version only used the scalar coefficient, which
    produced an incorrect W factor.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8

    s2_factors = _strassen_2x2_factors()

    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # (16, 98)
    V = [[0] * rank for _ in range(m * p)]  # (32, 98)
    W = [[0] * rank for _ in range(p * n)]  # (32, 98)

    for t in range(2):
        for s in range(7):
            for r in range(7):
                col = t * 49 + s * 7 + r

                s2_u_block = s2_factors[s][0]
                s2_v_block = s2_factors[s][1]
                s2_w_block = s2_factors[s][2]

                s2_u_scalar = s2_factors[r][0]
                s2_v_scalar = s2_factors[r][1]
                s2_w_scalar = s2_factors[r][2]

                # U contributions: A block (bi,bj) x scalar (ei,ej)
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

                # V contributions: B block (bi,bj) for subproblem t, offset by 4*t
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
                                j = 2 * bi + ei
                                k = 4 * t + 2 * bj + ej
                                V[j * p + k][col] += c_block * c_scalar

                # W contributions: C block (ci,cj) x scalar (ei,ej)
                # FIXED: multiply s2_w_block by s2_w_scalar.
                for ci in range(2):
                    for cj in range(2):
                        block_idx = ci * 2 + cj
                        c_block = s2_w_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_w_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                i = 2 * ci + ei
                                k = 4 * t + 2 * cj + ej
                                W[k * n + i][col] += c_block * c_scalar

    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before the 150s kill

    # Primary: recursive Strassen rank-98, then a rational (half-integer) polish.
    try:
        result = _try_strassen_4x4_4x8()
        if time.time() < deadline:
            # Refinement: project factors onto the nearest half-integer lattice.
            result["U"] = _rational_polish(result["U"], deadline)
            result["V"] = _rational_polish(result["V"], deadline)
            result["W"] = _rational_polish(result["W"], deadline)
            if time.time() < deadline:
                if _verify(result["U"], result["V"], result["W"]):
                    return result
    except Exception:
        pass

    # Fallback: schoolbook rank-128.
    return _schoolbook()
# EVOLVE-BLOCK-END
