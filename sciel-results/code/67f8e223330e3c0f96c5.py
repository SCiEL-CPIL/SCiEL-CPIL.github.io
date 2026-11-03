"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
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
    Each factor is (u_vec, v_vec, w_vec) where each vec is 4-element,
    row-major: index (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3.
    """
    factors = []
    # p1 = (A00+A11)(B00+B11);  c11 += p1, c22 += p1
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # p2 = (A10+A11)*B00;       c21 += p2, c22 -= p2
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # p3 = A00*(B01-B11);       c12 += p3, c22 += p3
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    # p4 = A11*(B10-B00);       c11 += p4, c21 += p4
    factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    # p5 = (A00+A01)*B11;       c11 -= p5, c12 += p5
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    # p6 = (A10-A00)*(B00+B01); c22 += p6
    factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    # p7 = (A01-A11)*(B10+B11); c11 += p7
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    return factors

def _try_strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using recursive Strassen.

    Split the 4x8 matrix B into two 4x4 blocks B1, B2 (columns 0..3 and 4..7).
    C1 = A*B1 and C2 = A*B2, each a 4x4*4x4 product decomposed with rank 49
    (7 top-level Strassen products, each a 2x2*2x2 Strassen of rank 7).
    Total rank = 98.

    Indexing convention (matches _verify):
      U is (n*m, rank), row index = i*m + j  (i row of A, j col of A / row of B)
      V is (m*p, rank), row index = j*p + k  (j row of B, k col of B)
      W is (p*n, rank), row index = k*n + i  (k row of C, i col of C)
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    s2 = _strassen_2x2_factors()

    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for t in range(2):          # two 4x4 subproblems
        for s in range(7):      # top-level Strassen product
            for r in range(7):  # scalar Strassen product
                col = t * 49 + s * 7 + r

                u_block = s2[s][0]   # top-level A-block coefficients
                v_block = s2[s][1]   # top-level B-block coefficients
                w_block = s2[s][2]   # top-level C-block coefficients   <-- FIXED
                u_scalar = s2[r][0]  # scalar A coefficients
                v_scalar = s2[r][1]  # scalar B coefficients
                w_scalar = s2[r][2]  # scalar C coefficients

                # U: A element (i, j) = (2*bi+ei, 2*bj+ej), block index bi*2+bj
                for bi in range(2):
                    for bj in range(2):
                        cb = u_block[bi * 2 + bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = u_scalar[ei * 2 + ej]
                                if cs == 0:
                                    continue
                                i = 2 * bi + ei
                                j = 2 * bj + ej
                                U[i * m + j][col] += cb * cs

                # V: B element (j, k) = (2*bi+ei, 4*t + 2*bj + ej)
                for bi in range(2):
                    for bj in range(2):
                        cb = v_block[bi * 2 + bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = v_scalar[ei * 2 + ej]
                                if cs == 0:
                                    continue
                                j = 2 * bi + ei
                                k = 4 * t + 2 * bj + ej
                                V[j * p + k][col] += cb * cs

                # W: C element (k, i) = (4*t + 2*cj + ej, 2*ci + ei)
                for ci in range(2):
                    for cj in range(2):
                        cb = w_block[ci * 2 + cj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = w_scalar[ei * 2 + ej]
                                if cs == 0:
                                    continue
                                i = 2 * ci + ei
                                k = 4 * t + 2 * cj + ej
                                W[k * n + i][col] += cb * cs

    return {"U": U, "V": V, "W": W}

def _reduce_rank_linear(U, V, W, max_time=30.0):
    """
    Greedily remove columns that are linear combinations of the remaining
    columns while preserving the matrix-multiplication tensor exactly.

    Column r is removable iff, for every (i,j,k),
        U[i,j,r]*V[j,k,r]*W[k,i,r]  =  sum_{s != r} c_s * U[i,j,s]*V[j,k,s]*W[k,i,s]
    for some coefficients c_s. We build a linear system over the 4*4*8 = 128
    entries (i,j,k) with unknowns c_s (s != r) and solve least-squares; if the
    residual is ~0 the column is dropped.
    """
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    start_time = time.time()

    while True:
        if time.time() - start_time > max_time:
            break
        rank = U_arr.shape[1]
        if rank <= 1:
            break

        # Tensor T[i,j,k,r] = U[i,j,r]*V[j,k,r]*W[k,i,r]
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        T = U3 * V3 * W3  # shape (n, m, p, rank)
        T_flat = T.reshape(-1, rank)  # (128, rank)

        removed = False
        for r in range(rank):
            if time.time() - start_time > max_time:
                break
            target = T_flat[:, r]
            mask = [s for s in range(rank) if s != r]
            A_mat = T_flat[:, mask]
            # Solve A_mat @ c ~= target
            try:
                c, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, target, rcond=None)
            except Exception:
                continue
            residual = A_mat @ c - target
            if np.linalg.norm(residual) < 1e-8:
                # Column r is a linear combination of the others; drop it.
                U_arr = np.delete(U_arr, r, axis=1)
                V_arr = np.delete(V_arr, r, axis=1)
                W_arr = np.delete(W_arr, r, axis=1)
                removed = True
                break

        if not removed:
            break

    return U_arr, V_arr, W_arr

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

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.

    Uses a recursive Strassen construction (two independent 4x4x4 rank-49
    decompositions, total rank 98), then attempts rank reduction by removing
    linearly-dependent columns. Falls back to schoolbook (rank 128).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    best_result = None
    best_rank = float('inf')

    # Try the Strassen-based decomposition
    try:
        result = _try_strassen_4x4_4x8()
        if _verify(result["U"], result["V"], result["W"]):
            best_result = result
            best_rank = len(result["U"][0])

            # Attempt rank reduction by removing linearly-dependent columns
            if time.time() < deadline:
                U_arr = np.array(result["U"], dtype=float)
                V_arr = np.array(result["V"], dtype=float)
                W_arr = np.array(result["W"], dtype=float)
                U_arr, V_arr, W_arr = _reduce_rank_linear(
                    U_arr, V_arr, W_arr, max_time=deadline - time.time())
                new_result = {
                    "U": U_arr.tolist(),
                    "V": V_arr.tolist(),
                    "W": W_arr.tolist(),
                }
                if _verify(new_result["U"], new_result["V"], new_result["W"]):
                    if len(U_arr[0]) < best_rank:
                        best_result = new_result
                        best_rank = len(U_arr[0])
    except Exception:
        pass

    if best_result is not None:
        return best_result

    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
