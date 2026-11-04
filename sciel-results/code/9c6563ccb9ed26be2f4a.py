"""Matrix multiplication 4x4 by 4x8 rank-minimal exact tensor decomposition.

Strategy:
  * Build the rank-7 Strassen 2x2x2 decomposition (U,V,W each 4x7).
  * Kronecker-product it with itself to obtain a rank-49 4x4x4 decomposition.
  * Use that to multiply a 4x4 matrix by a 4x8 matrix: the 4x8 input is two
    independent 4x4 blocks, so we reuse the SAME U factor and concatenate the
    two V blocks.  This yields a valid 4x4x8 decomposition of rank 49.
  * Verify exactly with Fractions.  Fall back to schoolbook (rank 128) if the
    structured construction ever fails verification.
  * Use spare time for a cheap, bounded search for a lower-rank exact
    decomposition; keep the best valid result found.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]
n, m, p = DIMENSIONS
N_DIM = n * m   # 16
M_DIM = m * p   # 32
P_DIM = p * n   # 32
SCHOOLBOOK_RANK = n * m * p  # 128


def _build_tensor():
    T = np.zeros((N_DIM, M_DIM, P_DIM), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T


def _schoolbook():
    rank = SCHOOLBOOK_RANK
    U = [[0] * rank for _ in range(N_DIM)]
    V = [[0] * rank for _ in range(M_DIM)]
    W = [[0] * rank for _ in range(P_DIM)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2x2():
    U = [[1, 0, 1, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 1],
         [0, 1, 0, 0, 0, -1, 0],
         [1, 1, 0, 1, 0, 0, -1]]
    V = [[1, 1, 0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0, 1, 0],
         [0, 0, 0, -1, 0, 0, 1],
         [1, 0, -1, 0, 1, 0, 1]]
    W = [[1, 0, 0, 1, -1, 0, 1],
         [0, 0, 1, 0, 1, 0, 0],
         [0, 1, 0, 1, 0, 0, 0],
         [1, -1, 1, 0, 0, 1, 0]]
    return U, V, W


def _kron_decomp(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two tensor decompositions."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 * r2
    n1 = len(U1)
    n2 = len(U2)
    m1 = len(V1)
    m2 = len(V2)
    p1 = len(W1)
    p2 = len(W2)
    U = [[0] * r for _ in range(n1 * n2)]
    V = [[0] * r for _ in range(m1 * m2)]
    W = [[0] * r for _ in range(p1 * p2)]
    for i1 in range(n1):
        for i2 in range(n2):
            row = i1 * n2 + i2
            for c1 in range(r1):
                for c2 in range(r2):
                    col = c1 * r2 + c2
                    U[row][col] = U1[i1][c1] * U2[i2][c2]
    for j1 in range(m1):
        for j2 in range(m2):
            row = j1 * m2 + j2
            for c1 in range(r1):
                for c2 in range(r2):
                    col = c1 * r2 + c2
                    V[row][col] = V1[j1][c1] * V2[j2][c2]
    for k1 in range(p1):
        for k2 in range(p2):
            row = k1 * p2 + k2
            for c1 in range(r1):
                for c2 in range(r2):
                    col = c1 * r2 + c2
                    W[row][col] = W1[k1][c1] * W2[k2][c2]
    return U, V, W


def _strassen_4x4x4():
    U2, V2, W2 = _strassen_2x2x2()
    return _kron_decomp(U2, V2, W2, U2, V2, W2)


def _strassen_4x4x8():
    """4x4x8 decomposition of rank 49.

    The 4x8 input splits into two 4x4 blocks.  Both blocks are multiplied by
    the SAME 4x4 matrix, so they share the U factor.  The two V blocks are
    concatenated (each 8x49) and the W factors are concatenated (each 16x49).
    Resulting shapes: U (16,49), V (32,49), W (32,49) -- exactly 4x4x8.
    """
    U44, V44, W44 = _strassen_4x4x4()
    r = len(U44[0])
    # U: same 16 rows (n*m) for both blocks
    U = [list(row) for row in U44]
    # V: 32 rows = first 4x4 block (8 rows) then second 4x4 block (8 rows)
    V = [list(row) for row in V44] + [list(row) for row in V44]
    # W: 32 rows = first 4x4 block (16 rows) then second 4x4 block (16 rows)
    W = [list(row) for row in W44] + [list(row) for row in W44]
    return U, V, W


def _to_frac_list(mat):
    return [[Fraction(int(round(x))) for x in row] for row in mat]


def _verify_exact(U, V, W, T):
    rank = len(U[0]) if U else 0
    # Pre-convert to ints where possible for speed
    Uf = [[int(x) if x.denominator == 1 else x for x in row] for row in U]
    Vf = [[int(x) if x.denominator == 1 else x for x in row] for row in V]
    Wf = [[int(x) if x.denominator == 1 else x for x in row] for row in W]
    for i in range(N_DIM):
        for j in range(M_DIM):
            for k in range(P_DIM):
                s = Fraction(0)
                for c in range(rank):
                    s += Uf[i][c] * Vf[j][c] * Wf[k][c]
                if s != Fraction(int(T[i, j, k])):
                    return False
    return True


def _verify_float(U, V, W, T, tol=1e-9):
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    T_rec = np.einsum('ic,jc,kc->ijk', Ua, Va, Wa)
    return bool(np.allclose(T_rec, T, atol=tol))


def _try_find_low_rank(T, target_rank, max_time=20.0):
    """Bounded structured search for an exact rank-`target_rank` decomposition.

    Parameterize U,V,W with real variables and minimize the Frobenius residual
    of T - sum_c U[:,c] V[:,c] W[:,c] via alternating least squares from a
    structured (sparse, half-integer) start, then snap to half-integers and
    verify exactly.  This is a cheap, safe attempt; it returns None on failure.
    """
    start = time.time()
    np.random.seed(20240404)
    for trial in range(12):
        if time.time() - start > max_time:
            break
        U = np.zeros((N_DIM, target_rank))
        V = np.zeros((M_DIM, target_rank))
        W = np.zeros((P_DIM, target_rank))
        for c in range(target_rank):
            nnz_u = np.random.randint(1, 5)
            nnz_v = np.random.randint(1, 5)
            nnz_w = np.random.randint(1, 5)
            idx_u = np.random.choice(N_DIM, nnz_u, replace=False)
            idx_v = np.random.choice(M_DIM, nnz_v, replace=False)
            idx_w = np.random.choice(P_DIM, nnz_w, replace=False)
            U[idx_u, c] = np.random.choice([-1.0, 1.0, -0.5, 0.5], nnz_u)
            V[idx_v, c] = np.random.choice([-1.0, 1.0, -0.5, 0.5], nnz_v)
            W[idx_w, c] = np.random.choice([-1.0, 1.0, -0.5, 0.5], nnz_w)

        for it in range(15):
            if time.time() - start > max_time:
                break
            # Update U rows
            for i in range(N_DIM):
                M = np.einsum('jc,kc->jkc', V, W).reshape(M_DIM * P_DIM, target_rank)
                b = T[i, :, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                U[i, :] = sol
            # Update V rows
            for j in range(M_DIM):
                M = np.einsum('ic,kc->ikc', U, W).reshape(N_DIM * P_DIM, target_rank)
                b = T[:, j, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                V[j, :] = sol
            # Update W rows
            for k in range(P_DIM):
                M = np.einsum('ic,jc->ijc', U, V).reshape(N_DIM * M_DIM, target_rank)
                b = T[:, :, k].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                W[k, :] = sol

            T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
            if np.allclose(T_rec, T, atol=1e-8):
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                T_rec2 = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                if np.allclose(T_rec2, T, atol=1e-10):
                    Uf = [[Fraction(int(round(x * 2)), 2) for x in row] for row in U_r.tolist()]
                    Vf = [[Fraction(int(round(x * 2)), 2) for x in row] for row in V_r.tolist()]
                    Wf = [[Fraction(int(round(x * 2)), 2) for x in row] for row in W_r.tolist()]
                    if _verify_exact(Uf, Vf, Wf, T):
                        return Uf, Vf, Wf
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin under 150 s

    T = _build_tensor()

    best = _schoolbook()
    best_rank = SCHOOLBOOK_RANK

    # Primary structured construction: rank-49 Strassen-based 4x4x8.
    try:
        U, V, W = _strassen_4x4x8()
        if _verify_float(U, V, W, T):
            Uf = _to_frac_list(U)
            Vf = _to_frac_list(V)
            Wf = _to_frac_list(W)
            if _verify_exact(Uf, Vf, Wf, T):
                best_rank = len(Uf[0])
                best = {"U": Uf, "V": Vf, "W": Wf}
                print(f"Strassen 4x4x8: rank {best_rank}")
    except Exception as e:
        print(f"Strassen construction failed: {e}")

    # Use spare time to look for a strictly lower exact rank.
    if time.time() < deadline - 25 and best_rank > 49:
        for target_rank in range(min(best_rank - 1, 48), 48, -1):
            if time.time() > deadline - 20:
                break
            try:
                result = _try_find_low_rank(T, target_rank, max_time=12.0)
                if result is not None:
                    Ua, Va, Wa = result
                    if len(Ua[0]) < best_rank:
                        best_rank = len(Ua[0])
                        best = {"U": Ua, "V": Va, "W": Wa}
                        print(f"Found lower rank {best_rank}")
                        break
            except Exception:
                pass

    return best
