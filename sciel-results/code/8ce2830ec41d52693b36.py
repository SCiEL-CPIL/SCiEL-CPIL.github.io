"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition.

Design:
  - The schoolbook rank is n*m*p = 56.
  - View the 2x14 matrix B as 7 blocks of 2x2. Each block is multiplied by the
    same 2x2 matrix A. A 2x2*2x2 product has bilinear complexity 7 (Strassen),
    so 7 blocks give a rank-49 decomposition.
  - Attempt exact linear-algebraic rank reduction by finding integer linear
    dependencies among the 49 rank-1 tensors.

Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that exactly decompose the matrix multiplication tensor; score is the rank
"""
import numpy as np
import time
from fractions import Fraction

try:
    from scipy.linalg import lstsq as _scipy_lstsq
except Exception:
    _scipy_lstsq = None

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # hard safety margin inside the 150s wall-clock

    size_a = n * m      # 4
    size_b = m * p      # 28
    size_c = p * n      # 28

    # True multiplication tensor: T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i.
    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i * m + j, j * p + k, k * n + i] = 1.0

    # ------------------------------------------------------------------ #
    #  Verification helpers
    # ------------------------------------------------------------------ #
    def verify(U, V, W, atol=1e-9):
        Ua = np.asarray(U, dtype=np.float64)
        Va = np.asarray(V, dtype=np.float64)
        Wa = np.asarray(W, dtype=np.float64)
        T = np.einsum('ar,br,cr->abc', Ua, Va, Wa, optimize=True)
        return np.allclose(T, T_true, atol=atol)

    def schoolbook():
        rank = n * m * p  # 56
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # ------------------------------------------------------------------ #
    #  Rank-49 Strassen block decomposition
    # ------------------------------------------------------------------ #
    def strassen49():
        rank = 49
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)

        # Strassen terms for 2x2 * 2x2:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        for l in range(7):
            base = 7 * l
            k0 = 2 * l
            k1 = 2 * l + 1

            # A indices: A00=0, A01=1, A10=2, A11=3
            # B indices: B0k0=k0, B0k1=k1, B1k0=p+k0, B1k1=p+k1
            # C indices: C0k0=k0*n, C1k0=k0*n+1, C0k1=k1*n, C1k1=k1*n+1

            # M1 = (A00 + A11)(B00 + B11)
            c = base + 0
            U[0, c] += 1; U[3, c] += 1
            V[k0, c] += 1; V[p + k1, c] += 1
            W[k0 * n, c] += 1; W[k1 * n + 1, c] += 1

            # M2 = (A10 + A11)B00
            c = base + 1
            U[2, c] += 1; U[3, c] += 1
            V[k0, c] += 1
            W[k0 * n + 1, c] += 1; W[k1 * n + 1, c] -= 1

            # M3 = A00(B01 - B11)
            c = base + 2
            U[0, c] += 1
            V[k1, c] += 1; V[p + k1, c] -= 1
            W[k1 * n, c] += 1; W[k1 * n + 1, c] += 1

            # M4 = A11(B10 - B00)
            c = base + 3
            U[3, c] += 1
            V[p + k0, c] += 1; V[k0, c] -= 1
            W[k0 * n, c] += 1; W[k0 * n + 1, c] += 1

            # M5 = (A00 + A01)B11
            c = base + 4
            U[0, c] += 1; U[1, c] += 1
            V[p + k1, c] += 1
            W[k0 * n, c] -= 1; W[k1 * n, c] += 1

            # M6 = (A10 - A00)(B00 + B01)
            c = base + 5
            U[2, c] += 1; U[0, c] -= 1
            V[k0, c] += 1; V[k1, c] += 1
            W[k1 * n + 1, c] += 1

            # M7 = (A01 - A11)(B10 + B11)
            c = base + 6
            U[1, c] += 1; U[3, c] -= 1
            V[p + k0, c] += 1; V[p + k1, c] += 1
            W[k0 * n, c] += 1

        return U, V, W

    # ------------------------------------------------------------------ #
    #  Exact linear-algebraic rank reduction
    # ------------------------------------------------------------------ #
    def _flat_tensor(Ur, Vr, Wr):
        return np.einsum('a,b,c->abc', Ur, Vr, Wr).reshape(-1)

    def _int_rref_nullspace_int(A):
        """Compute an integer basis for the right null space of integer matrix A (m x n)
        using exact Fraction arithmetic with row-reduction. Returns list of integer
        vectors (length n) forming a basis for {x : A @ x = 0}."""
        from fractions import Fraction
        m, n = A.shape
        # Work on augmented copy with Fractions
        M = [[Fraction(int(A[i, j])) for j in range(n)] for i in range(m)]
        pivot_cols = []
        row = 0
        for col in range(n):
            if row >= m:
                break
            # Find pivot
            piv = None
            for r in range(row, m):
                if M[r][col] != 0:
                    piv = r
                    break
            if piv is None:
                continue
            M[row], M[piv] = M[piv], M[row]
            pv = M[row][col]
            # Normalize pivot row
            for j in range(col, n):
                M[row][j] = M[row][j] / pv
            # Eliminate other rows
            for r in range(m):
                if r == row:
                    continue
                factor = M[r][col]
                if factor != 0:
                    for j in range(col, n):
                        M[r][j] = M[r][j] - factor * M[row][j]
            pivot_cols.append(col)
            row += 1
        # Free columns
        free_cols = [c for c in range(n) if c not in pivot_cols]
        null_basis = []
        for fc in free_cols:
            v = [Fraction(0)] * n
            v[fc] = Fraction(1)
            for pi, pc in enumerate(pivot_cols):
                v[pc] = -M[pi][fc]
            # Convert to integers: find lcm of denominators
            denoms = [int(v[i].denominator) for i in range(n) if v[i] != 0]
            if not denoms:
                continue
            # lcm
            import math
            lcm = 1
            for d in denoms:
                lcm = lcm * d // math.gcd(lcm, d)
            iv = [int(v[i].numerator * lcm // v[i].denominator) for i in range(n)]
            # Normalize by gcd
            g = 0
            for x in iv:
                g = math.gcd(g, abs(x))
            if g > 1:
                iv = [x // g for x in iv]
            null_basis.append(iv)
        return null_basis

    def reduce_rank(U, V, W, max_iter=5):
        """Try to find a strictly smaller exact decomposition by exploiting
        linear dependencies among the rank-1 tensors. Returns (U,V,W) of the
        best (smallest) exact decomposition found, or None."""
        rank = W.shape[1]
        flat_dim = size_a * size_b * size_c

        # Build the matrix of rank-1 tensors as integer matrix (rank x flat_dim)
        Tmat = np.array([_flat_tensor(U[:, r], V[:, r], W[:, r]) for r in range(rank)],
                        dtype=np.int64)  # (rank, flat_dim)

        # Quick numeric rank check
        try:
            s = np.linalg.svd(Tmat.astype(np.float64), compute_uv=False)
        except np.linalg.LinAlgError:
            return None
        tol = 1e-9 * (s[0] if s.size else 1.0)
        num_rank = int(np.sum(s > tol))
        if num_rank >= rank:
            return None  # all independent: no linear reduction possible

        # Use exact integer null space computation
        # Transpose: we need null space of Tmat (rank x flat_dim) viewed as linear map
        # from R^rank to R^flat_dim. Null space has dimension rank - num_rank.
        # We want vectors y in R^rank such that Tmat @ y = 0.
        # Equivalently, y is in the left null space of Tmat^T, i.e., right null of Tmat.
        # Tmat is (rank, flat_dim), so right null space has dim rank - rank(Tmat).
        # Use exact method: compute null space of Tmat as integer matrix.
        nulls = _int_rref_nullspace_int(Tmat)

        if not nulls:
            return None

        # Greedy: iteratively drop a rank-1 tensor that is an integer combination of
        # the rest (a null vector with a unit entry). Rebuild and re-verify.
        Ucur, Vcur, Wcur = U.copy(), V.copy(), W.copy()
        kept = list(range(rank))
        progress = True
        while progress and time.time() < deadline - 5:
            progress = False
            cur_rank = len(kept)
            if cur_rank <= num_rank:
                break

            # Rebuild the current tensor matrix
            Tcur = np.array([_flat_tensor(Ucur[:, r], Vcur[:, r], Wcur[:, r]) for r in range(cur_rank)],
                            dtype=np.int64)

            # Find null space using exact method
            null_cur = _int_rref_nullspace_int(Tcur)

            # Find an integer null vector with a unit entry
            found = False
            for xi in null_cur:
                # Find a unit coordinate to eliminate
                elim = None
                for idx, val in enumerate(xi):
                    if val == 1 or val == -1:
                        elim = idx
                        break
                if elim is None:
                    continue

                # T_{kept[elim]} = -sum_{i != elim} xi_i * T_{kept[i]}
                # New column for kept[i] = old column + (-xi_i) * old column of elim
                oldU = Ucur[:, elim]
                oldV = Vcur[:, elim]
                oldW = Wcur[:, elim]

                newU_cols = []
                newV_cols = []
                newW_cols = []
                newkept = []
                for i in range(cur_rank):
                    if i == elim:
                        continue
                    coeff = -int(xi[i])
                    if coeff == 0:
                        newU_cols.append(Ucur[:, i])
                        newV_cols.append(Vcur[:, i])
                        newW_cols.append(Wcur[:, i])
                    else:
                        newU_cols.append(Ucur[:, i] + coeff * oldU)
                        newV_cols.append(Vcur[:, i] + coeff * oldV)
                        newW_cols.append(Wcur[:, i] + coeff * oldW)
                    newkept.append(kept[i])

                newU = np.column_stack(newU_cols)
                newV = np.column_stack(newV_cols)
                newW = np.column_stack(newW_cols)

                if verify(newU, newV, newW):
                    Ucur, Vcur, Wcur = newU, newV, newW
                    kept = newkept
                    progress = True
                    found = True
                    break

            if not found:
                break

        if len(kept) < rank:
            Uf = Ucur[:, kept]
            Vf = Vcur[:, kept]
            Wf = Wcur[:, kept]
            if verify(Uf, Vf, Wf):
                return Uf, Vf, Wf
        return None

    # ------------------------------------------------------------------ #
    #  Main
    # ------------------------------------------------------------------ #
    U, V, W = strassen49()
    if verify(U, V, W):
        # Attempt exact rank reduction; keep the 49 result if nothing smaller.
        reduced = reduce_rank(U, V, W)
        if reduced is not None:
            Ur, Vr, Wr = reduced
            if verify(Ur, Vr, Wr) and Ur.shape[1] < U.shape[1]:
                return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: guaranteed schoolbook rank 56.
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
