"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    for r in range(rank):
        # contribution to C[i,k] from factor r: (U[:,r] outer V[:,r]) contracted over j
        # C[i,k] = sum_j U[i*m+j,r] * V[j*p+k,r]
        # build per-r contribution matrix of shape (n, p)
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += Ua[i*m + j, r] * Va[j*p + k, r]
                if s * Wa[k*n + i, r] != 1.0:
                    return False
    return True

def _is_half_integer(x):
    return abs(x * 2 - round(x * 2)) < 1e-9

def _to_int_matrix(M):
    return [[int(round(v)) for v in row] for row in M]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115.0  # safety margin before 150s wall-clock kill

    # ---- Structural baseline: 6 x 2x2x2 Strassen blocks = rank 42 ----
    def build_strassen():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_strassen_block(i_start, j_start, k_start, offset):
            # M1: (A00+A11)(B00+B11)
            c = offset
            U[(i_start+0)*m + (j_start+0)][c] += 1
            U[(i_start+1)*m + (j_start+1)][c] += 1
            V[(j_start+0)*p + (k_start+0)][c] += 1
            V[(j_start+1)*p + (k_start+1)][c] += 1
            W[(k_start+0)*n + (i_start+0)][c] += 1
            W[(k_start+1)*n + (i_start+1)][c] += 1
            # M2: (A10+A11)B00
            c = offset + 1
            U[(i_start+1)*m + (j_start+0)][c] += 1
            U[(i_start+1)*m + (j_start+1)][c] += 1
            V[(j_start+0)*p + (k_start+0)][c] += 1
            W[(k_start+0)*n + (i_start+1)][c] += 1
            W[(k_start+1)*n + (i_start+1)][c] -= 1
            # M3: A00(B01-B11)
            c = offset + 2
            U[(i_start+0)*m + (j_start+0)][c] += 1
            V[(j_start+0)*p + (k_start+1)][c] += 1
            V[(j_start+1)*p + (k_start+1)][c] -= 1
            W[(k_start+1)*n + (i_start+0)][c] += 1
            W[(k_start+1)*n + (i_start+1)][c] += 1
            # M4: A11(B10-B00)
            c = offset + 3
            U[(i_start+1)*m + (j_start+1)][c] += 1
            V[(j_start+1)*p + (k_start+0)][c] += 1
            V[(j_start+0)*p + (k_start+0)][c] -= 1
            W[(k_start+0)*n + (i_start+0)][c] += 1
            W[(k_start+0)*n + (i_start+1)][c] += 1
            # M5: (A00+A01)B11
            c = offset + 4
            U[(i_start+0)*m + (j_start+0)][c] += 1
            U[(i_start+0)*m + (j_start+1)][c] += 1
            V[(j_start+1)*p + (k_start+1)][c] += 1
            W[(k_start+0)*n + (i_start+0)][c] -= 1
            W[(k_start+1)*n + (i_start+0)][c] += 1
            # M6: (A10-A00)(B00+B01)
            c = offset + 5
            U[(i_start+1)*m + (j_start+0)][c] += 1
            U[(i_start+0)*m + (j_start+0)][c] -= 1
            V[(j_start+0)*p + (k_start+0)][c] += 1
            V[(j_start+0)*p + (k_start+1)][c] += 1
            W[(k_start+1)*n + (i_start+1)][c] += 1
            # M7: (A01-A11)(B10+B11)
            c = offset + 6
            U[(i_start+0)*m + (j_start+1)][c] += 1
            U[(i_start+1)*m + (j_start+1)][c] -= 1
            V[(j_start+1)*p + (k_start+0)][c] += 1
            V[(j_start+1)*p + (k_start+1)][c] += 1
            W[(k_start+0)*n + (i_start+0)][c] += 1

        # A has 2 rows (i=0,1), 4 cols (j=0..3) -> 2 column blocks of width 2
        # B has 4 rows, 6 cols (k=0..5) -> 3 column blocks of width 2
        for jb in range(2):
            for kb in range(3):
                add_strassen_block(0, 2*jb, 2*kb, 7*(3*jb + kb))
        return U, V, W, rank

    U, V, W, rank = build_strassen()
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook rank 48 (always valid)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_U, best_V, best_W, best_rank = U, V, W, rank

    # ---- Structural refinement: search for a lower-rank decomposition ----
    # The 2x4x6 tensor is a Kronecker/product structure. We exploit it by
    # treating the 4 middle dimension as two independent 2-dim channels and the
    # 6 output as three 2-dim channels (already the Strassen form). To go below
    # 42 we look for a rank-r decomposition with r < 42 via a randomized
    # gradient-free refinement: start from the rank-42 factor matrices, then
    # attempt to zero out / merge columns while preserving exactness.
    #
    # Because exactness is a hard equality constraint, we use a feasibility
    # repair: for a target rank r, we keep r columns and solve a small linear
    # system per output entry to repair W (which is linear in W given U,V).
    # Specifically, for fixed U,V of width r, W must satisfy for each (i,k):
    #   sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is linear in W. We set up the linear system and check consistency.

    def try_rank(r):
        """Attempt to find a valid rank-r decomposition. Returns (U,V,W) or None."""
        # We reuse the rank-42 U,V,W as a starting point and try to project
        # down to r columns by selecting r linearly-independent "directions".
        # Build the per-column contribution tensors T_r[i,k] (n x p) where
        # T_r = sum_j (U[:,r] outer V[:,r]). Then we need W[r] such that
        # sum_r W[r] * T_r[i,k] = 1 for all (i,k).
        # Represent each column r by its (n*p) vector t_r. We need:
        #   sum_r W[r] * t_r = 1  (vector of length n*p)
        # This is solvable iff the t_r span the all-ones vector.
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        # Build t_r for r in 0..rank-1
        cols = []
        for r in range(rank):
            t = np.zeros(n * p)
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += Ua[i*m + j, r] * Va[j*p + k, r]
                    t[k*n + i] = s
            cols.append(t)
        cols = np.array(cols)  # (rank, n*p)
        ones = np.ones(n * p)
        # We need a subset of r columns and weights W such that sum = ones.
        # Simplest: use r columns that span ones. Try the first r columns.
        # Solve least-squares; if exact, good.
        A = cols[:r]  # (r, n*p)
        # solve A^T w = ones  (w of length r)
        At = A.T  # (n*p, r)
        try:
            w, res, rank_, sv = np.linalg.lstsq(At, ones, rcond=None)
        except np.linalg.LinAlgError:
            return None
        # check exactness
        if not np.allclose(At @ w, ones, atol=1e-6):
            return None
        # Build new U,V,W using first r columns and weights w
        nU = [[int(round(Ua[i*m+j, r2])) for r2 in range(r)] for i in range(n) for j in range(m)]
        nV = [[int(round(Va[j*p+k, r2])) for r2 in range(r)] for j in range(m) for k in range(p)]
        nW = [[int(round(w[r2]))] for r2 in range(r)]  # placeholder
        # W is indexed by (k*n+i)
        nW = [[0] * r for _ in range(p * n)]
        for r2 in range(r):
            for idx in range(p * n):
                nW[idx][r2] = int(round(w[r2]))
        if _verify(nU, nV, nW, n, m, p, r):
            return nU, nV, nW
        return None

    # Try to reduce rank below 42, but only if we have spare time.
    for r in range(best_rank - 1, max(best_rank - 8, 30), -1):
        if time.time() > deadline - 20.0:
            break
        res = try_rank(r)
        if res is not None:
            best_U, best_V, best_W, best_rank = res
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
