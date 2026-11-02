import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]
N, M, P = DIMENSIONS  # 4, 5, 6
NM, MP, PN = N * M, M * P, P * N  # 20, 30, 24


def _u_idx(i, j):
    return i * M + j


def _v_idx(j, k):
    return j * P + k


def _w_idx(k, i):
    return k * N + i


def schoolbook():
    """Exact schoolbook decomposition, rank = N*M*P = 120."""
    r = NM * MP  # wrong; rank = N*M*P
    r = N * M * P
    U = [[Fraction(0)] * r for _ in range(NM)]
    V = [[Fraction(0)] * r for _ in range(MP)]
    W = [[Fraction(0)] * r for _ in range(PN)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[_u_idx(i, j)][col] = Fraction(1)
                V[_v_idx(j, k)][col] = Fraction(1)
                W[_w_idx(k, i)][col] = Fraction(1)
                col += 1
    return U, V, W


def strassen_222():
    """Strassen 2x2*2x2->2x2 rank-7 decomposition.
    U: (4,7) rows A[i,j] with i,j in {0,1} (row-major i*2+j)
    V: (4,7) rows B[j,k]
    W: (4,7) rows C[k,i]
    """
    U = [[Fraction(0)] * 7 for _ in range(4)]
    V = [[Fraction(0)] * 7 for _ in range(4)]
    W = [[Fraction(0)] * 7 for _ in range(4)]
    # m1 = (a11+a22)(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # m2 = (a21+a22)b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    # m3 = a11(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # m4 = a22(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    # m5 = (a11+a12)b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    # m6 = (a21-a11)(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # m7 = (a12-a22)(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W


def _as_float_mat(X):
    return np.array([[float(x) for x in row] for row in X], dtype=float)


def _residual(U, V, W):
    """Return the residual 3-way tensor T[i,j,k] = 1 - sum_r U[i,j]V[j,k]W[k,i].
    Computed efficiently: R[i,j,k] = 1 - (U * (V*W))[i,j,k]."""
    Uf = _as_float_mat(U)
    Vf = _as_float_mat(V)
    Wf = _as_float_mat(W)
    # S[j,k] = sum_r V[j,k]*W[k,i]... but W is per (k,i). Build M[j,i,k] = sum_r V[j,k]*W[k,i]
    # M[j,i,k] = (V^T @ W)[k,j]?? Let's do: for each r, outer of V[:,r] (over j,k) and W[:,r] (over k,i)
    # We want T[j,k,i] = sum_r V[j,k,r]*W[k,i,r]. Then R[i,j,k] = 1 - T[j,k,i].
    # Build T of shape (M, P, N): T[j,k,i] = sum_r V[j,k,r]*W[k,i,r]
    # = einsum('jkr,kr->jki', V, W)
    T = np.einsum('jkr,kr->jki', Vf, Wf)  # shape (M,P,N)
    # U is (N*M, r) -> reshape (N,M,r). R[i,j,k] = 1 - sum_r U[i,j,r]*T[j,k,i]
    U3 = Uf.reshape(N, M, -1)
    R = 1.0 - np.einsum('ijr,jki->ijk', U3, T)
    return R  # shape (N, M, P)


def _residual_norm(R):
    return float(np.abs(R).max())


def _add_rank1(U, V, W, u_vec, v_vec, w_vec):
    """Add a rank-1 component: u_vec (N*M,), v_vec (M*P,), w_vec (P*N,) with Fraction values."""
    r = len(U[0])
    U.append(list(u_vec))
    V.append(list(v_vec))
    W.append(list(w_vec))
    return r + 1


def _try_peel(R, U, V, W, candidates, max_iter, deadline):
    """Greedy peeling: try to extract rank-1 components from residual.
    candidates: list of (u_pattern, v_pattern, w_pattern) each a dict of idx->value (Fraction).
    We attempt to add components that reduce residual norm."""
    best_rank = len(U[0]) if U else 0
    # Work in float for speed
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        for (uc, vc, wc) in candidates:
            # Build full vectors
            u_vec = [Fraction(0)] * NM
            v_vec = [Fraction(0)] * MP
            w_vec = [Fraction(0)] * PN
            for idx, val in uc.items():
                u_vec[idx] = val
            for idx, val in vc.items():
                v_vec[idx] = val
            for idx, val in wc.items():
                w_vec[idx] = val
            # Compute effect on residual: adding component changes R by -outer(u,v,w)
            # new R = R - u[i,j]*v[j,k]*w[k,i]
            uf = np.array(u_vec, dtype=float)
            vf = np.array(v_vec, dtype=float)
            wf = np.array(w_vec, dtype=float)
            comp = np.einsum('ij,jk,ki->ijk', uf.reshape(N, M), vf.reshape(M, P), wf.reshape(P, N))
            newR = R - comp
            newnorm = _residual_norm(newR)
            if newnorm < _residual_norm(R) - 1e-12:
                # Accept
                U.append(u_vec)
                V.append(v_vec)
                W.append(w_vec)
                R = newR
                improved = True
                break
        if not improved:
            break
    return U, V, W


def _build_block_decomposition():
    """The current program's block construction (Strassen + schoolbook), rank 108.
    Returns (U, V, W) as list-of-lists of Fraction, and the rank."""
    total_rank = 108
    U = [[Fraction(0)] * total_rank for _ in range(NM)]
    V = [[Fraction(0)] * total_rank for _ in range(MP)]
    W = [[Fraction(0)] * total_rank for _ in range(PN)]
    offset = 0

    def add_sub(U_sub, V_sub, W_sub, u_map, v_map, w_map, off):
        for r in range(len(U_sub[0])):
            for (sub_u, big_u) in u_map:
                if U_sub[sub_u][r] != 0:
                    U[big_u][off + r] = U_sub[sub_u][r]
            for (sub_v, big_v) in v_map:
                if V_sub[sub_v][r] != 0:
                    V[big_v][off + r] = V_sub[sub_v][r]
            for (sub_w, big_w) in w_map:
                if W_sub[sub_w][r] != 0:
                    W[big_w][off + r] = W_sub[sub_w][r]
        return off + len(U_sub[0])

    S_U, S_V, S_W = strassen_222()

    def schoolbook_212():
        U_s = [[Fraction(0)] * 4 for _ in range(2)]
        V_s = [[Fraction(0)] * 4 for _ in range(2)]
        W_s = [[Fraction(0)] * 4 for _ in range(4)]
        col = 0
        for i in range(2):
            for k in range(2):
                U_s[i][col] = 1
                V_s[0][col] = 1
                W_s[k * 2 + i][col] = 1
                col += 1
        return U_s, V_s, W_s

    SB_U, SB_V, SB_W = schoolbook_212()

    # 18 blocks as in current program
    blocks = []
    # (u_map, v_map, w_map, sub)
    # Strassen blocks (12) and schoolbook (6)
    def sm(u0, v0, w0):
        return (
            [(0*5+u0+0,0),(0*5+u0+1,1),(1*5+u0+0,2),(1*5+u0+1,3)],
            [(v0*6+0,0),(v0*6+1,1),(v0+1 if False else (v0)*6+0,2),(v0*6+1,3)],
            None
        )
    # Just replicate the exact 18 blocks from the current program.
    # Block (0,0): A[0:2,0:2]*B[0:2,0:2]->C[0:2,0:2]
    blocks.append(([(0,0),(1,1),(5,2),(6,3)], [(0,0),(1,1),(6,2),(7,3)], [(0,0),(4,1),(1,2),(5,3)], 'S'))
    # Block (0,1): A[0:2,0:2]*B[0:2,2:4]->C[2:4,0:2]
    blocks.append(([(0,0),(1,1),(5,2),(6,3)], [(2,0),(3,1),(8,2),(9,3)], [(8,0),(12,1),(9,2),(13,3)], 'S'))
    # Block (0,2): A[0:2,0:2]*B[0:2,4:6]->C[4:6,0:2]
    blocks.append(([(0,0),(1,1),(5,2),(6,3)], [(4,0),(5,1),(10,2),(11,3)], [(16,0),(20,1),(17,2),(21,3)], 'S'))
    # Block (1,0): A[0:2,2:4]*B[2:4,0:2]->C[0:2,0:2]
    blocks.append(([(2,0),(3,1),(7,2),(8,3)], [(12,0),(13,1),(18,2),(19,3)], [(0,0),(4,1),(1,2),(5,3)], 'S'))
    # Block (1,1): A[0:2,2:4]*B[2:4,2:4]->C[2:4,0:2]
    blocks.append(([(2,0),(3,1),(7,2),(8,3)], [(14,0),(15,1),(20,2),(21,3)], [(8,0),(12,1),(9,2),(13,3)], 'S'))
    # Block (1,2): A[0:2,2:4]*B[2:4,4:6]->C[4:6,0:2]
    blocks.append(([(2,0),(3,1),(7,2),(8,3)], [(16,0),(17,1),(22,2),(23,3)], [(16,0),(20,1),(17,2),(21,3)], 'S'))
    # Block (2,0): A[0:2,4:5]*B[4:5,0:2]->C[0:2,0:2]  (schoolbook 2x1*1x2)
    blocks.append(([(4,0),(9,1)], [(24,0),(25,1)], [(0,0),(4,1),(1,2),(5,3)], 'SB'))
    # Block (2,1): A[0:2,4:5]*B[4:5,2:4]->C[2:4,0:2]
    blocks.append(([(4,0),(9,1)], [(26,0),(27,1)], [(8,0),(12,1),(9,2),(13,3)], 'SB'))
    # Block (2,2): A[0:2,4:5]*B[4:5,4:6]->C[4:6,0:2]
    blocks.append(([(4,0),(9,1)], [(28,0),(29,1)], [(16,0),(20,1),(17,2),(21,3)], 'SB'))
    # Block (3,0): A[2:4,0:2]*B[0:2,0:2]->C[0:2,2:4]
    blocks.append(([(10,0),(11,1),(15,2),(16,3)], [(0,0),(1,1),(6,2),(7,3)], [(2,0),(6,1),(3,2),(7,3)], 'S'))
    # Block (3,1): A[2:4,0:2]*B[0:2,2:4]->C[2:4,2:4]
    blocks.append(([(10,0),(11,1),(15,2),(16,3)], [(2,0),(3,1),(8,2),(9,3)], [(10,0),(14,1),(11,2),(15,3)], 'S'))
    # Block (3,2): A[2:4,0:2]*B[0:2,4:6]->C[4:6,2:4]
    blocks.append(([(10,0),(11,1),(15,2),(16,3)], [(4,0),(5,1),(10,2),(11,3)], [(18,0),(22,1),(19,2),(23,3)], 'S'))
    # Block (4,0): A[2:4,2:4]*B[2:4,0:2]->C[0:2,2:4]
    blocks.append(([(12,0),(13,1),(17,2),(18,3)], [(12,0),(13,1),(18,2),(19,3)], [(2,0),(6,1),(3,2),(7,3)], 'S'))
    # Block (4,1): A[2:4,2:4]*B[2:4,2:4]->C[2:4,2:4]
    blocks.append(([(12,0),(13,1),(17,2),(18,3)], [(14,0),(15,1),(20,2),(21,3)], [(10,0),(14,1),(11,2),(15,3)], 'S'))
    # Block (4,2): A[2:4,2:4]*B[2:4,4:6]->C[4:6,2:4]
    blocks.append(([(12,0),(13,1),(17,2),(18,3)], [(16,0),(17,1),(22,2),(23,3)], [(18,0),(22,1),(19,2),(23,3)], 'S'))
    # Block (5,0): A[2:4,4:5]*B[4:5,0:2]->C[0:2,2:4]
    blocks.append(([(14,0),(19,1)], [(24,0),(25,1)], [(2,0),(6,1),(3,2),(7,3)], 'SB'))
    # Block (5,1): A[2:4,4:5]*B[4:5,2:4]->C[2:4,2:4]
    blocks.append(([(14,0),(19,1)], [(26,0),(27,1)], [(10,0),(14,1),(11,2),(15,3)], 'SB'))
    # Block (5,2): A[2:4,4:5]*B[4:5,4:6]->C[4:6,2:4]
    blocks.append(([(14,0),(19,1)], [(28,0),(29,1)], [(18,0),(22,1),(19,2),(23,3)], 'SB'))

    for (u_map, v_map, w_map, kind) in blocks:
        if kind == 'S':
            offset = add_sub(S_U, S_V, S_W, u_map, v_map, w_map, offset)
        else:
            offset = add_sub(SB_U, SB_V, SB_W, u_map, v_map, w_map, offset)

    if offset != total_rank:
        return None
    # Verify
    R = _residual(U, V, W)
    if _residual_norm(R) > 1e-9:
        return None
    return (U, V, W, total_rank)


def construct():
    start = time.time()
    deadline = start + 115

    # Best known: try block decomposition (rank 108)
    best = None
    try:
        res = _build_block_decomposition()
        if res is not None:
            best = res
    except Exception:
        best = None

    # Fallback to schoolbook if block failed
    if best is None:
        U, V, W = schoolbook()
        best = (U, V, W, N * M * P)

    U, V, W, cur_rank = best

    # Now attempt peeling / rank reduction on top of the block decomposition.
    # Strategy: we have a valid rank-108 decomposition. We try to find a smaller one
    # by treating the multiplication tensor as a 3-way array and doing greedy rank-1
    # extraction with integer/half-integer support, seeded by structured candidates.

    # Build candidate rank-1 components. A rank-1 component for matmul has the form:
    # u[i,j], v[j,k], w[k,i] such that the product u[i,j]*v[j,k]*w[k,i] approximates
    # a "slice" of the identity tensor. We generate structured candidates from
    # combinations of row/col index subsets (like Strassen terms) and random ones.

    # First, compute residual of the current decomposition (should be ~0).
    R = _residual(U, V, W)

    # Instead of peeling from residual (which is ~0), we attempt a GLOBAL re-optimization:
    # try to find a rank-r decomposition for r < 108 by greedy rank-1 extraction from
    # the target tensor T[i,j,k]=1.

    # Greedy rank-1 extraction from target tensor:
    # Start with residual = T (all ones), extract rank-1 components.
    # A rank-1 component is (u,v,w) with entries in {0, +/-1, +/-1/2}.
    # We want to minimize rank such that sum_r u_r[i,j] v_r[j,k] w_r[k,i] = 1 for all i,j,k.

    # Generate a pool of structured candidates.
    candidates = []

    # Type 1: single-entry u, single-entry v, single-entry w (schoolbook basis)
    # These give u[i0,j0]=1, v[j0,k0]=1, w[k0,i0]=1 -> covers exactly (i0,j0,k0).
    # Not useful for reduction (that's schoolbook).

    # Type 2: Strassen-like: u has 2 entries, v has 2 entries, w has 2 entries.
    # Enumerate small support patterns.
    # For reduction, we want components that cover multiple (i,j,k) tuples at once.

    # We'll do a randomized search: try random rank-1 components with small support
    # and see if they reduce the residual norm of the target tensor.

    # Target tensor T = ones (N,M,P). Residual starts as T.
    T = np.ones((N, M, P), dtype=float)
    residual = T.copy()

    # We'll build a new decomposition from scratch via greedy extraction.
    new_U = []
    new_V = []
    new_W = []

    # Candidate generator: random sparse vectors with values in {0,1,-1} or {0,1,-1,0.5,-0.5}
    import random
    random.seed(42)

    def random_candidate():
        # Support sizes: u in {1,2,3}, v in {1,2,3}, w in {1,2,3}
        su = random.randint(1, 3)
        sv = random.randint(1, 3)
        sw = random.randint(1, 3)
        u_idx = random.sample(range(NM), su)
        v_idx = random.sample(range(MP), sv)
        w_idx = random.sample(range(PN), sw)
        vals = [1, -1, 0.5, -0.5]
        u_vec = np.zeros(NM)
        v_vec = np.zeros(MP)
        w_vec = np.zeros(PN)
        for idx in u_idx:
            u_vec[idx] = random.choice(vals)
        for idx in v_idx:
            v_vec[idx] = random.choice(vals)
        for idx in w_idx:
            w_vec[idx] = random.choice(vals)
        return u_vec, v_vec, w_vec

    # Structured candidates: Strassen-like patterns
    # For each pair of row-blocks, col-blocks, etc.
    # We'll add the 18 Strassen/schoolbook block components as candidates too.

    # Greedy extraction loop
    max_iter = 2000
    iters_done = 0
    improved_count = 0
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        iters_done += 1
        best_gain = 1e-12
        best_cand = None
        # Try a batch of random candidates
        for _ in range(200):
            u_vec, v_vec, w_vec = random_candidate()
            comp = np.einsum('ij,jk,ki->ijk',
                             u_vec.reshape(N, M),
                             v_vec.reshape(M, P),
                             w_vec.reshape(P, N))
            newR = residual - comp
            # We want to reduce the L2 norm of residual, but also we need exactness.
            # For greedy, minimize ||newR||_F
            gain = float(np.sum(residual**2) - np.sum(newR**2))
            if gain > best_gain:
                best_gain = gain
                best_cand = (u_vec, v_vec, w_vec, newR)
        if best_cand is None:
            break
        u_vec, v_vec, w_vec, newR = best_cand
        # Convert to Fraction (round to nearest half-integer)
        def to_frac(x):
            hx = round(x * 2) / 2
            return Fraction(int(round(x * 2)), 2)
        new_U.append([to_frac(x) for x in u_vec])
        new_V.append([to_frac(x) for x in v_vec])
        new_W.append([to_frac(x) for x in w_vec])
        residual = newR
        improved_count += 1
        if _residual_norm(residual) < 1e-9:
            break

    # Check if the greedy extraction achieved exact decomposition
    if _residual_norm(residual) < 1e-9:
        new_rank = len(new_U)
        if new_rank < cur_rank:
            # Verify exactly
            R2 = _residual(new_U, new_V, new_W)
            if _residual_norm(R2) < 1e-9:
                U, V, W, cur_rank = new_U, new_V, new_W, new_rank

    # If we still have the block decomposition, that's our best.
    # Convert to float and return.
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W],
    }
