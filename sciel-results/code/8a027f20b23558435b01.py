"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = 142.0  # safety margin before the 150 s kill

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # ---- Build the block-Strassen base decomposition (rank 36) ----
    # 4 Strassen 2x2x2 blocks (28 terms) + 8 direct terms for the last column (k=4).
    U = [[0.0] * 36 for _ in range(n * m)]
    V = [[0.0] * 36 for _ in range(m * p)]
    W = [[0.0] * 36 for _ in range(p * n)]
    col = 0

    def strassen_block(y, z, w):
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        w00, w01, w10, w11 = w
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(*y00)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M2 = (y10+y11)z00
        U[u_idx(*y10)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w_idx(*w10)][col] += 1; W[w_idx(*w11)][col] += -1
        col += 1
        # M3 = y00(z01-z11)
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1; V[v_idx(*z11)][col] += -1
        W[w_idx(*w01)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M4 = y11(z10-z00)
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z00)][col] += -1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w10)][col] += 1
        col += 1
        # M5 = (y00+y01)z11
        U[u_idx(*y00)][col] += 1; U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += -1; W[w_idx(*w01)][col] += 1
        col += 1
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(*y10)][col] += 1; U[u_idx(*y00)][col] += -1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(*y01)][col] += 1; U[u_idx(*y11)][col] += -1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1
        col += 1

    AL = [(0,0),(0,1),(1,0),(1,1)]
    AR = [(0,2),(0,3),(1,2),(1,3)]
    B11 = [(0,0),(0,1),(1,0),(1,1)]
    B12 = [(0,2),(0,3),(1,2),(1,3)]
    B21 = [(2,0),(2,1),(3,0),(3,1)]
    B22 = [(2,2),(2,3),(3,2),(3,3)]
    C11 = [(0,0),(1,0),(0,1),(1,1)]
    C12 = [(2,0),(3,0),(2,1),(3,1)]

    strassen_block(AL, B11, C11)
    strassen_block(AR, B21, C11)
    strassen_block(AL, B12, C12)
    strassen_block(AR, B22, C12)

    # 8 direct terms for the last column (k=4).
    for (ai, aj) in [(0,0),(0,1),(1,0),(1,1)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1
    for (ai, aj) in [(0,2),(0,3),(1,2),(1,3)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1

    assert col == 36, col

    # ---- Vectorized verification ----
    # For each rank r, check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1.
    # Equivalent to: for each (j,k), sum_c (U[:,c]*W[:,c])_row(i*m+j) * V[j*p+k,c] == 1
    # We precompute the per-column outer products.
    def verify_fast(Ua, Va, Wa, R):
        # Ua: (n*m, R), Va: (m*p, R), Wa: (p*n, R)
        # For each column c: contribution to (i,j,k) is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Total = sum_c ... We check all (i,j,k) triples.
        # Build via einsum over the 40 triples.
        # triples[i,j,k] index: u = i*m+j, v = j*p+k, w = k*n+i
        # result[i,j,k] = sum_c U[u,c]*V[v,c]*W[w,c]
        # Use broadcasting: for each c, outer of U[:,c], V[:,c], W[:,c] is huge; instead loop c (R<=36).
        acc = np.zeros((n, m, p), dtype=np.float64)
        for c in range(R):
            uc = Ua[:, c]   # (n*m,)
            vc = Va[:, c]   # (m*p,)
            wc = Wa[:, c]   # (p*n,)
            # reshape to (n,m), (m,p), (p,n)
            uc_r = uc.reshape(n, m)
            vc_r = vc.reshape(m, p)
            wc_r = wc.reshape(p, n)
            # (i,j,k) = uc_r[i,j]*vc_r[j,k]*wc_r[k,i]
            # term[i,j,k] = uc_r[i,j]*vc_r[j,k]*wc_r[k,i]
            # einsum: 'ij,jk,ki->ijk'
            acc += np.einsum('ij,jk,ki->ijk', uc_r, vc_r, wc_r, optimize=True)
        target = np.ones((n, m, p))
        return np.allclose(acc, target, atol=1e-7, rtol=0.0)

    # ---- Merge machinery ----
    # Merging column c2 into column c1 with sign s: for every row a,
    #   U[a,c1] += s*U[a,c2]; V[a,c1] += s*V[a,c2]; W[a,c1] += s*W[a,c2]
    # and drop column c2. This is valid iff the resulting rank-1 tensor of c2
    # equals s * (rank-1 tensor of c1) restricted appropriately -- i.e. the two
    # columns are "proportional" in the tensor sense. We test by construction.
    def merge(Ua, Va, Wa, R, c1, c2, s):
        # return new arrays with c2 folded into c1 and dropped
        # new column order: [0..c2-1, c2+1..R-1] with c1 adjusted
        idx = [c for c in range(R) if c != c2]
        # new index of c1
        new_c1 = c1 if c1 < c2 else c1 - 1
        U2 = Ua[:, idx].copy()
        V2 = Va[:, idx].copy()
        W2 = Wa[:, idx].copy()
        # add s * old c2 into new_c1
        U2[:, new_c1] += s * Ua[:, c2]
        V2[:, new_c1] += s * Va[:, c2]
        W2[:, new_c1] += s * Wa[:, c2]
        return U2, V2, W2, R - 1

    # Convert base to arrays
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    R = 36

    if not verify_fast(Ua, Va, Wa, R):
        # Fallback: return the schoolbook-style base as-is (should not happen).
        return {"U": U, "V": V, "W": W}

    best_rank = R
    best = (Ua.copy(), Va.copy(), Wa.copy())

    # Greedy merge search: repeatedly find any valid merge, apply it, repeat.
    # Pool = all ordered pairs (c1, c2), c1 != c2, both signs.
    # To keep it fast, we only test a candidate merge if a cheap necessary
    # condition holds: the supports must be "compatible" (we just test directly
    # but cap the number of tests per iteration using the deadline).
    cur_U, cur_V, cur_W, cur_R = Ua, Va, Wa, R

    max_iters = 8
    for it in range(max_iters):
        if time.time() > deadline - 8.0:
            break
        if cur_R <= best_rank:
            # update best
            if cur_R < best_rank:
                best_rank = cur_R
                best = (cur_U.copy(), cur_V.copy(), cur_W.copy())
        found = False
        # candidate pool
        R_now = cur_R
        # order pairs to try: prefer Strassen/Strassen, then mixed, then direct
        # (heuristic). We just iterate all.
        tested = 0
        max_tests = 4000
        for c2 in range(R_now):
            if time.time() > deadline - 6.0:
                break
            for c1 in range(R_now):
                if c1 == c2:
                    continue
                for s in (1.0, -1.0):
                    tested += 1
                    if tested > max_tests:
                        break
                    U2, V2, W2, R2 = merge(cur_U, cur_V, cur_W, R_now, c1, c2, s)
                    if verify_fast(U2, V2, W2, R2):
                        cur_U, cur_V, cur_W, cur_R = U2, V2, W2, R2
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break

    # Final: ensure best is at least as good as current
    if cur_R < best_rank:
        best_rank = cur_R
        best = (cur_U.copy(), cur_V.copy(), cur_W.copy())

    bU, bV, bW = best
    # Convert back to list-of-lists with integer/half-integer entries.
    def to_lists(A):
        return [[(float(x) if abs(x - round(x)) > 1e-9 else int(round(x))) for x in row] for row in A]

    return {
        "U": to_lists(bU),
        "V": to_lists(bV),
        "W": to_lists(bW),
    }
# EVOLVE-BLOCK-END
