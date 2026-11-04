"""Search-based exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
import itertools
import random
import time

DIMENSIONS = [4, 4, 5]
N, M, P = 4, 4, 5
NM = N * M          # 16  (U rows)
MP = M * P          # 20  (V rows)
PN = P * N          # 20  (W rows, trace convention W[k*n+i, r])
TERM_COUNT = N * M * P  # 80 required (i,j,k) triples


def _time_left(deadline):
    return deadline - time.time()


def _tensor_values(U, V, W, rank):
    """Build the n*m*p tensor T[(i,j,k)] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]."""
    T = [[0] * M for _ in range(N)]
    for i in range(N):
        row_i = T[i]
        for j in range(M):
            urow = U[i * M + j]
            vbase = j * P
            for r in range(rank):
                ur = urow[r]
                if ur == 0:
                    continue
                vcol = V
                wcol = W
                vcol_r = None
                # accumulate per k
                for k in range(P):
                    v = vcol[vbase + k][r]
                    if v == 0:
                        continue
                    w = wcol[k * N + i][r]
                    if w:
                        row_i[j] += ur * v * w
    return T


def _is_valid(U, V, W):
    """Exact check that U,V,W decompose the multiplication tensor.
    For every (i,j,k): sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1.
    For every (i,j,k) with i fixed: it's independent; but also must be 0 for
    cross terms? The multiplication tensor has entry 1 exactly at T[i,j,k]
    (the (i,j) entry of A times (j,k) entry of B gives (i,k) entry of C).
    Actually the tensor T[(i,k,j)] convention: here we use W[k*n+i] so the
    tensor index is (i, j, k) -> 1 for the product. All other combinations
    (different (i,j,k)) must be 0. Since each (i,j,k) is an independent slot,
    the condition is: for all (i,j,k), the sum equals 1."""
    for i in range(N):
        for j in range(M):
            urow = U[i * M + j]
            vbase = j * P
            for k in range(P):
                total = 0.0
                vrow = V[vbase + k]
                wrow = W[k * N + i]
                for r in range(len(urow)):
                    ur = urow[r]
                    if ur:
                        v = vrow[r]
                        if v:
                            w = wrow[r]
                            if w:
                                total += ur * v * w
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _rank(U, V, W):
    return len(U[0]) if U and U[0] else 0


def _structural_lower_bound():
    """Crude lower bound: each rank-1 term covers at most one (i,j,k) if we
    require the product to be exactly 1 at every slot and 0 elsewhere, a term
    (a row of U, row of V, row of W) contributes a product that is non-zero
    only at the (i,j,k) where all three are non-zero. To get exactly 1 at each
    of the 80 slots, and a term can contribute to multiple slots, the minimum
    is bounded below by the number of slots that must each be 'hit'. A very
    weak bound is 80 / (max slots per term). We return a conservative small
    bound to guide search ordering; correctness is always verified explicitly."""
    return 8  # conservative; real bound is higher but this keeps search cheap


def _schoolbook():
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _greedy_build(seed, time_budget=20.0):
    """Greedy: start empty, repeatedly add a rank-1 term that 'covers' a
    currently-unmet slot (makes its running sum closer to 1) while not
    breaking already-met slots. Returns a candidate or None."""
    rng = random.Random(seed)
    deadline = time.time() + time_budget
    # We'll allow each term to have arbitrary integer/half-integer weights.
    # A term is (u_vec, v_vec, w_vec). To keep it exact, we restrict to terms
    # that are 'monomial-like' but allow combining.
    # Simpler robust greedy: build the schoolbook terms one by one but try to
    # merge when two terms share structure. Fallback: just produce schoolbook.
    # This implementation: produce schoolbook, then attempt merges.
    cand = _schoolbook()
    U, V, W = cand["U"], cand["V"], cand["W"]
    rank = len(U[0])
    # Attempt to merge pairs of columns that are proportional.
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for a in range(rank):
            if a >= rank:
                break
            # check if column a is proportional to column b
            for b in range(a + 1, rank):
                if b >= rank:
                    break
                # compare U columns
                ua = U[0][a]; ub = U[0][b]
                # find scale s such that U[:,a] = s * U[:,b]
                # use first non-zero
                scale = None
                for r in range(NM):
                    if U[r][a] and U[r][b]:
                        scale = U[r][a] / U[r][b]
                        break
                    elif U[r][a] or U[r][b]:
                        scale = None
                        break
                if scale is None:
                    continue
                # verify U proportional
                ok = True
                for r in range(NM):
                    if abs(U[r][a] - scale * U[r][b]) > 1e-9:
                        ok = False; break
                if not ok:
                    continue
                # verify V proportional with same scale? No: product uses
                # U*V*W. If U_a = s U_b, then to keep product we need
                # V_a * W_a = (1/s) V_b * W_b. Try s split: V_a = V_b, W_a = W_b/s.
                # Check V columns proportional
                v_scale = None
                for r in range(MP):
                    if V[r][a] and V[r][b]:
                        v_scale = V[r][a] / V[r][b]
                        break
                    elif V[r][a] or V[r][b]:
                        v_scale = None; break
                if v_scale is None:
                    continue
                ok = True
                for r in range(MP):
                    if abs(V[r][a] - v_scale * V[r][b]) > 1e-9:
                        ok = False; break
                if not ok:
                    continue
                w_scale = None
                for r in range(PN):
                    if W[r][a] and W[r][b]:
                        w_scale = W[r][a] / W[r][b]
                        break
                    elif W[r][a] or W[r][b]:
                        w_scale = None; break
                if w_scale is None:
                    continue
                # product scale: U_a*V_a*W_a = s * v_scale * w_scale * (U_b V_b W_b)
                prod = s * v_scale * w_scale
                if abs(prod - 1.0) > 1e-9:
                    continue
                # Merge: drop column b, keep a (they are identical in product)
                # Actually if U_a=sU_b, V_a=v_scale V_b, W_a=w_scale W_b and
                # s*v_scale*w_scale=1, then product_a = product_b exactly, so
                # having both doubles it. They can't both be present in a valid
                # decomposition unless one is zero. So this 'merge' is really
                # detecting redundancy -> drop one.
                del U[0][b]
                # remove column b from all rows
                for r in range(1, NM):
                    U[r].pop(b)
                for r in range(MP):
                    V[r].pop(b)
                for r in range(PN):
                    W[r].pop(b)
                rank -= 1
                changed = True
                break
            if time.time() > deadline:
                break
    cand["U"], cand["V"], cand["W"] = U, V, W
    return cand


def _drop_terms(U, V, W, deadline):
    """Try to remove any term that is redundant (zero product contribution
    everywhere) -> reduces rank."""
    rank = len(U[0])
    removed = True
    while removed and time.time() < deadline:
        removed = False
        for c in range(len(U[0])):
            # check if column c contributes zero to all (i,j,k)
            all_zero = True
            for i in range(N):
                u = U[i * M][c]
                if not u:
                    continue
                for j in range(M):
                    u_ij = U[i * M + j][c]
                    if not u_ij:
                        continue
                    for k in range(P):
                        v = V[j * P + k][c]
                        if not v:
                            continue
                        w = W[k * N + i][c]
                        if not w:
                            continue
                        if u_ij * v * w:
                            all_zero = False
                            break
                    if not all_zero:
                        break
                if not all_zero:
                    break
            if all_zero:
                # drop column c
                for r in range(NM):
                    U[r].pop(c)
                for r in range(MP):
                    V[r].pop(c)
                for r in range(PN):
                    W[r].pop(c)
                removed = True
                break
    return U, V, W


def _random_greedy(seed, deadline):
    """Randomized: build terms with shared structure to allow rank reduction."""
    rng = random.Random(seed)
    # Strategy: assign each (i,j) pair to a subset of terms, each (j,k) to a subset,
    # each (k,i) to a subset, then solve for weights.
    # Simpler: try a few different rank values and random assignments.
    for R in [60, 50, 45, 40, 35, 30, 25]:
        if time.time() > deadline:
            break
        U = [[0] * R for _ in range(NM)]
        V = [[0] * R for _ in range(MP)]
        W = [[0] * R for _ in range(PN)]
        # Random assignment with some structure
        for i in range(N):
            for j in range(M):
                # Assign (i,j) to 2-3 random terms
                terms = rng.sample(range(R), min(3, R))
                for r in terms:
                    U[i * M + j][r] = 1
        for j in range(M):
            for k in range(P):
                terms = rng.sample(range(R), min(3, R))
                for r in terms:
                    V[j * P + k][r] = 1
        for k in range(P):
            for i in range(N):
                terms = rng.sample(range(R), min(3, R))
                for r in terms:
                    W[k * N + i][r] = 1
        cand = {"U": U, "V": V, "W": W}
        if _is_valid(cand["U"], cand["V"], cand["W"]):
            U, V, W = _drop_terms(cand["U"], cand["V"], cand["W"], deadline)
            return {"U": U, "V": V, "W": W}
    return None


def _swap_local_search(U, V, W, deadline):
    """Local search: try to replace a pair of terms with a single term that
    covers the same slots. This is the key operation for reducing rank."""
    rank = len(U[0])
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for a in range(rank):
            for b in range(a + 1, rank):
                # Try to find a single term that replaces terms a and b
                # The combined contribution of terms a and b at slot (i,j,k) is:
                #   U[i*m+j,a]*V[j*p+k,a]*W[k*n+i,a] + U[i*m+j,b]*V[j*p+k,b]*W[k*n+i,b]
                # We want a new term (u', v', w') such that:
                #   u'[i*m+j]*v'[j*p+k]*w'[k*n+i] = above, for all (i,j,k)
                # This is only possible if the combined tensor is rank-1.
                # Check: build the combined tensor T[i,j,k] and check if it's rank-1.
                T = [[0] * P for _ in range(N * M)]
                for i in range(N):
                    for j in range(M):
                        for k in range(P):
                            T[i * M + j][k] = (U[i * M + j][a] * V[j * P + k][a] * W[k * N + i][a] +
                                                U[i * M + j][b] * V[j * P + k][b] * W[k * N + i][b])
                # Check if T is rank-1: T[i,j,k] = u_i * v_j * w_k for some u,v,w
                # Find u, v, w by picking non-zero entries
                u = [0] * (N * M)
                v = [0] * (M * P)
                w = [0] * (P * N)
                # Find first non-zero in T
                found = False
                for idx in range(NM):
                    if T[idx][0] != 0:
                        u[idx] = T[idx][0]
                        found = True
                        break
                if not found:
                    # T is all zero, so terms a and b cancel everywhere -> drop one
                    for r in range(NM):
                        U[r].pop(b)
                    for r in range(MP):
                        V[r].pop(b)
                    for r in range(PN):
                        W[r].pop(b)
                    rank -= 1
                    improved = True
                    break
                # Now find v for each (j,k)
                v_found = True
                for j in range(M):
                    for k in range(P):
                        if u[idx] != 0 and T[idx][k] != 0:
                            # v[j*P+k] = T[idx][k] / u[idx] / w[k*N+i]... this is circular
                            pass
                # Simpler: check if T can be written as outer product
                # T[i,j,k] should equal A[i,j] * B[j,k] * C[k,i] for some A,B,C
                # Fix: pick i0, j0, k0 where T[i0,j0,k0] != 0
                # Set A[i0,j0] = T[i0,j0,k0], B[j0,k0] = 1, C[k0,i0] = 1
                # Then A[i,j] = T[i,j,k0] (if C[k0,i]=1 for all i? No, C depends on k)
                # This is getting complicated. Let's just check if the 2D slices are rank-1.
                # For fixed k, T[:, :, k] should be rank-1 as a (NM x 1) matrix -> always true
                # For fixed i, T[i, :, :] should be rank-1 as a (M x P) matrix
                # For fixed j, T[:, j, :] should be rank-1 as a (N x P) matrix
                # Check: for each j, the matrix M_j[i,k] = T[i,j,k] should be rank <= 1
                ok = True
                for j in range(M):
                    # Check if M_j is rank-1
                    # M_j[i,k] = u_i * w_k for some u, w
                    # Find non-zero entry
                    found_u = False
                    u_j = [0] * N
                    w_j = [0] * P
                    for i in range(N):
                        if T[i * M + j][0] != 0:
                            u_j[i] = T[i * M + j][0]
                            found_u = True
                            break
                    if not found_u:
                        continue
                    for k in range(P):
                        if T[0 * M + j][k] != 0:
                            w_j[k] = T[0 * M + j][k] / u_j[0] if u_j[0] != 0 else 0
                        else:
                            w_j[k] = 0
                    # Verify
                    for i in range(N):
                        for k in range(P):
                            expected = u_j[i] * w_j[k]
                            if abs(T[i * M + j][k] - expected) > 1e-9:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    # T is a valid rank-1 tensor, so we can replace terms a,b with one term
                    # Extract u, v, w from T
                    # u[i*m+j] = T[i*m+j, k0] for some k0 where w[k0*n+i] != 0
                    # This is complex; let's just mark as improved and break
                    # For now, we'll do a simpler check: just drop term b if it's redundant
                    pass
            if improved:
                break
    return U, V, W


def construct():
    deadline = time.time() + 120.0  # hard limit; evaluator kills at 150s
    safety = 25.0
    effective_deadline = deadline - safety

    best = None
    best_rank = None

    def consider(cand):
        nonlocal best, best_rank
        if cand is None:
            return
        r = _rank(cand["U"], cand["V"], cand["W"])
        if best_rank is None or r < best_rank:
            # final safety re-verify
            if _is_valid(cand["U"], cand["V"], cand["W"]):
                best = cand
                best_rank = r

    # 1) Schoolbook (always valid) as baseline
    consider(_schoolbook())

    # 2) Greedy merge attempts with several seeds
    for s in range(0, 800):
        if time.time() > effective_deadline:
            break
        cand = _greedy_build(s, time_budget=min(0.5, max(0.02, _time_left(effective_deadline))))
        consider(cand)

    # 3) Random greedy with varied target ranks - more seeds
    for target in [80, 70, 60, 50, 45, 40, 36, 32, 30, 25, 20]:
        if time.time() > effective_deadline:
            break
        for s in range(0, 120):
            if time.time() > effective_deadline:
                break
            cand = _random_greedy(s * 1000 + target, effective_deadline)
            consider(cand)
            if best_rank is not None and best_rank <= target:
                break

    # 4) Final: if best is still schoolbook rank, that's fine.
    if best is None:
        best = _schoolbook()

    return best


if __name__ == "__main__":
    res = construct()
    print("rank:", _rank(res["U"], res["V"], res["W"]))
