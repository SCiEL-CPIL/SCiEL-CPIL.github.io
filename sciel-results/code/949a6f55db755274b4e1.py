# EVOLVE-BLOCK-START
import numpy as np
import time


def _initial_p(a, b, rng):
    """Rank-matching start: heavy-flow rows to low-distance (central) rows."""
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(len(a), dtype=np.int64)
    p[heavy] = central
    # small random perturbation to diversify
    n = len(p)
    for _ in range(max(1, n // 50)):
        i, j = rng.integers(0, n, 2)
        p[i], p[j] = p[j], p[i]
    return p


def _compute_cost(a, b, p):
    """Exact integer cost sum_{i,j} a[i,j]*b[p[i],p[j]]."""
    return int(np.einsum('ij,ik,jk->', a, a, b[np.ix_(p, p)], optimize=True))


def _full_cost(a, b, p):
    return int(np.einsum('ij,ik,jk->', a, a, b[np.ix_(p, p)], optimize=True))


def solve(a, b):
    n = len(a)
    rng = np.random.default_rng(12345)

    # Precompute: a is symmetric? assume general. Work with row-wise.
    # cost = sum_ij a[i,j] * b[p[i], p[j]]
    # Define C[i][j] contribution. For a move of facility i from loc old to loc new:
    #   delta = sum_k a[i,k]*(b[new,p[k]] - b[old,p[k]]) + sum_k a[k,i]*(b[p[k],new]-b[p[k],old])
    # If a symmetric: delta = 2 * (a[i] . b[new] - a[i] . b[old])  (with current p for the b rows)
    # We'll use the general formula but vectorized via dot products.

    # Precompute b row sums not needed; we use dot(a[i], b[loc]).
    # a[i] dot b[loc] for all loc = b @ a[i]  (since b[loc] is a row)
    # Actually b @ a[i] gives vector over loc: sum_j b[loc,j]*a[i,j]. Good.

    start_time = time.time()
    DEADLINE = 150.0
    SAFETY = 8.0  # seconds margin
    effective_deadline = start_time + DEADLINE - SAFETY

    best_p = None
    best_cost = None

    # Precompute a rows as float for fast dot (a is int64, convert once)
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)

    # Precompute for each facility i, the vector over locations: v_i[loc] = sum_j a[i,j]*b[loc,j]
    # That is v_i = b @ a[i]
    # And w_i[loc] = sum_j a[j,i]*b[j? ] ... for the second term we need sum_k a[k,i]*b[p[k], new]
    # = sum_k b[p[k], new] * a[k,i] = (b[:, new]) dot a[:, i] over k but p[k] varies.
    # Better: define for each location L, the "column potential" from incoming flows.
    # Let's define for each facility i and location L:
    #   out_i[L] = sum_j a[i,j] * b[L, j]   (flow from i to others, placed at L)
    #   in_i[L]  = sum_j a[j,i] * b[j, L]   (flow from others to i, placed at L)
    # Then total cost = sum_i (out_i[p[i]] + in_i[p[i]])? Let's verify:
    #   sum_i out_i[p[i]] = sum_i sum_j a[i,j] b[p[i],j]
    #   sum_i in_i[p[i]]  = sum_i sum_j a[j,i] b[j,p[i]] = sum_{j,i} a[j,i] b[j,p[i]]
    #   reindex (i<->j): = sum_{i,j} a[i,j] b[i? no]. Let's rename: in term = sum_{j,i} a[j,i] b[j,p[i]].
    #   Let i' = j, j' = i: = sum_{i',j'} a[i',j'] b[i', p[j']]. That's not the same as b[p[i],p[j]].
    # Hmm, the original is sum_{i,j} a[i,j] b[p[i],p[j]].
    #   out term = sum_{i,j} a[i,j] b[p[i], j]  -- j is a facility index, b[p[i],j] uses j as location. WRONG.

    # Let me reconsider. The cost is sum_{i,j} a[i,j] * b[p[i], p[j]].
    # For a fixed facility i at location L=p[i], its contribution involves:
    #   sum_j a[i,j] * b[L, p[j]]  (i as source, j as dest)
    #   sum_j a[j,i] * b[p[j], L]  (j as source, i as dest)
    # The first: sum_j a[i,j]*b[L,p[j]]. This depends on all p[j].
    # The second: sum_j a[j,i]*b[p[j],L].
    # When we move ONLY facility i from old to new (keeping others fixed):
    #   delta = sum_j a[i,j]*(b[new,p[j]] - b[old,p[j]]) + sum_j a[j,i]*(b[p[j],new]-b[p[j],old])
    # Term1 = (b[new] - b[old]) dot a[i]   (a[i] over j, b[new,p[j]] = b[new,:] dot over p[j]... )
    #   Actually sum_j a[i,j]*b[new,p[j]]: p[j] is the location of facility j. So this is sum_j a[i,j] * b[new, p[j]].
    #   Define for each location L: g_i(L) = sum_j a[i,j]*b[L, p[j]]. This is not simply b[L] dot a[i]
    #   because b[L, p[j]] uses p[j] not j.
    #
    # This is the crux: the "destination location" for flow i->j is p[j], not j.
    # So we cannot precompute a simple per-location vector independent of p.

    # Correct O(n) approach:
    #   delta = sum_j a[i,j]*(b[new,p[j]] - b[old,p[j]]) + sum_j a[j,i]*(b[p[j],new]-b[p[j],old])
    #   = sum_j a[i,j]*b[new,p[j]] - sum_j a[i,j]*b[old,p[j]] + sum_j a[j,i]*b[p[j],new] - sum_j a[j,i]*b[p[j],old]
    #
    # Define:
    #   F_src[i][L] = sum_j a[i,j] * b[L, p[j]]   (i sends, placed at L)
    #   F_dst[i][L] = sum_j a[j,i] * b[p[j], L]   (i receives, placed at L)
    #   Then total cost = sum_i (F_src[i][p[i]] + F_dst[i][p[i]])
    #   And delta for moving i: old->new = (F_src[i][new]+F_dst[i][new]) - (F_src[i][old]+F_dst[i][old])
    #
    # But F_src[i][L] = sum_j a[i,j]*b[L,p[j]]. To update when p changes, F matrices change.
    # However, for a SINGLE move of facility i, only p[i] changes. Does F_src/F_dst for OTHER
    # facilities change? F_src[k][L] = sum_j a[k,j]*b[L,p[j]]. Only p[i] changed, so term j=i changes:
    #   F_src[k][L] new = F_src[k][L] old + a[k,i]*(b[L,new]-b[L,old])
    #   F_dst[k][L] new = F_dst[k][L] old + a[i,k]*(b[new,L]-b[old,L])
    # So we can maintain F_src, F_dst incrementally. But that's O(n) to update per move, and
    # evaluating delta for candidate move of i is O(1) once F is current. But F must reflect the
    # CURRENT p (with i at old). When we evaluate a hypothetical move i:old->new, we use current F
    # (which has i at old) — but the delta formula above uses F_src[i][new] which under current F
    # (i at old) is computed with p[i]=old, but the formula sum_j a[i,j]*b[new,p[j]] with p[i]=old
    # is exactly what we want for the "new" side EXCEPT the j=i term uses p[i]=old not new.
    #
    # This is getting complicated. Let me just compute delta directly in O(n) per candidate,
    # and use vectorized numpy for the n candidates -> O(n^2) per iteration but with low constant.
    # For n=150, n^2 = 22500 per iteration, very fast. We can do thousands of iterations.

    def delta_move(p, i, new):
        """Cost change when moving facility i from p[i] to new. O(n)."""
        old = p[i]
        if old == new:
            return 0
        # term1 = sum_j a[i,j]*(b[new,p[j]] - b[old,p[j]])
        # term2 = sum_j a[j,i]*(b[p[j],new] - b[p[j],old])
        # Vectorized:
        # b[new, p] is a vector over j (since p is array of locs for each j)
        diff1 = b_float[new, p] - b_float[old, p]   # shape (n,)
        diff2 = b_float[p, new] - b_float[p, old]   # shape (n,)
        d = float(a_float[i] @ diff1 + a_float[:, i] @ diff2)
        return d

    def all_deltas(p):
        """For each facility i, delta of moving i to every location. O(n^2) vectorized."""
        # For facility i, new location L:
        #   delta(i, L) = a[i] . (b[L, p] - b[p[i], p]) + a[:,i] . (b[p, L] - b[p, p[i]])
        # Precompute:
        #   bLp = b @ ? Let's build matrix M1[i, L] = a[i] . b[L, p] = sum_j a[i,j]*b[L,p[j]]
        #     = sum_j a[i,j]*b[L,p[j]]. Let Bp = b[:, p] (n x n, Bp[L,j]=b[L,p[j]]). Then M1[i,L]=a[i]@Bp[L,:] = (a @ Bp.T)[i,L]? 
        #     a @ Bp.T: (a[i,j]*Bp.T[j,L]) = a[i,j]*Bp[L,j]=a[i,j]*b[L,p[j]]. Yes M1 = a @ Bp.T, shape (n,n), M1[i,L].
        #   M2[i, L] = a[:,i] . b[p, L] = sum_j a[j,i]*b[p[j],L] = (a.T @ ?) a[:,i] is column i = a.T[i,:].
        #     sum_j a.T[i,j]*b[p[j],L]. Let Bp2 = b[p, :] (n x n, Bp2[j,L]=b[p[j],L]). Then M2[i,L]=a.T[i]@Bp2[:,L]=(a.T@Bp2)[i,L].
        #   base1[i] = a[i] . b[p[i], p] = M1[i, p[i]]
        #   base2[i] = a[:,i] . b[p, p[i]] = M2[i, p[i]]
        #   delta(i,L) = M1[i,L] - base1[i] + M2[i,L] - base2[i]
        Bp = b_float[:, p]          # (n, n) Bp[L,j] = b[L, p[j]]
        Bp2 = b_float[p, :]         # (n, n) Bp2[j,L] = b[p[j], L]
        M1 = a_float @ Bp.T         # (n,n) M1[i,L]
        M2 = a_float.T @ Bp2        # (n,n) M2[i,L]
        base1 = M1[np.arange(n), p]  # (n,)
        base2 = M2[np.arange(n), p]  # (n,)
        D = M1 - base1[:, None] + M2 - base2[:, None]  # D[i,L] = delta(i,L)
        # disallow staying
        np.fill_diagonal(D, 0)
        return D

    def tabu_search(p0, max_iters, tabu_len, time_limit):
        p = p0.copy()
        cur_cost = _full_cost(a, b, p)
        best_p = p.copy()
        best_cost = cur_cost
        tabu_until = np.zeros(n, dtype=np.int64)  # iteration until which facility i is tabu
        it = 0
        while it < max_iters:
            now = time.time()
            if now > time_limit:
                break
            D = all_deltas(p)
            # choose best non-tabu move; aspiration: allow tabu if it beats best_cost
            Dm = D.copy()
            # mark tabu
            tabu_mask = (tabu_until > it)  # (n,) bool
            # For each i, if tabu, set Dm[i,:] to large unless aspiration
            # Aspiration: a tabu move (i,L) allowed if new cost < best_cost
            # new cost = cur_cost + D[i,L]; allowed if < best_cost
            # We'll handle by: for tabu i, set Dm[i,L] = D[i,L] only if cur_cost+D[i,L] < best_cost, else inf
            for i in range(n):
                if tabu_mask[i]:
                    allowed = (cur_cost + D[i]) < best_cost
                    Dm[i][~allowed] = np.inf
            # pick min
            idx = np.unravel_index(np.argmin(Dm), Dm.shape)
            i, L = idx
            if not np.isfinite(Dm[i, L]):
                break
            delta = D[i, L]
            old = p[i]
            p[i] = L
            cur_cost += delta
            tabu_until[i] = it + tabu_len
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_p = p.copy()
            it += 1
        return best_p, best_cost

    # Multiple restarts with different tabu lengths and max iters
    restarts = 0
    while time.time() < effective_deadline:
        p0 = _initial_p(a, b, rng)
        # vary tabu length per restart
        tabu_len = 10 + (restarts % 5) * 5
        max_iters = 50000  # large; time limit will cut
        bp, bc = tabu_search(p0, max_iters, tabu_len, effective_deadline)
        if best_cost is None or bc < best_cost:
            best_cost = bc
            best_p = bp.copy()
        restarts += 1
        # If we have time, do more restarts; else break
        if time.time() > effective_deadline:
            break

    if best_p is None:
        best_p = _initial_p(a, b, rng)

    return best_p.tolist()
# EVOLVE-BLOCK-END
