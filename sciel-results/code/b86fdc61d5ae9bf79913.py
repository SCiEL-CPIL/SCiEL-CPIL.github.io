"""Stronger QAP solver for tho150: fast incremental cost + time-budgeted local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_vectors(a, b):
    """Row sums of a (flow) and b (distance) used for O(n) delta updates."""
    row_a = a.sum(axis=1)
    row_b = b.sum(axis=1)
    return row_a, row_b


def _cost(p, a, b, row_a, row_b):
    """Exact cost sum_{i,j} a[i,j]*b[p[i],p[j]]."""
    # sum_i (a[i,:] * b[p[i],:]).sum()
    # = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # Compute per-row: for each i, sum_j a[i,j]*b[p[i], p[j]]
    # = sum_i ( a[i,:] @ b[p[i], :] )
    # Vectorize: for row i, we need a[i,:] dot b[p[i],:]
    # = sum_j a[i,j]*b[p[i],j]
    # Group by location: cost = sum_loc ( b[loc,:] @ (sum over i with p[i]=loc of a[i,:]^T) )
    # Simpler: build matrix M where M[i,j] = a[i,j], and for each i compute dot with b[p[i]]
    # Use: cost = sum_i ( a[i,:] * b[p[i],:] ).sum()
    # = sum_i ( a[i,:] * b[p[i],:] ).sum()
    # Efficient: cost = np.einsum('ij,ij->', a, b[np.arange(len(a)), :])
    n = len(a)
    idx = np.arange(n)
    return np.einsum('ij,ij->', a, b[p[idx], :])


def solve(a, b):
    n = len(a)
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 120s
    row_a, row_b = _build_vectors(a, b)

    # ---- Initial solution: rank matching (heavy flow -> central/low-distance) ----
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    # Local improvement pass to get a decent start
    def try_improve(p, iters, swap_only=False):
        """First-improvement local search with swap (and relocate) moves."""
        best_cost = _cost(p, a, b, row_a, row_b)
        for _ in range(iters):
            if time.time() - t0 > DEADLINE:
                break
            improved = False
            # Swap moves: for pair (i,j) with i<j, delta of swapping locations
            # We'll scan pairs in a fixed order; accept first improving swap.
            # To keep O(n^2) per pass manageable, scan all pairs but break early on first improvement.
            # Use incremental delta formula for swap (i,j):
            # Let x=p[i], y=p[j].
            # delta = 2 * sum_{k != i,j} ( a[i,k]*b[x,p[k]] + a[j,k]*b[y,p[k]] - a[i,k]*b[y,p[k]] - a[j,k]*b[x,p[k]] )
            #       + a[i,j]*(b[x,y]-b[y,x]) + a[j,i]*(b[y,x]-b[x,y])  ... but b symmetric? not guaranteed.
            # Simpler robust approach: recompute cost after swap only when promising.
            # Given n=150, n^2=22500 pairs; recomputing full cost each is too slow.
            # Use delta formula (works for general non-symmetric b):
            # After swapping, new cost - old cost:
            # Only rows i and j change their location.
            # Old contributions involving i or j (as the "i" index) or as location x or y.
            # We derive:
            # Let S = set of all k.
            # cost = sum_{u,v} a[u,v] * b[p[u], p[v]]
            # Terms affected by swapping p[i]<->p[j]:
            #  - terms where u==i or v==i or u==j or v==j (since p[i] or p[j] changes)
            # Compute delta = new - old over those terms.
            # old terms with u=i: sum_v a[i,v]*b[x, p[v]]
            # new terms with u=i: sum_v a[i,v]*b[y, p[v]]  (since p[i] becomes y)
            #   but note for v==j, p[v] is x in old, y in new? No: p[j] changes too.
            # This is getting complex; handle the (i,j) and (j,i) cross terms explicitly.
            #
            # Let's define for a fixed swap of locations x=p[i], y=p[j]:
            # For u not in {i,j}: p[u] unchanged.
            #   old: a[u,i]*b[p[u],x] + a[i,u]*b[x,p[u]] + a[u,j]*b[p[u],y] + a[j,u]*b[y,p[u]]
            #   new: a[u,i]*b[p[u],y] + a[i,u]*b[y,p[u]] + a[u,j]*b[p[u],x] + a[j,u]*b[x,p[u]]
            # For u==i (row i):
            #   old: sum_v a[i,v]*b[x,p[v]]
            #   new: sum_v a[i,v]*b[y,p'[v]]  where p'[j]=x, p'[i]=y, p'[v]=p[v] for v!=i,j
            # For u==j (row j):
            #   old: sum_v a[j,v]*b[y,p[v]]
            #   new: sum_v a[j,v]*b[x,p'[v]]
            # This is intricate. Given complexity and n=150, I'll use a simpler but still
            # efficient delta for the common symmetric-distance case, and fall back to
            # recomputation if needed. For robustness with possibly non-symmetric b,
            # I'll compute delta by direct formula for the affected terms using arrays.
            #
            # Practical approach: precompute for speed, but given time, use a moderate
            # strategy: try swaps in random order, recompute cost only for a limited set,
            # OR use the standard QAP swap delta (valid when b is symmetric).
            # QAPLIB distances are symmetric, so b[x,y]==b[y,x]. Use that.

            # Standard swap delta for symmetric b:
            # delta(i,j) = 2 * [ sum_{k != i,j} ( a[i,k]*(b[x,p[k]] - b[y,p[k]]) + a[j,k]*(b[y,p[k]] - b[x,p[k]]) ) ]
            #            + a[i,j]*(b[x,y] - b[y,x]) + a[j,i]*(b[y,x] - b[x,y])  -> 0 if symmetric
            #            + (a[i,j] + a[j,i]) * (b[x,y] - b[x,y]) -> 0
            # Wait, need to be careful. Let me use the well-known formula:
            # For symmetric b, swapping i (at x) and j (at y):
            # delta = 2 * sum_{k != i,j} [ a[i,k]*(b[x,p[k]] - b[y,p[k]]) + a[j,k]*(b[y,p[k]] - b[x,p[k]]) ]
            #       + (a[i,j] + a[j,i]) * (b[x,y] - b[y,x])  # = 0 if symmetric
            #       + a[i,j]*(b[x,y] - b[x,y]) ... 
            # Actually the cross terms: originally a[i,j]*b[x,y] and a[j,i]*b[y,x].
            # After swap: a[i,j]*b[y,x] and a[j,i]*b[x,y].
            # If symmetric b: b[x,y]=b[y,x], so cross terms unchanged. delta_cross = 0.
            # So delta = 2 * sum_{k != i,j} [ a[i,k]*(b[x,p[k]] - b[y,p[k]]) + a[j,k]*(b[y,p[k]] - b[x,p[k]]) ]
            #          = 2 * sum_{k != i,j} [ (a[i,k] - a[j,k]) * (b[x,p[k]] - b[y,p[k]]) ]
            # Good, that's clean for symmetric b.

            # Precompute per-location distance-to-x and to-y? That's O(n) each, and we have O(n^2) pairs -> O(n^3). Too slow.
            # Instead, for each i, precompute D_i[loc] = b[loc, :] ? No.
            # Let's restructure: for fixed i and location x=p[i], the vector b[x, :] is fixed.
            # For each i, define f_i[k] = b[x, p[k]] - b[y, p[k]] for the candidate j's y.
            # But y depends on j. So for each i, and for each candidate j, we need b[y=p[j], p[k]] for all k.
            # This is b[p[j], p[k]] = (b @ onehot)... 
            # Precompute Bp[k] = b[p[k], :] (row of b at location p[k]). Then b[p[j], p[k]] = Bp[j][p[k]].
            # And b[x, p[k]] = Bp[i][p[k]].
            # So (b[x,p[k]] - b[y,p[k]]) = (Bp[i][p[k]] - Bp[j][p[k]]) = (Bp[i] - Bp[j])[p[k]].
            # Then sum_k (a[i,k]-a[j,k])*(Bp[i]-Bp[j])[p[k]] = dot( (a[i,:]-a[j,:]), (Bp[i]-Bp[j])[p] )
            # But (a[i,:]-a[j,:]) and (Bp[i]-Bp[j])[p] are both length n vectors (indexed by k).
            # dot over k: sum_k (a[i,k]-a[j,k]) * (Bp[i]-Bp[j])[p[k]].
            # Precompute Bp = b[p, :] shape (n,n). Then (Bp[i]-Bp[j])[p] = Bp[i][p] - Bp[j][p].
            # So delta = 2 * dot( a[i,:]-a[j,:], Bp[i][p] - Bp[j][p] )  ... but this includes k=i and k=j terms.
            # We need to exclude k=i and k=j. Let's compute full dot then subtract k=i and k=j contributions.
            # Full: S = dot(a[i]-a[j], Bp[i][p]-Bp[j][p])
            #   term k=i: (a[i,i]-a[j,i]) * (Bp[i][p[i]] - Bp[j][p[i]]) = (a[i,i]-a[j,i])*(b[x,x]-b[y,x])
            #   term k=j: (a[i,j]-a[j,j]) * (Bp[i][p[j]] - Bp[j][p[j]]) = (a[i,j]-a[j,j])*(b[x,y]-b[y,y])
            # delta = 2*(S - term_i - term_j)  [for symmetric b, cross term 0]
            # This is O(n) per pair if we precompute Bp. Bp[i][p] is a vector; dot is O(n). For O(n^2) pairs that's O(n^3)=3.4e6*... n=150 -> n^3=3.375e6 per pass? No: n^2 pairs * n = n^3 = 3.4e6 ops per pass. That's fine!
            # Actually n^3 = 150^3 = 3,375,000 dot-element ops per full pass. Very fast.

            # Let's implement: precompute Bp = b[p, :] (n x n). For each pass, recompute Bp (O(n^2)).
            # Then for pairs, delta = 2*( dot(a[i]-a[j], Bp[i][p]-Bp[j][p]) - ti - tj )
            # We can vectorize the pair loop somewhat, but a Python loop over n^2/2 ~ 11250 pairs with O(n) dot each = 11250*150 = 1.7e6 numpy scalar ops... but each dot is a numpy call (overhead). Better to vectorize.
            # Vectorize: for fixed i, compute for all j the delta.
            # Let's do: for each i, compute diff_a = a[i,:] - a[:,i]? No, a[j,:] for each j.
            # We want for each j: dot(a[i]-a[j], Bp[i][p]-Bp[j][p]).
            # = dot(a[i], Bp[i][p]) - dot(a[i], Bp[j][p]) - dot(a[j], Bp[i][p]) + dot(a[j], Bp[j][p])
            # Precompute:
            #   c1 = dot(a[i], Bp[i][p])  (scalar)
            #   c2[j] = dot(a[i], Bp[j][p])  for all j  -> a[i,:] @ (Bp[:,?]) ... Bp[j][p] is row j of Bp indexed by p.
            #     Bp[j][p] = b[p[j], p] (vector over k). So c2[j] = a[i,:] dot b[p[j], p] = (a[i,:] @ b) [p[j]]? 
            #     a[i,:] @ b = vector v_i where v_i[loc] = sum_k a[i,k]*b[k,loc]. Then v_i[p[j]] = sum_k a[i,k]*b[k,p[j]] = sum_k a[i,k]*b[p[j],k] (symmetric) = dot(a[i], b[p[j],:]) = dot(a[i], Bp[j]) ... but Bp[j] = b[p[j],:], and we need dot with b[p[j], p] (only positions p). 
            #     Hmm, Bp[j][p] is not full Bp[j], it's Bp[j] permuted by p. So c2[j] = sum_k a[i,k]*Bp[j][p[k]] = sum_k a[i,k]*b[p[j],p[k]].
            #     This equals (a[i,:] * b[p[j], p]) summed. Not directly a simple mat-vec.
            # This is getting complicated to fully vectorize. Given n=150, a Python loop over i (150) and inner vectorized over j (150) might be OK.
            # For fixed i: 
            #   Ai = a[i,:] (n,)
            #   Bi_p = Bp[i][p] (n,)  # = b[x, p]
            #   For each j: need dot(Ai - a[j,:], Bi_p - Bp[j][p])
            #   = dot(Ai, Bi_p) - dot(Ai, Bp[j][p]) - dot(a[j,:], Bi_p) + dot(a[j,:], Bp[j][p])
            #   Precompute for fixed i:
            #     s1 = dot(Ai, Bi_p)
            #     s2[j] = dot(Ai, Bp[j][p])  -> Ai @ (Bp[:,?])... Bp[j][p] for all j: that's a matrix M2 where M2[j,k]=Bp[j][p[k]]=b[p[j],p[k]]. M2 = b[p[:,None], p[None,:]] = b[np.ix_(p,p)]. So M2 is (n,n): M2[j,k]=b[p[j],p[k]]. Then s2 = Ai @ M2  (n,) -> s2[j]=sum_k a[i,k]*b[p[j],p[k]].
            #     s3[j] = dot(a[j,:], Bi_p) = a[j,:] @ Bi_p. Bi_p is fixed vector. s3 = a @ Bi_p (n,).
            #     s4[j] = dot(a[j,:], Bp[j][p]) = sum_k a[j,k]*b[p[j],p[k]] = row j of (a @ M2)? a @ M2 has [j,k]=sum_l a[j,l]*M2[l,k]. Not it.
            #     s4[j] = (a @ M2)[j, j]? No. s4[j] = sum_k a[j,k]*M2[j,k] = row-wise dot of a and M2 at row j. = np.einsum('jk,jk->j', a, M2).
            #   So delta_full[j] = s1 - s2[j] - s3[j] + s4[j]
            #   Then subtract term_i and term_j:
            #     term_i = (a[i,i]-a[j,i])*(b[x,x]-b[y,x])  -- depends on j! Can't precompute per i alone.
            #     term_j = (a[i,j]-a[j,j])*(b[x,y]-b[y,y])  -- depends on j.
            #   Hmm, the excluded terms depend on j. So we can't fully vectorize the exclusion without per-j work.
            #   Alternative: include k=i,j in the sum and then the formula is slightly off. Let me reconsider: maybe just include all k and accept the formula as-is? The standard formula excludes k=i,j because those are the "cross" terms handled separately (and are 0 for symmetric). If I include them in the sum, I'd be double counting or miscounting.
            #   Let me just do the per-j exclusion in a vectorized way:
            #     term_i[j] = (a[i,i]-a[j,i])*(bxx - bxy)  where bxx=b[x,x], bxy=b[x,y]
            #     term_j[j] = (a[i,j]-a[j,j])*(bxy - byy)  where byy=b[y,y]
            #     bxx, byy are scalars (per i: bxx=b[x,x] fixed; byy=b[y,y]=b[p[j],p[j]] depends on j).
            #     bxy = b[x, p[j]] depends on j.
            #   So term_i[j] and term_j[j] are vectorizable:
            #     bxx = b[x,x] (scalar)
            #     bxy_vec = b[x, p]  (n,)  # b[x, p[j]] for each j
            #     byy_vec = b[p, p]  (n,)  # b[p[j],p[j]] for each j
            #     term_i_vec = (a[i,i] - a[:,i]) * (bxx - bxy_vec)   # a[j,i] for each j
            #     term_j_vec = (a[i,:] - np.diag(a)) * (bxy_vec - byy_vec)  # a[i,j] and a[j,j]
            #     delta[j] = 2*(delta_full[j] - term_i_vec[j] - term_j_vec[j])
            #   This is fully vectorizable per i! 
            #   Let me implement this.

            Bp = b[p, :]  # (n,n): Bp[j,k] = b[p[j], k]
            M2 = b[np.ix_(p, p)]  # (n,n): M2[j,k] = b[p[j], p[k]]
            diag_a = np.diag(a)
            for i in range(n):
                if time.time() - t0 > DEADLINE:
                    break
                x = p[i]
                Ai = a[i, :]
                Bi_p = Bp[i, p]  # b[x, p[k]] for k
                bxx = b[x, x]
                bxy_vec = b[x, p]  # b[x, p[j]]
                byy_vec = b[p, p]  # b[p[j], p[j]]
                s1 = np.dot(Ai, Bi_p)
                s2 = Ai @ M2  # s2[j] = sum_k a[i,k]*b[p[j],p[k]]
                s3 = a @ Bi_p  # s3[j] = sum_k a[j,k]*b[x,p[k]]
                s4 = np.einsum('jk,jk->j', a, M2)  # s4[j] = sum_k a[j,k]*b[p[j],p[k]]
                delta_full = s1 - s2 - s3 + s4
                term_i_vec = (a[i, i] - a[:, i]) * (bxx - bxy_vec)
                term_j_vec = (a[i, :] - diag_a) * (bxy_vec - byy_vec)
                delta = 2.0 * (delta_full - term_i_vec - term_j_vec)
                # Exclude i itself (delta[i] is meaningless)
                delta[i] = np.inf
                # Find best j
                j = int(np.argmin(delta))
                if delta[j] < -1e-9:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    best_cost += delta[j]
                    improved = True
                    # Update Bp and M2 for changed rows i, j
                    Bp[i] = b[p[i], :]
                    Bp[j] = b[p[j], :]
                    M2[i] = b[p[i], p]
                    M2[j] = b[p[j], p]
                    if not swap_only:
                        pass
                    continue  # restart scan from i=0? or continue. Continue for speed.
            if not improved:
                # Try relocate moves? For simplicity, do a second scan with relocate.
                # Relocate: move facility i from x to an empty location y (but all locations are used, so relocate = swap with some j, already covered). 
                # Actually with a permutation, relocate = swap. So swap-only is full 1-swap neighborhood.
                # To escape local optima, do random restarts or 2-opt. For time, do a few random perturbations.
                break
        return best_cost, p

    # Initial improvement
    best_cost, p = try_improve(p, iters=200, swap_only=True)

    # Random restarts / perturbations to escape local optima
    rng = np.random.default_rng(12345)
    best_p = p.copy()
    best_best_cost = best_cost

    while time.time() - t0 < DEADLINE:
        # Perturb: random swap of a few pairs
        cur = best_p.copy()
        n_swaps = rng.integers(2, 8)
        for _ in range(n_swaps):
            i1, j1 = rng.integers(0, n, size=2)
            if i1 != j1:
                cur[i1], cur[j1] = cur[j1], cur[i1]
        cur_cost, cur = try_improve(cur, iters=100, swap_only=True)
        if cur_cost < best_best_cost:
            best_best_cost = cur_cost
            best_p = cur.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
