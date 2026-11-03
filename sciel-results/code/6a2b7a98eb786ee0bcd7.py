# EVOLVE-BLOCK-START
import time
import numpy as np


def _qap_cost(a, b, p):
    # exact integer cost
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j, rowsum_a, rowsum_b):
    # delta for swapping positions i and j in permutation p (p is array of facility->location)
    # p[x] = location of facility x
    li = p[i]; lj = p[j]
    # current contribution of i and j with everything
    # We'll compute using row sums for the linear part and pair term.
    # cost = sum_i ra[i]*rb[p[i]] + 2*sum_{i<j} a[i,j]*b[p[i],p[j]]
    # For swap of i,j, only terms involving i or j change.
    # Linear part: ra[i]*rb[li]+ra[j]*rb[lj] -> ra[i]*rb[lj]+ra[j]*rb[li]
    lin_old = rowsum_a[i]*rowsum_b[li] + rowsum_a[j]*rowsum_b[lj]
    lin_new = rowsum_a[i]*rowsum_b[lj] + rowsum_a[j]*rowsum_b[li]
    # Pair part: for every k != i,j, a[i,k]*b[li,p[k]] + a[j,k]*b[lj,p[k]]
    # -> a[i,k]*b[lj,p[k]] + a[j,k]*b[li,p[k]]
    # plus the i-j pair: a[i,j]*b[li,lj] + a[j,i]*b[lj,li] -> a[i,j]*b[lj,li]+a[j,i]*b[li,lj]
    # Since b symmetric, b[li,lj]=b[lj,li].
    k = np.ones(len(a), dtype=bool); k[i] = False; k[j] = False
    pk = p[k]
    # a[i,k] and a[j,k]
    aik = a[i, k]; ajk = a[j, k]
    b_lipk = b[li, pk]; b_ljpk = b[lj, pk]
    pair_old = np.dot(aik, b_lipk) + np.dot(ajk, b_ljpk) + a[i,j]*b[li,lj] + a[j,i]*b[lj,li]
    pair_new = np.dot(aik, b_ljpk) + np.dot(ajk, b_lipk) + a[i,j]*b[lj,li] + a[j,i]*b[li,lj]
    return (lin_new - lin_old) + 2*(pair_new - pair_old)


def _local_search(a, b, p, deadline, rng):
    n = len(a)
    rowsum_a = a.sum(axis=1)
    rowsum_b = b.sum(axis=1)
    p = p.astype(np.int64, copy=True)
    best_cost = _qap_cost(a, b, p)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # vectorized best-swap search over pairs
        # We'll scan in a shuffled order to avoid cycling; do a full pass, take best improving swap.
        best_d = 0
        best_pair = None
        # To keep it fast, evaluate swaps in batches
        idx = np.arange(n)
        # iterate over i
        for i in range(n):
            if time.time() >= deadline:
                break
            j = idx[i+1:]
            if len(j) == 0:
                continue
            li = p[i]
            lj = p[j]
            # vectorized over j
            lin_old = rowsum_a[i]*rowsum_b[li] + rowsum_a[j]*rowsum_b[lj]
            lin_new = rowsum_a[i]*rowsum_b[lj] + rowsum_a[j]*rowsum_b[li]
            # pair_old/new: for each j, sum over k != i,j
            # a[i,k]*b[li,p[k]] etc. Precompute per i:
            # We need for each j: sum_{k!=i,j} a[i,k]*b[li,p[k]] and a[j,k]*b[lj,p[k]]
            # Use full dot then subtract the k=j term.
            # precompute for this i: ai = a[i,:]; bi_li = b[li,p]; bi_lj = b[lj,p]
            ai = a[i, :]
            bi_li = b[li, p]
            bi_lj = b[lj, p]
            # sum over all k of ai[k]*bi_li[k]
            tot_i = np.dot(ai, bi_li)
            # subtract k=i and k=j terms
            sub_i = ai[i]*bi_li[i] + ai[j]*bi_li[j]
            s_i = tot_i - sub_i
            # for each j: sum_{k!=i,j} a[j,k]*b[lj,p[k]]
            # = dot(a[j,:], b[lj,p]) - a[j,i]*b[lj,p[i]] - a[j,j]*b[lj,p[j]]
            aj = a[j, :]
            tot_j = np.dot(aj, bi_lj)
            sub_j = aj[i]*bi_lj[i] + aj[j]*bi_lj[j]
            s_j = tot_j - sub_j
            # also need the i-j direct pair terms:
            # pair_old includes a[i,j]*b[li,lj] + a[j,i]*b[lj,li]
            # pair_new includes a[i,j]*b[lj,li] + a[j,i]*b[li,lj]
            # b symmetric so b[li,lj]=b[lj,li]; the i-j direct term is unchanged? 
            # old: a[i,j]*b[li,lj] + a[j,i]*b[lj,li]
            # new: a[i,j]*b[lj,li] + a[j,i]*b[li,lj]
            # = a[i,j]*b[li,lj] + a[j,i]*b[li,lj] (since symmetric)
            # old = a[i,j]*b[li,lj] + a[j,i]*b[li,lj]
            # So direct i-j term unchanged. Good, cancels.
            pair_old = s_i + s_j
            pair_new = (tot_i - ai[i]*bi_lj[i] - ai[j]*bi_lj[j]) + (tot_j - aj[i]*bi_li[i] - aj[j]*bi_li[j])
            d = (lin_new - lin_old) + 2*(pair_new - pair_old)
            # find best (most negative) d
            imin = int(np.argmin(d))
            if d[imin] < best_d:
                best_d = int(d[imin])
                best_pair = (i, int(j[imin]))
        if best_pair is not None:
            i, j = best_pair
            p[i], p[j] = p[j], p[i]
            best_cost += best_d
            improved = True
    return p, best_cost


def _greedy_init(a, b, rng):
    n = len(a)
    # Sort facilities by total flow descending; assign to locations by shortest remaining distance.
    # Build a mapping: place facility in order, choose location minimizing distance to already placed.
    order = np.argsort(-a.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    placed_loc = set()
    # distance from a location to the set of placed locations = min over placed
    # We'll compute incrementally.
    dist_to_placed = np.full(n, np.inf)
    for fac in order:
        # choose location with smallest dist_to_placed (tie: smallest total distance)
        # for first, choose smallest row sum of b
        if len(placed_loc) == 0:
            cand = np.argmin(b.sum(axis=1))
        else:
            cand = np.argmin(dist_to_placed)
        p[fac] = cand
        placed_loc.add(cand)
        # update dist_to_placed for unplaced locations
        for loc in range(n):
            if loc not in placed_loc:
                d = b[loc, cand]
                if d < dist_to_placed[loc]:
                    dist_to_placed[loc] = d
    return p


def _rank_init(a, b):
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(len(a), dtype=np.int64)
    p[heavy] = central
    return p


def solve(a, b):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s
    n = len(a)
    rng = np.random.default_rng(12345)

    # candidate initial permutations
    p_greedy = _greedy_init(a, b, rng)
    p_rank = _rank_init(a, b)

    best_p = None
    best_cost = None

    def consider(p):
        nonlocal best_p, best_cost
        c = _qap_cost(a, b, p)
        if best_cost is None or c < best_cost:
            best_cost = c
            best_p = p.copy()
        return p, c

    # polish each start
    for start in [p_greedy, p_rank]:
        if time.time() >= deadline:
            break
        p, c = consider(start)
        p, c = _local_search(a, b, p, deadline, rng)
        if c < best_cost:
            best_cost = c
            best_p = p.copy()

    # randomized restarts if time remains
    while time.time() < deadline - 5.0:
        # random perturbation of best_p
        p = best_p.copy()
        # do a few random swaps
        for _ in range(3):
            i, j = rng.choice(n, size=2, replace=False)
            p[i], p[j] = p[j], p[i]
        p, c = _local_search(a, b, p, deadline, rng)
        if c < best_cost:
            best_cost = c
            best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
