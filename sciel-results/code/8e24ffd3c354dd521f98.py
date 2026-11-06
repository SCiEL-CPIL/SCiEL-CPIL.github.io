"""Integer sum-product extremal sets: max-weight-clique over smooth-number graph."""
import time
import random
import math
from functools import lru_cache

DEADLINE = 55.0  # wall-clock safety margin vs 60s/90s kill
N = 20
MAXV = 10000


def _sumset_size(A):
    s = set()
    for i, a in enumerate(A):
        for j in range(i, len(A)):
            s.add(a + A[j])
    return len(s)


def _prodset_size(A):
    s = set()
    for i, a in enumerate(A):
        for j in range(i, len(A)):
            s.add(a * A[j])
    return len(s)


def _objective(A):
    return max(_sumset_size(A), _prodset_size(A))


def _smoothness_score(x):
    """Higher = more small-prime factors (collision-friendly)."""
    n = x
    score = 0
    p = 2
    while p * p <= n:
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            # weight smaller primes more heavily
            score += e * (10 - min(p, 10))
        p += 1 if p == 2 else 2
    if n > 1:
        score += (10 - min(n, 10))
    return score


def _build_pool():
    """Candidate pool: numbers with high smoothness score, plus some structure."""
    scores = {}
    for x in range(1, MAXV + 1):
        scores[x] = _smoothness_score(x)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    # Take top candidates; include 1 and 2 explicitly
    pool = [1, 2]
    for x, s in ranked:
        if x not in pool:
            pool.append(x)
        if len(pool) >= 400:
            break
    # ensure distinct, positive, <= MAXV
    pool = sorted(set(x for x in pool if 1 <= x <= MAXV))
    return pool


def _pair_features(a, b, pool_idx, sum_counts, prod_counts):
    """Collision density for pair (a,b) with existing set represented by counts."""
    # We'll compute marginal contribution if both added to a current set.
    # For clique heuristic, precompute pairwise 'edge weight' based on
    # how often a+b and a*b collide with other pool pairs.
    pass


def _precompute_edge_weights(pool):
    """
    For each pair (i,j) in pool, compute a weight = number of other pool elements
    k such that pool[i]+pool[j] == pool[k]+pool[l] for some l, OR
    pool[i]*pool[j] == pool[k]*pool[l] for some l.
    Higher weight = more collision potential = more attractive for small sets.
    We approximate with: for sum s = a+b, count distinct pairs in pool producing s;
    for product p = a*b, count distinct pairs in pool producing p.
    Edge weight = (sum_collision_count + prod_collision_count) with a cap.
    """
    n = len(pool)
    # Build sum -> list of (i,j) with i<=j
    sum_map = {}
    prod_map = {}
    for i in range(n):
        a = pool[i]
        for j in range(i, n):
            b = pool[j]
            s = a + b
            p = a * b
            if p <= 10**12:  # safety
                prod_map.setdefault(p, []).append((i, j))
            sum_map.setdefault(s, []).append((i, j))
    # Count collisions: for each pair, how many pairs share the same sum/product
    edge_w = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = pool[i] + pool[j]
            p = pool[i] * pool[j]
            sc = len(sum_map.get(s, []))
            pc = len(prod_map.get(p, [])) if p <= 10**12 else 1
            w = sc + pc
            edge_w[i][j] = w
            edge_w[j][i] = w
    return edge_w, sum_map, prod_map


def _clique_branch_and_bound(pool, edge_w, k, time_limit):
    """
    Greedy + local search to find a k-subset maximizing total pairwise edge weight
    (proxy for collision density). Returns the subset as list of values.
    """
    n = len(pool)
    start = time.time()
    best_subset = None
    best_score = -1

    # Greedy construction: repeatedly pick node with highest sum of edge weights to current set
    def greedy_seed():
        chosen = []
        remaining = set(range(n))
        for _ in range(k):
            if not remaining:
                break
            if not chosen:
                # pick node with highest degree (sum of edge weights)
                best = max(remaining, key=lambda i: sum(edge_w[i][j] for j in remaining if j != i))
            else:
                best = max(remaining, key=lambda i: sum(edge_w[i][j] for j in chosen))
            chosen.append(best)
            remaining.discard(best)
        return chosen

    def subset_score(sub):
        s = 0
        for a in range(len(sub)):
            for b in range(a + 1, len(sub)):
                s += edge_w[sub[a]][sub[b]]
        return s

    # Multiple restarts with different random seeds
    random.seed(42)
    for restart in range(200):
        if time.time() > time_limit:
            break
        if restart == 0:
            sub = greedy_seed()
        else:
            # random subset then greedy improvement
            sub = random.sample(range(n), k)
        # Local search: swap to improve edge weight
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            cur = subset_score(sub)
            sub_set = set(sub)
            # try all swaps
            best_swap = None
            best_new = cur
            for i in range(k):
                for j in range(n):
                    if j in sub_set:
                        continue
                    # compute delta
                    old = 0
                    for t in range(k):
                        if t != i:
                            old += edge_w[sub[i]][sub[t]]
                    new = 0
                    for t in range(k):
                        if t != i:
                            new += edge_w[j][sub[t]]
                    delta = new - old
                    if delta > best_new - cur:
                        best_new = cur + delta
                        best_swap = (i, j)
            if best_swap:
                i, j = best_swap
                sub[i] = j
                improved = True

        sc = subset_score(sub)
        if sc > best_score:
            best_score = sc
            best_subset = sub[:]
            # objective check
            vals = [pool[i] for i in best_subset]
            # keep for later

    if best_subset is None:
        best_subset = greedy_seed()
    return [pool[i] for i in best_subset]


def _local_search_on_objective(A, pool, time_limit):
    """Improve A by swaps to minimize max(|A+A|,|A*A|)."""
    A = list(A)
    best = _objective(A)
    pool_set = set(pool)
    rng = random.Random(123)
    start = time.time()
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        cur = _objective(A)
        A_set = set(A)
        # try each position, each candidate swap
        for i in range(len(A)):
            if time.time() > time_limit:
                break
            # sample a few candidates for speed
            candidates = rng.sample(pool, min(60, len(pool)))
            for c in candidates:
                if c in A_set:
                    continue
                A[i] = c
                new = _objective(A)
                if new < cur:
                    cur = new
                    improved = True
                    break
                A[i] = A[i]  # revert (already same)
            # restore if not improved at this i
            # (we already reverted)
        if cur < best:
            best = cur
    return A, best


def construct():
    start = time.time()
    deadline = start + DEADLINE

    pool = _build_pool()
    # Precompute edge weights (can be slow for 400 nodes: 400^2/2 = 80k pairs, fine)
    edge_w, sum_map, prod_map = _precompute_edge_weights(pool)

    # Phase 1: clique-style selection via edge-weight local search
    t1 = min(deadline - 15.0, start + 20.0)
    A = _clique_branch_and_bound(pool, edge_w, N, t1)

    # Phase 2: local search on true objective
    t2 = deadline - 5.0
    A, obj = _local_search_on_objective(A, pool, t2)

    # Phase 3: random restarts on objective if time remains
    if time.time() < deadline - 3.0:
        best_A = A[:]
        best_obj = obj
        rng = random.Random(7)
        while time.time() < deadline - 3.0:
            # random start from pool, biased toward smooth
            cand = rng.sample(pool, N)
            A2, obj2 = _local_search_on_objective(cand, pool, deadline - 3.0)
            if obj2 < best_obj:
                best_obj = obj2
                best_A = A2[:]

    # Final safety: ensure distinct, positive, <= MAXV, length N
    best_A = list(dict.fromkeys(best_A))  # dedupe preserving order
    if len(best_A) < N:
        # fill from pool
        for x in pool:
            if x not in best_A:
                best_A.append(x)
            if len(best_A) >= N:
                break
    best_A = best_A[:N]
    return best_A


if __name__ == "__main__":
    res = construct()
    print(res)
    print("obj:", _objective(res))
