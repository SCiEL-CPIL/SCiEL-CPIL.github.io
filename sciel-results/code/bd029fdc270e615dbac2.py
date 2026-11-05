import time
import random
import itertools
from collections import deque

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin before 120s (evaluator kills at 150s)

def _deadline():
    return time.time() >= _DEADLINE

# Precompute masks for subsets of a given size over n vertices.
# We'll build on the fly per n to keep memory bounded.

def _all_subsets_of_size(n, k):
    """Yield bitmasks for all k-subsets of {0..n-1}."""
    # Use combinations for clarity; n<=22 so fine.
    for comb in itertools.combinations(range(n), k):
        mask = 0
        for v in comb:
            mask |= (1 << v)
        yield mask

def _degrees_in_subset(adj, mask, n):
    """Return list of (vertex, degree_within_mask) for vertices in mask.
    adj[i] is a bitset of neighbors of i (including possibly i? we keep no self loops).
    """
    res = []
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        m ^= lsb
        # degree of v within mask = popcount(adj[v] & mask)
        deg = (adj[v] & mask).bit_count()
        res.append((v, deg))
    return res

def _is_regular_mask(adj, mask):
    """True if induced subgraph on mask is regular (all same degree)."""
    first = -1
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        m ^= lsb
        deg = (adj[v] & mask).bit_count()
        if first == -1:
            first = deg
        elif deg != first:
            return False
    return True

def _has_regular_subgraph(adj, n, min_size=6):
    """Return True if any subset of size >= min_size induces a regular subgraph."""
    # Check sizes from min_size up to n.
    # Optimization: if a subset is regular, all its supersets? Not necessarily.
    # We must check all. But we can early-stop on first found.
    for k in range(min_size, n + 1):
        for mask in _all_subsets_of_size(n, k):
            if _is_regular_mask(adj, mask):
                return True
    return False

def _count_regular_subgraphs(adj, n, min_size=6):
    """Count number of regular induced subgraphs of size >= min_size (for scoring)."""
    cnt = 0
    for k in range(min_size, n + 1):
        for mask in _all_subsets_of_size(n, k):
            if _is_regular_mask(adj, mask):
                cnt += 1
    return cnt

def _random_graph(n, p=0.5, rng=None):
    rng = rng or random
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _valid(adj, n):
    return not _has_regular_subgraph(adj, n, 6)

def _try_local_search(adj, n, rng, max_iters=20000):
    """Local search: flip single edges to eliminate regular subgraphs.
    Returns improved adj or None if no improvement within budget.
    """
    if _deadline():
        return adj
    best = adj[:]
    best_cnt = _count_regular_subgraphs(best, n, 6)
    if best_cnt == 0:
        return best
    # We want to reduce count. Try flips that reduce count.
    # To speed up, evaluate effect of flipping edge (i,j) on the count is expensive;
    # instead do a hill-climb: flip a random edge, if count decreases keep it.
    cur = best[:]
    cur_cnt = best_cnt
    no_improve = 0
    for it in range(max_iters):
        if _deadline():
            break
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        bit = 1 << j
        if cur[i] & bit:
            # remove edge
            cur[i] ^= bit
            cur[j] ^= (1 << i)
        else:
            cur[i] ^= bit
            cur[j] ^= (1 << i)
        new_cnt = _count_regular_subgraphs(cur, n, 6)
        if new_cnt < cur_cnt:
            cur_cnt = new_cnt
            if cur_cnt < best_cnt:
                best = cur[:]
                best_cnt = cur_cnt
            no_improve = 0
        else:
            # revert
            cur[i] ^= bit
            cur[j] ^= (1 << i)
            no_improve += 1
        if no_improve > 500:
            # perturb: flip a few random edges
            for _ in range(5):
                a = rng.randrange(n)
                b = rng.randrange(n)
                if a == b:
                    continue
                bit = 1 << b
                cur[a] ^= bit
                cur[b] ^= (1 << a)
            cur_cnt = _count_regular_subgraphs(cur, n, 6)
            no_improve = 0
    return best

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def construct():
    rng = random.Random(12345)
    best_n = 5
    best_adj = [0] * 5  # empty graph on 5 is trivially valid
    best_edges = []

    # Try to find valid graphs for increasing n.
    # Start from known-good: empty graph on any n<6 is valid (no subset of size>=6).
    # For n>=6, empty graph: the whole set of size n is regular (all degree 0) -> invalid.
    # So we need non-trivial graphs.

    # Strategy: for target n, try many random graphs + local search.
    for target_n in range(6, 23):
        if _deadline():
            break
        found = False
        # Try several random seeds / densities
        for attempt in range(200):
            if _deadline():
                break
            # Vary density
            p = rng.uniform(0.2, 0.8)
            adj = _random_graph(target_n, p, rng)
            if _valid(adj, target_n):
                found = True
                best_n = target_n
                best_adj = adj
                best_edges = _edges_from_adj(adj, target_n)
                break
            else:
                # local search to try to fix
                improved = _try_local_search(adj, target_n, rng, max_iters=2000)
                if _valid(improved, target_n):
                    found = True
                    best_n = target_n
                    best_adj = improved
                    best_edges = _edges_from_adj(improved, target_n)
                    break
        if found:
            # Try to go higher on next loop; but also try to extend this graph by adding vertices.
            # For now, continue to higher target_n.
            continue
        else:
            # Could not find valid graph for this target_n; stop trying higher.
            break

    # Ensure we at least return something valid.
    if _valid(best_adj, best_n):
        return {'n': best_n, 'edges': best_edges}

    # Fallback: empty graph on 5 (always valid)
    return {'n': 5, 'edges': []}
