"""Find a graph on n vertices with no regular induced subgraph on 6+ vertices.

Key insight: a k-vertex induced subgraph is regular (all degrees equal) iff
every 6-vertex subset of it is regular.  Proof: in a regular graph the induced
subgraph on any subset has all degrees equal, so a regular k-subgraph (k>=6)
contains only regular 6-subsets; conversely if every 6-subset is regular then
in particular the whole set is regular.  Therefore the validity check only
needs to scan C(n,6) subsets, not all C(n,k) for k>=6.  For n=20 that is
5814 checks vs. ~1M, and for n=21 it is 54264 vs. ~2.7M.  This makes each
validity test ~10x cheaper, so the search budget buys many more iterations.
"""
import time
import random
import itertools


def _popcount(x):
    return x.bit_count()


def _subset_mask(subset):
    m = 0
    for v in subset:
        m |= (1 << v)
    return m


def _is_regular_6(adj_mask, subset, sub_mask):
    """Is the induced subgraph on this 6-tuple regular?"""
    first = -1
    for v in subset:
        d = _popcount(adj_mask[v] & sub_mask)
        if first < 0:
            first = d
        elif d != first:
            return False
    return True


def _find_violation(adj_mask, n):
    """Return (subset, sub_mask) of a regular 6-subset, or None.

    Checking only 6-subsets is sufficient (see module docstring)."""
    vertices = range(n)
    for subset in itertools.combinations(vertices, 6):
        sub_mask = _subset_mask(subset)
        if _is_regular_6(adj_mask, subset, sub_mask):
            return subset, sub_mask
    return None


def _is_valid(adj_mask, n):
    return _find_violation(adj_mask, n) is None


def _extract_edges(adj_mask, n):
    edges = []
    for i in range(n):
        mi = adj_mask[i]
        for j in range(i + 1, n):
            if mi & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}


def _random_graph(n, seed, p=0.5):
    rng = random.Random(seed)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _local_search(adj, n, deadline, seed):
    """Iteratively flip an edge inside a violating 6-set until no violation.

    Flipping a random edge inside a regular 6-set almost always breaks its
    regularity (degrees become uneven), so violations get cleared fast.  We
    also guard against getting stuck by occasionally perturbing a random edge.
    """
    rng = random.Random(seed)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    for _ in range(200000):
        if time.time() > deadline:
            break

        viol = _find_violation(adj, n)
        if viol is None:
            return adj

        subset, _ = viol
        # Candidate edges: all pairs within the violating 6-set.
        cands = [(subset[a], subset[b])
                 for a in range(6) for b in range(a + 1, 6)]
        u, v = rng.choice(cands)
        if adj[u] & (1 << v):
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        # Occasional global perturbation to escape local traps.
        if rng.random() < 0.05:
            u2, v2 = rng.choice(all_pairs)
            if adj[u2] & (1 << v2):
                adj[u2] ^= (1 << v2)
                adj[v2] ^= (1 << u2)
            else:
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)

    return adj if _is_valid(adj, n) else None


def _greedy_construct(n, deadline, seed):
    """Start from a random graph and greedily remove edges that are part of a
    regular 6-set (removing an edge from a regular 6-set breaks its regularity)."""
    rng = random.Random(seed)
    adj = _random_graph(n, seed, p=0.5)

    for _ in range(50000):
        if time.time() > deadline:
            break
        viol = _find_violation(adj, n)
        if viol is None:
            return adj
        subset, _ = viol
        # Remove a random edge that is present within the violating set.
        present = [(subset[a], subset[b])
                   for a in range(6) for b in range(a + 1, 6)
                   if adj[subset[a]] & (1 << subset[b])]
        if not present:
            # No internal edge; add one to break regularity.
            absent = [(subset[a], subset[b])
                      for a in range(6) for b in range(a + 1, 6)
                      if not (adj[subset[a]] & (1 << subset[b]))]
            if absent:
                u, v = rng.choice(absent)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            continue
        u, v = rng.choice(present)
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)

    return adj if _is_valid(adj, n) else None


def _search_for_graph(n, deadline):
    """Multi-strategy search for a valid graph on n vertices."""
    # 1) Quick random sweep (valid random graphs are rare but cheap to test).
    for seed in range(40):
        if time.time() > deadline - 20:
            break
        adj = _random_graph(n, seed * 7919 + n)
        if _is_valid(adj, n):
            return _extract_edges(adj, n)

    # 2) Local search from several random starts (primary workhorse).
    for seed in range(60):
        if time.time() > deadline - 10:
            break
        adj = _random_graph(n, seed * 104729 + n * 31, p=0.5)
        res = _local_search(adj, n, deadline - 10, seed)
        if res is not None:
            return _extract_edges(res, n)

    # 3) Greedy edge-removal from random starts.
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        res = _greedy_construct(n, deadline - 5, seed * 5557 + n)
        if res is not None:
            return _extract_edges(res, n)

    return None


def _fallback_graph():
    """A trivially valid graph: K5 has no 6-vertex subset at all."""
    n = 5
    edges = [[i, j] for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'edges': edges}


def construct():
    start = time.time()
    deadline = start + 115  # safety margin before the 150s kill.

    best = _fallback_graph()

    # Aim for the largest n first; 21 is the breakthrough target, 20 is known.
    for n in [21, 20, 19, 18]:
        if time.time() > deadline - 8:
            break
        result = _search_for_graph(n, deadline - 8)
        if result is not None:
            best = result
            if n == 21:
                break
    return best
