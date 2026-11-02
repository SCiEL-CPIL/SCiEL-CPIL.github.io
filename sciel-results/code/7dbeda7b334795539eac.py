"""Randomized k-regular graph construction with exact girth certification."""
import random
from collections import deque

# EVOLVE-BLOCK-START
def _is_simple_and_regular(halfedge_pairs, n, k):
    """Build adjacency from half-edge pairs; check simple + regular. Return adj or None."""
    adj = [[] for _ in range(n)]
    for v in range(n):
        for i in range(k):
            a = v * k + i
            b = halfedge_pairs[a]
            if b >= n * k:
                return None  # invalid index
            u = b // k
            if u == v:
                return None  # loop
            adj[v].append(u)
    # check regularity and no multi-edges
    for v in range(n):
        if len(adj[v]) != k:
            return None
        if len(set(adj[v])) != k:
            return None
    # check symmetry
    for v in range(n):
        for u in adj[v]:
            if v not in adj[u]:
                return None
    return adj


def _girth_exceeds(adj, n, g):
    """Return True iff girth > g-1, i.e. no cycle shorter than g.
    Use non-backtracking walks from every vertex."""
    # We look for any cycle of length < g.
    # Non-backtracking BFS: state = (prev, curr). Start from each directed edge.
    # A cycle of length L is detected when we return to the start vertex after L steps
    # without backtracking.
    # Simpler: BFS on non-backtracking paths up to depth g-1; if we ever reach a state
    # (prev, curr) where prev == start_vertex and depth >= 1, we have a cycle of length depth.
    # But we need cycles of length < g.
    #
    # Approach: for each starting vertex s, do BFS over non-backtracking paths starting
    # from each neighbor. Track (prev, curr) -> (start, depth). If we reach (prev=s, curr)
    # with depth < g, cycle found.
    #
    # To be efficient, we do a global BFS over the non-backtracking graph of directed edges.
    # State: (prev, curr). Transition: from (prev, curr) go to (curr, next) for next in adj[curr]
    # if next != prev.
    # We want to detect if there's a non-backtracking walk from (s, v) back to (u, s) for
    # some u != v (so the last step isn't backtracking) with total length < g.
    #
    # Equivalent: girth = min over all directed edges (s,v) of the shortest non-backtracking
    # path from (v, s) ... hmm let me think differently.
    #
    # Standard method: girth is the minimum length of a cycle. A cycle of length L corresponds
    # to a non-backtracking closed walk of length L. We can find the girth by doing BFS from
    # each vertex over non-backtracking paths and checking if we can return to the start
    # in fewer than g steps.
    #
    # For each vertex s, BFS where state is (prev, curr), initialized with (None, s) -> but
    # non-backtracking from s: first step to any neighbor. So states are (prev, curr) with
    # prev a neighbor of s (or None for start). We track distance. If we reach (prev, s)
    # with prev != None and distance < g, we have a cycle.
    #
    # Actually simpler: initialize queue with all (s, v) for v in adj[s], depth 1.
    # Then expand: from (prev, curr), for each next in adj[curr] with next != prev:
    #   new state (curr, next), depth+1.
    #   If next == s and depth+1 < g: cycle found (length depth+1).
    #   If state (curr, next) not visited at this depth, mark and enqueue.
    #
    # We need to be careful: we want the SHORTEST cycle, so BFS by depth works.
    #
    # To avoid O(n) BFS per vertex being too slow, we can do a single BFS over all
    # directed edges simultaneously, but we need to track the start vertex.
    #
    # Given n up to ~1116 and g=9, depth up to 8, branching ~k-1=4, so per start vertex
    # we explore at most sum_{i=0}^{7} 4^i ~ 16384 states. For 1116 vertices that's ~18M
    # which might be slow in Python. Let's optimize: stop as soon as we find any cycle < g.
    #
    # Better: do BFS from all vertices simultaneously? No, we need to track start.
    #
    # Alternative: compute all-pairs shortest paths on the non-backtracking directed edge
    # graph, but that's complex.
    #
    # Let's just do per-vertex BFS and hope it's fast enough. For g=9, depth 8, k=5:
    # max states per vertex ~ 5 * 4^7 ~ 32768. For 1116 vertices ~ 36M. In Python this
    # could take ~30-60s. Risky but let's try with optimizations.
    #
    # Optimization: use a visited set per start vertex, but share across starts? No.
    #
    # Let's try a different approach: since g is small (9), we can enumerate all non-backtracking
    # walks of length up to g-1 from each vertex and check for cycles. Use a set to avoid
    # re-exploring the same (prev, curr) state within a single start vertex.

    for s in range(n):
        # BFS from s
        visited = set()
        queue = deque()
        for v in adj[s]:
            state = (s, v)
            visited.add(state)
            queue.append((state, 1))
        while queue:
            (prev, curr), depth = queue.popleft()
            if depth >= g - 1:
                continue
            for nxt in adj[curr]:
                if nxt == prev:
                    continue
                if nxt == s:
                    # cycle of length depth+1
                    if depth + 1 < g:
                        return False
                    continue
                new_state = (curr, nxt)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, depth + 1))
    return True


def _try_random_regular(n, k, rng):
    """Generate a random simple k-regular graph on n vertices. Return adj or None."""
    max_attempts = 200
    for _ in range(max_attempts):
        halfedges = list(range(n * k))
        rng.shuffle(halfedges)
        pairs = [None] * (n * k)
        for i in range(0, n * k, 2):
            a = halfedges[i]
            b = halfedges[i + 1]
            pairs[a] = b
            pairs[b] = a
        adj = _is_simple_and_regular(pairs, n, k)
        if adj is not None:
            return adj
    return None


def construct(k, g):
    """Construct a k-regular graph with girth at least g, minimizing n."""
    rng = random.Random(42)
    
    # Moore bound for k-regular graph of girth g:
    # For even g = 2m: n >= 1 + k * sum_{i=0}^{m-1} (k-1)^i
    # For odd g = 2m+1: n >= 1 + k * sum_{i=0}^{m-1} (k-1)^i + k*(k-1)^m
    if g % 2 == 0:
        m = g // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(m))
    else:
        m = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(m)) + k * (k - 1) ** m
    
    # We need n such that a k-regular graph exists: n*k must be even, n >= k+1
    # Start from Moore bound and go up
    n = moore
    # Adjust n so that n*k is even
    if (n * k) % 2 != 0:
        n += 1
    # n must be at least k+1
    n = max(n, k + 1)
    
    max_n = 2000  # cap to avoid infinite search
    attempts_per_n = 50
    
    while n <= max_n:
        for attempt in range(attempts_per_n):
            adj = _try_random_regular(n, k, rng)
            if adj is not None:
                if _girth_exceeds(adj, n, g):
                    # Build edge list
                    edges = []
                    for v in range(n):
                        for u in adj[v]:
                            if v < u:
                                edges.append((v, u))
                    return n, edges
        # Try next n
        n += 1
        if (n * k) % 2 != 0:
            n += 1
    
    # Fallback: return a trivial construction if we couldn't find one
    # This shouldn't happen for reasonable k, g
    raise RuntimeError("Failed to construct graph")
# EVOLVE-BLOCK-END
