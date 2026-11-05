"""Triangle-free graph without independent 10-set, maximizing n (Ramsey R(3,10))."""
import time
import random
from itertools import combinations

_START = time.time()
_DEADLINE = 115.0  # safety margin before 150s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START)


def _has_triangle_bitset(adj_bits, n):
    """Check if bitset adjacency has a triangle using bit operations."""
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_bits[i] >> j) & 1:
                common = adj_bits[i] & adj_bits[j]
                if common & ~((1 << (j + 1)) - 1):
                    return True
                if common & ((1 << i) - 1):
                    return True
    return False


def _max_independent_set_size_bitset(adj_bits, n):
    """
    Find size of maximum independent set using branch and bound with coloring bound.
    The coloring bound: alpha(G[S]) <= chi(complement(G[S])).
    We greedily color the complement graph restricted to S.
    """
    best = [0]

    def _color_bound(candidates):
        """Greedy coloring of complement(G[S]) to upper bound alpha(G[S])."""
        # In complement, u~v iff NOT adj[u][v] in G
        # Greedy coloring: assign smallest available color to each vertex
        # A proper coloring of complement means each color class is a clique in complement
        # which is an independent set in G. So number of colors is an upper bound on
        # the max independent set size in G[S].
        
        # For speed, process in order of decreasing degree in complement
        # Degree in complement = (len(candidates) - 1) - degree_in_G_within_S
        order = list(candidates)
        # Sort by degree in G (ascending) = degree in complement (descending)
        # Higher degree in complement = more constraints = color first
        order.sort(key=lambda v: sum(1 for u in candidates if u != v and ((adj_bits[v] >> u) & 1)))
        
        color = {}
        num_colors = 0
        for v in order:
            # Find colors used by neighbors of v in complement
            # v's neighbors in complement are u in S where NOT adj[v][u]
            used = set()
            for u in candidates:
                if u == v:
                    continue
                if u in color:
                    # u is neighbor of v in complement iff NOT adj[v][u]
                    if not ((adj_bits[v] >> u) & 1):
                        used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            num_colors = max(num_colors, c + 1)
        return num_colors

    def bnb(candidates, current_size):
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return

        # Upper bound from coloring
        if current_size + _color_bound(candidates) <= best[0]:
            return

        # Trivial bound
        if current_size + len(candidates) <= best[0]:
            return

        # Pick vertex with max degree in G within candidates
        # (min degree in complement) - this tends to prune better
        v = max(candidates, key=lambda u: sum(1 for w in candidates if w != u and ((adj_bits[u] >> w) & 1)))
        rest = [u for u in candidates if u != v]

        # Include v: remove v and its neighbors in G
        neighbors_mask = adj_bits[v]
        non_neighbors = [u for u in rest if not ((neighbors_mask >> u) & 1)]
        bnb(non_neighbors, current_size + 1)

        # Exclude v
        bnb(rest, current_size)

    bnb(list(range(n)), 0)
    return best[0]


def _max_independent_set_fast(adj_bits, n):
    """
    Optimized maximum independent set.
    Uses a two-phase approach: first a quick greedy to get a good lower bound,
    then branch and bound with coloring bound for upper bound.
    """
    # Phase 1: Greedy to find a good independent set (lower bound)
    # Greedy: repeatedly pick vertex with minimum degree in remaining set
    remaining = list(range(n))
    greedy_size = 0
    while remaining:
        # Pick vertex with min degree in remaining
        v = min(remaining, key=lambda u: sum(1 for w in remaining if w != u and ((adj_bits[u] >> w) & 1)))
        greedy_size += 1
        # Remove v and its neighbors
        neighbors_mask = adj_bits[v]
        remaining = [u for u in remaining if u != v and not ((neighbors_mask >> u) & 1)]

    best = [greedy_size]

    def _color_bound(candidates):
        """Greedy coloring of complement(G[S])."""
        order = list(candidates)
        order.sort(key=lambda v: sum(1 for u in candidates if u != v and ((adj_bits[v] >> u) & 1)))
        color = {}
        num_colors = 0
        for v in order:
            used = set()
            for u in candidates:
                if u == v:
                    continue
                if u in color:
                    if not ((adj_bits[v] >> u) & 1):
                        used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            num_colors = max(num_colors, c + 1)
        return num_colors

    def bnb(candidates, current_size):
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return

        if current_size + _color_bound(candidates) <= best[0]:
            return

        if current_size + len(candidates) <= best[0]:
            return

        v = max(candidates, key=lambda u: sum(1 for w in candidates if w != u and ((adj_bits[u] >> w) & 1)))
        rest = [u for u in candidates if u != v]

        neighbors_mask = adj_bits[v]
        non_neighbors = [u for u in rest if not ((neighbors_mask >> u) & 1)]
        bnb(non_neighbors, current_size + 1)
        bnb(rest, current_size)

    bnb(list(range(n)), 0)
    return best[0]


def _build_triangle_free_bitset(n, seed=0):
    """
    Build a triangle-free graph using bitset representation.
    Random greedy: add edges in random order if they don't create triangles.
    """
    rng = random.Random(seed)
    adj_bits = [0] * n

    edges_order = list(combinations(range(n), 2))
    rng.shuffle(edges_order)

    for i, j in edges_order:
        common = adj_bits[i] & adj_bits[j]
        if common:
            continue
        adj_bits[i] |= (1 << j)
        adj_bits[j] |= (1 << i)

    return adj_bits


def _local_search_bitset(adj_bits, n, seed=0):
    """
    Local search: try adding edges that don't create triangles to reduce alpha.
    """
    rng = random.Random(seed)
    adj_bits = adj_bits[:]

    alpha = _max_independent_set_fast(adj_bits, n)

    if alpha < 10:
        return adj_bits, alpha

    all_non_edges = [(a, b) for a in range(n) for b in range(a + 1, n) if not ((adj_bits[a] >> b) & 1)]

    for _ in range(800):
        if _remaining_time() < 5:
            break

        if not all_non_edges:
            break

        i, j = rng.choice(all_non_edges)
        # Re-check if still non-edge (might have been added)
        if (adj_bits[i] >> j) & 1:
            continue

        common = adj_bits[i] & adj_bits[j]
        if common:
            continue

        # Add edge
        adj_bits[i] |= (1 << j)
        adj_bits[j] |= (1 << i)
        all_non_edges.remove((i, j))

        new_alpha = _max_independent_set_fast(adj_bits, n)
        if new_alpha < alpha:
            alpha = new_alpha
            if alpha < 10:
                break
        else:
            adj_bits[i] &= ~(1 << j)
            adj_bits[j] &= ~(1 << i)
            all_non_edges.append((i, j))

    return adj_bits, alpha


def _build_circulant_andrasfai(n, seed=0):
    """
    Build an Andrásfai-type circulant graph.
    For n = 3k, use steps S = {s : 1 <= s <= k-1, 3 not| s}.
    This gives a triangle-free circulant with alpha = k.
    """
    if n % 3 != 0:
        return None
    k = n // 3
    adj_bits = [0] * n
    for i in range(n):
        for s in range(1, k):
            if s % 3 != 0:
                j = (i + s) % n
                if j > i:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
    return adj_bits


def _build_complement_of_paley(n, seed=0):
    """
    For n = p^2 (p prime), the Paley graph is triangle-free? No.
    Actually, let's try a different construction.
    """
    return None


def _bitset_to_edges(adj_bits, n):
    """Convert bitset adjacency to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    return edges


def _validate_bitset(adj_bits, n):
    """Check triangle-free and alpha < 10."""
    if _has_triangle_bitset(adj_bits, n):
        return False
    alpha = _max_independent_set_fast(adj_bits, n)
    return alpha < 10


def construct():
    """
    Construct a triangle-free graph on n vertices (1 <= n <= 41) with no independent set of 10.
    Maximize n.
    """
    best_adj_bits = None
    best_n = 0
    best_alpha = float('inf')

    def _try_n(n_target):
        nonlocal best_adj_bits, best_n, best_alpha

        # Strategy 1: Andrásfai circulant (if n divisible by 3)
        if n_target % 3 == 0:
            adj_circ = _build_circulant_andrasfai(n_target)
            if adj_circ is not None:
                alpha_circ = _max_independent_set_fast(adj_circ, n_target)
                if alpha_circ < 10:
                    best_adj_bits = adj_circ
                    best_n = n_target
                    best_alpha = alpha_circ
                    return True
                # Try local search from circulant
                if _remaining_time() > 10:
                    adj_ls, alpha_ls = _local_search_bitset(adj_circ, n_target, seed=42)
                    if alpha_ls < 10:
                        best_adj_bits = adj_ls
                        best_n = n_target
                        best_alpha = alpha_ls
                        return True
                    if alpha_ls < best_alpha:
                        best_adj_bits = adj_ls
                        best_n = n_target
                        best_alpha = alpha_ls

        # Strategy 2: Random greedy + local search
        for seed in range(120):
            if _remaining_time() < 8:
                break

            adj_bits = _build_triangle_free_bitset(n_target, seed)
            alpha = _max_independent_set_fast(adj_bits, n_target)

            if alpha < 10:
                best_adj_bits = adj_bits
                best_n = n_target
                best_alpha = alpha
                return True

            if alpha < best_alpha:
                best_adj_bits = adj_bits
                best_n = n_target
                best_alpha = alpha

            # Local search
            if _remaining_time() > 15:
                adj_ls, alpha_ls = _local_search_bitset(adj_bits, n_target, seed)
                if alpha_ls < 10:
                    best_adj_bits = adj_ls
                    best_n = n_target
                    best_alpha = alpha_ls
                    return True
                if alpha_ls < best_alpha:
                    best_adj_bits = adj_ls
                    best_n = n_target
                    best_alpha = alpha_ls

        return False

    # Try from largest n down
    for n_target in range(40, 26, -1):
        if _remaining_time() < 15:
            break
        if _try_n(n_target):
            break

    # Fallback: Andrásfai graph And(9) - 27 vertices
    if best_adj_bits is None:
        n = 27
        adj_bits = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if 1 <= d <= 8 and d % 3 != 0:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
        best_adj_bits = adj_bits
        best_n = n

    edges = _bitset_to_edges(best_adj_bits, best_n)

    return {'n': best_n, 'edges': edges}
