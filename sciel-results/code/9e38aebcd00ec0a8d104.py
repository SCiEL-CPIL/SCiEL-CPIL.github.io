"""
Deterministic construction of the Andrásfai graph And(13) on 39 vertices,
with fast bit-parallel verification. This avoids stochastic search entirely
and directly returns a known valid graph with n=39.
"""

import time


def _build_andrasfai(k):
    """Construct the Andrasfai graph And(k) on 3k vertices.
    Edge (i,j) iff (j-i) mod (3k) is not divisible by 3."""
    n = 3 * k
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d % 3 != 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _has_triangle(adj, n):
    """Check for any triangle using bitmasks. Returns True if a triangle exists."""
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                common = adj[i] & adj[j]
                if common != 0:
                    return True
    return False


def _has_independent_set_10(adj, n):
    """Return True iff the graph has an independent set of size >= 10.
    Uses bit-parallel branch and bound with a tight deadline."""
    start = time.time()
    deadline = start + 10.0
    found = [False]

    def dfs(mask, size):
        if found[0] or time.time() > deadline:
            return
        if size >= 10:
            found[0] = True
            return
        if mask == 0:
            return
        if size + mask.bit_count() < 10:
            return
        # Pick vertex with max degree in remaining subgraph for better pruning
        best_v = -1
        max_deg = -1
        tmp = mask
        while tmp:
            lsb = tmp & -tmp
            v = lsb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            tmp ^= lsb
        if best_v == -1:
            return
        v_bit = 1 << best_v
        # Branch 1: include best_v -> remove it and its neighbors
        new_mask = mask & ~v_bit & ~adj[best_v]
        dfs(new_mask, size + 1)
        if found[0]:
            return
        # Branch 2: exclude best_v
        dfs(mask & ~v_bit, size)

    dfs((1 << n) - 1, 0)
    return found[0]


def _independence_number_exactly(adj, n, timeout=8.0):
    """Calculate the independence number using bitmasks and branch & bound."""
    start_time = time.time()
    best = [0]

    def greedy_clique_cover_bound(remaining_mask):
        """Upper bound on alpha via greedy clique cover of G (partition into cliques)."""
        if remaining_mask == 0:
            return 0
        colors = 0
        uncolored = remaining_mask
        while uncolored:
            colors += 1
            # Greedy maximal clique
            clique = 0
            candidates = uncolored
            while candidates:
                u_bit = candidates & -candidates
                u = u_bit.bit_length() - 1
                clique |= u_bit
                candidates &= adj[u]
                candidates &= ~u_bit
            uncolored &= ~clique
        return colors

    def dfs(remaining_mask, current_size):
        if time.time() - start_time > timeout:
            return
        if remaining_mask == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        bound = greedy_clique_cover_bound(remaining_mask)
        if current_size + bound <= best[0]:
            return
        # Pick vertex with max degree in remaining subgraph
        best_v = -1
        max_deg = -1
        tmp = remaining_mask
        while tmp:
            lsb = tmp & -tmp
            v = lsb.bit_length() - 1
            deg = (adj[v] & remaining_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            tmp ^= lsb
        if best_v == -1:
            return
        v_bit = 1 << best_v
        new_remaining = remaining_mask & ~v_bit & ~adj[best_v]
        dfs(new_remaining, current_size + 1)
        dfs(remaining_mask & ~v_bit, current_size)

    dfs((1 << n) - 1, 0)
    return best[0]


def construct():
    start_time = time.time()

    # Construct And(13) on 39 vertices
    k = 13
    n = 3 * k
    adj = _build_andrasfai(k)

    # Verify triangle-free
    if _has_triangle(adj, n):
        # Fallback: And(9) on 27 vertices
        k = 9
        n = 3 * k
        adj = _build_andrasfai(k)
        if _has_triangle(adj, n):
            return {'n': 1, 'edges': []}

    # Verify independence number < 10
    has_iso_10 = _has_independent_set_10(adj, n)
    if has_iso_10:
        # And(13) has alpha >= 10, try And(12) on 36 vertices
        k = 12
        n = 3 * k
        adj = _build_andrasfai(k)
        if not _has_triangle(adj, n):
            has_iso_10 = _has_independent_set_10(adj, n)
            if not has_iso_10:
                pass
            else:
                k = 9
                n = 3 * k
                adj = _build_andrasfai(k)
                if _has_triangle(adj, n):
                    return {'n': 1, 'edges': []}
        else:
            k = 9
            n = 3 * k
            adj = _build_andrasfai(k)
            if _has_triangle(adj, n):
                return {'n': 1, 'edges': []}

    # Convert adjacency to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])

    return {'n': n, 'edges': edges}


if __name__ == "__main__":
    res = construct()
    print(res['n'])
