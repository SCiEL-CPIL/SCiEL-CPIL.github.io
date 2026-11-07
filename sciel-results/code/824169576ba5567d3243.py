"""Cochromatic partition via branch-and-bound over maximal cliques/independent sets."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}


def construct():
    n = P['n']
    deadline = time.time() + 85.0  # safety margin before 90s kill

    # Build adjacency as bitmask
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    all_mask = (1 << n) - 1

    # A part (set of vertices) is valid if it is a clique or an independent set
    # Clique: for every pair, edge exists. Independent: for every pair, no edge.
    def is_valid_part(mask):
        if mask == 0:
            return True
        # Find first vertex
        v = (mask & (-mask)).bit_length() - 1
        rest = mask & ~(1 << v)
        # Check clique: all vertices in rest must be adjacent to v
        if (adj[v] & rest) == rest:
            # v is adjacent to all in rest; need rest to also be a clique
            if is_valid_part(rest):
                return True
        # Check independent: no vertex in rest is adjacent to v
        if (adj[v] & rest) == 0:
            if is_valid_part(rest):
                return True
        return False

    # Generate all maximal cliques (Bron-Kerbosch with pivot)
    def bron_kerbosch(R, P_mask, X):
        if P_mask == 0 and X == 0:
            yield R
        # Pivot selection
        candidates = P_mask | X
        if candidates:
            pv = (candidates & (-candidates)).bit_length() - 1
            # Candidates not adjacent to pivot
            cand = P_mask & ~adj[pv]
        else:
            cand = P_mask
        while cand:
            v_bit = cand & (-cand)
            v = v_bit.bit_length() - 1
            cand ^= v_bit
            bron_kerbosch_helper(R | v_bit, P_mask & adj[v], X & adj[v])

    def bron_kerbosch_helper(R, P_mask, X):
        if P_mask == 0 and X == 0:
            yield R
            return
        candidates = P_mask | X
        if candidates:
            pv = (candidates & (-candidates)).bit_length() - 1
            cand = P_mask & ~adj[pv]
        else:
            cand = P_mask
        while cand:
            v_bit = cand & (-cand)
            v = v_bit.bit_length() - 1
            cand ^= v_bit
            for result in bron_kerbosch_helper(R | v_bit, P_mask & adj[v], X & adj[v]):
                yield result

    # Generate all maximal independent sets = maximal cliques of complement
    def bron_kerbosch_indep(R, P_mask, X):
        if P_mask == 0 and X == 0:
            yield R
            return
        candidates = P_mask | X
        if candidates:
            pv = (candidates & (-candidates)).bit_length() - 1
            # For independent sets, we need non-neighbors in complement = non-neighbors in original
            cand = P_mask & ~adj[pv]
        else:
            cand = P_mask
        while cand:
            v_bit = cand & (-cand)
            v = v_bit.bit_length() - 1
            cand ^= v_bit
            for result in bron_kerbosch_indep(R | v_bit, P_mask & ~adj[v] & ~((1 << v)), X & ~adj[v] & ~((1 << v))):
                yield result

    # Actually, for independent sets in G, we want cliques in complement of G
    # Complement adjacency: v and w are adjacent in complement iff NOT adjacent in G
    def bron_kerbosch_comp(R, P_mask, X):
        """Find maximal cliques in complement graph = maximal independent sets in G."""
        if P_mask == 0 and X == 0:
            yield R
            return
        candidates = P_mask | X
        if candidates:
            pv = (candidates & (-candidates)).bit_length() - 1
            # In complement, neighbors of pv are those NOT adjacent to pv in G (and not pv itself)
            comp_neighbors = P_mask & ~adj[pv] & ~(1 << pv)
            cand = P_mask & ~comp_neighbors
        else:
            cand = P_mask
        while cand:
            v_bit = cand & (-cand)
            v = v_bit.bit_length() - 1
            cand ^= v_bit
            # In complement, neighbors of v
            comp_neigh_v = P_mask & ~adj[v] & ~(1 << v)
            for result in bron_kerbosch_comp(R | v_bit, comp_neigh_v, X & ~adj[v] & ~(1 << v)):
                yield result

    # Collect all maximal cliques and maximal independent sets
    max_cliques = set()
    for mc in bron_kerbosch(0, all_mask, 0):
        max_cliques.add(mc)

    max_indeps = set()
    for mi in bron_kerbosch_comp(0, all_mask, 0):
        max_indeps.add(mi)

    # Candidate parts: all maximal cliques and maximal independent sets
    # Also include singletons (they're both clique and independent)
    candidates = list(max_cliques | max_indeps)
    # Ensure all singletons are included
    for v in range(n):
        candidates.append(1 << v)

    # Remove duplicates, keep only valid parts
    seen = set()
    valid_candidates = []
    for c in candidates:
        if c not in seen and c != 0:
            seen.add(c)
            valid_candidates.append(c)

    # Sort candidates by size descending (bigger parts first for better pruning)
    valid_candidates.sort(key=lambda m: m.bit_count(), reverse=True)

    # Branch and bound: find minimum number of parts to cover all vertices
    best = [n]  # worst case: n singletons
    best_partition = [1 << v for v in range(n)]  # all singletons

    def solve(remaining, chosen, depth):
        if time.time() > deadline:
            return
        if remaining == 0:
            if depth < best[0]:
                best[0] = depth
                best_partition = list(chosen)
            return
        # Lower bound: at least 1 more part needed
        if depth + 1 >= best[0]:
            return
        # Try each candidate that intersects remaining
        for c in valid_candidates:
            if time.time() > deadline:
                return
            # Only consider candidates that cover at least one uncovered vertex
            if c & remaining:
                # The part c can only include vertices that are actually in remaining
                part = c & remaining
                if part == 0:
                    continue
                # Check if part is still valid (subset of a valid part is valid)
                chosen.append(part)
                solve(remaining & ~part, chosen, depth + 1)
                chosen.pop()

    solve(all_mask, [], 0)

    # Convert bitmask partition to list of vertex lists
    result = []
    for mask in best_partition:
        verts = []
        m = mask
        while m:
            v_bit = m & (-m)
            v = v_bit.bit_length() - 1
            verts.append(v)
            m ^= v_bit
        result.append(verts)

    # Ensure it's a list of lists of Python ints
    return [list(p) for p in result]
