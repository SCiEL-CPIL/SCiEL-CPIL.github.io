"""Cochromatic partition solver: precompute maximal cliques/ind-sets, then set-partition search."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}


def _build_graph():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return n, adj


def _find_maximal_cliques(adj, n):
    """Bron-Kerbosch to find all maximal cliques."""
    cliques = []

    def bronk(R, P, X):
        if not P and not X:
            if R:
                cliques.append(frozenset(R))
            return
        # Choose pivot from P ∪ X to minimize branching
        candidates = P | X
        if candidates:
            pivot = max(candidates, key=lambda v: len(adj[v] & P))
            candidates = P - adj[pivot]
        else:
            candidates = set()
        for v in list(candidates):
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    bronk(set(), set(range(n)), set())
    return cliques


def _find_maximal_independent_sets(adj, n):
    """Maximal independent sets in G = maximal cliques in complement."""
    comp_adj = [set(range(n)) - adj[i] - {i} for i in range(n)]
    return _find_maximal_cliques(comp_adj, n)


def construct():
    n, adj = _build_graph()
    deadline = time.time() + 85  # safety margin before 90s kill

    neighbors = [set(adj[i]) for i in range(n)]
    non_neighbors = [set(range(n)) - neighbors[i] - {i} for i in range(n)]

    # Precompute all maximal cliques and maximal independent sets
    max_cliques = _find_maximal_cliques(neighbors, n)
    max_indep_sets = _find_maximal_independent_sets(neighbors, n)

    # All valid part candidates: each maximal clique and each maximal independent set
    # A part must be a clique or independent set; any subset of a maximal clique is also a clique,
    # but we only need maximal ones since we can always use a subset if needed.
    # However, for partitioning we need the parts to exactly cover vertices.
    # We'll use all maximal cliques and maximal independent sets as candidate parts.
    # Also include singletons (they are both cliques and independent sets).
    candidate_parts = []
    seen_masks = set()
    for mc in max_cliques:
        mask = 0
        for v in mc:
            mask |= (1 << v)
        if mask not in seen_masks:
            seen_masks.add(mask)
            candidate_parts.append(('clique', mask))
    for mi in max_indep_sets:
        mask = 0
        for v in mi:
            mask |= (1 << v)
        if mask not in seen_masks:
            seen_masks.add(mask)
            candidate_parts.append(('independent', mask))

    # Also add all singletons as candidates (in case no maximal set covers them optimally)
    for v in range(n):
        mask = 1 << v
        if mask not in seen_masks:
            seen_masks.add(mask)
            candidate_parts.append(('clique', mask))

    full_mask = (1 << n) - 1

    # Sort candidates: larger parts first (more coverage per part)
    candidate_parts.sort(key=lambda x: bin(x[1]).count('1'), reverse=True)

    # Greedy upper bound
    def greedy_partition():
        parts = []
        remaining = full_mask
        while remaining:
            # Try to find the largest candidate that fits entirely in remaining
            best_part = None
            best_size = 0
            for ptype, pmask in candidate_parts:
                if (pmask & remaining) == pmask:
                    sz = bin(pmask).count('1')
                    if sz > best_size:
                        best_size = sz
                        best_part = pmask
            if best_part is None:
                # Shouldn't happen since singletons are candidates
                v = (remaining & -remaining).bit_length() - 1
                best_part = 1 << v
            parts.append(best_part)
            remaining &= ~best_part
        return parts

    best_parts = greedy_partition()
    best_count = len(best_parts)

    # Lower bound: maximum clique size in G or in complement
    # Actually: we need at least max_clique_size parts if we use only independent sets,
    # and at least max_indep_size parts if we use only cliques.
    # But since we can mix, the lower bound is max(max_clique_in_G, max_clique_in_complement)
    # which is max(size of largest clique in G, size of largest independent set in G)
    max_clique_size = max(len(mc) for mc in max_cliques) if max_cliques else 1
    max_indep_size = max(len(mi) for mi in max_indep_sets) if max_indep_sets else 1
    lower_bound = max(max_clique_size, max_indep_size)

    # Actually, a better lower bound: the cochromatic number is at least
    # the maximum of (chromatic number of G, chromatic number of complement)
    # But computing chromatic number is hard. Use simple bounds.
    # For now, use: at least ceil(n / max_part_size) where max_part_size is the largest candidate
    max_candidate_size = max(bin(p[1]).count('1') for p in candidate_parts) if candidate_parts else 1
    lb1 = (n + max_candidate_size - 1) // max_candidate_size
    lower_bound = max(lower_bound, lb1)

    if lower_bound >= best_count:
        # Can't do better
        return [list(v for v in range(n) if (pmask >> v) & 1) for pmask in best_parts]

    # Set-partitioning search: choose a subset of candidate_parts whose masks are disjoint
    # and whose union is full_mask, minimizing the number of parts.
    # Use DFS with pruning.

    # Precompute: for each candidate, its mask and size
    cand_masks = [pmask for _, pmask in candidate_parts]
    cand_sizes = [bin(pmask).count('1') for pmask in cand_masks]

    # For efficiency, precompute which candidates are compatible (disjoint)
    # n_cands = len(cand_masks)
    # compatible[i][j] = (cand_masks[i] & cand_masks[j]) == 0
    # This is O(n_cands^2) which is fine for ~100 candidates

    n_cands = len(cand_masks)

    # Search: try to find a partition with k parts for k from lower_bound to best_count - 1
    def search_partition(k, remaining_mask, used_candidates, current_parts):
        """Try to cover remaining_mask with exactly k - len(current_parts) more parts."""
        nonlocal best_parts, best_count

        if time.time() > deadline:
            return None

        parts_needed = k - len(current_parts)
        if parts_needed == 0:
            if remaining_mask == 0:
                return current_parts[:]
            return None

        if remaining_mask == 0:
            return None  # Still need parts but nothing to cover

        # Pruning: if parts_needed is 1, remaining must be exactly one candidate
        if parts_needed == 1:
            for i in range(n_cands):
                if i not in used_candidates and cand_masks[i] == remaining_mask:
                    return current_parts + [cand_masks[i]]
            return None

        # Pruning: if parts_needed is 2, remaining must be union of two disjoint candidates
        if parts_needed == 2:
            for i in range(n_cands):
                if i in used_candidates:
                    continue
                m1 = cand_masks[i]
                if (m1 & remaining_mask) != m1:
                    continue
                rem2 = remaining_mask ^ m1
                if rem2 == 0:
                    continue
                for j in range(n_cands):
                    if j in used_candidates or j == i:
                        continue
                    if cand_masks[j] == rem2:
                        return current_parts + [m1, rem2]
            return None

        # General case: pick a vertex from remaining, try to cover it with some candidate
        # Choose vertex with fewest candidate options (most constrained)
        # For each vertex in remaining, count how many candidates can cover it
        rem_verts = [v for v in range(n) if (remaining_mask >> v) & 1]

        # Find the vertex with fewest covering candidates
        best_v = None
        best_count_opts = n_cands + 1
        for v in rem_verts:
            v_bit = 1 << v
            count_opts = 0
            for i in range(n_cands):
                if i in used_candidates:
                    continue
                if (cand_masks[i] & v_bit) and (cand_masks[i] & remaining_mask) == cand_masks[i]:
                    count_opts += 1
            if count_opts < best_count_opts:
                best_count_opts = count_opts
                best_v = v
            if best_count_opts <= 1:
                break

        if best_v is None:
            return None

        v_bit = 1 << best_v

        # Try each candidate that covers best_v and is a subset of remaining
        for i in range(n_cands):
            if i in used_candidates:
                continue
            m = cand_masks[i]
            if not (m & v_bit):
                continue
            if (m & remaining_mask) != m:
                continue

            new_remaining = remaining_mask ^ m
            new_used = used_candidates | {i}
            new_parts = current_parts + [m]

            result = search_partition(k, new_remaining, new_used, new_parts)
            if result is not None:
                return result

        return None

    # Try to find a partition with fewer parts than best_count
    for k in range(max(lower_bound, 1), best_count):
        result = search_partition(k, full_mask, set(), [])
        if result is not None:
            best_parts = result
            best_count = k
            break

    # If we found a better solution, update lower bound check
    if best_count <= lower_bound:
        pass  # Optimal

    # Validate the result
    def validate(parts):
        seen = 0
        for pmask in parts:
            if pmask & seen:
                return False
            seen |= pmask
            # Check if it's a clique or independent set
            verts = [v for v in range(n) if (pmask >> v) & 1]
            if len(verts) <= 1:
                continue
            is_cl = all(verts[j] in neighbors[verts[i]] for i in range(len(verts)) for j in range(i + 1, len(verts)))
            is_ind = all(verts[j] not in neighbors[verts[i]] for i in range(len(verts)) for j in range(i + 1, len(verts)))
            if not (is_cl or is_ind):
                return False
        if seen != full_mask:
            return False
        return True

    if not validate(best_parts):
        # Fallback to greedy
        best_parts = greedy_partition()

    # Convert to list of lists
    result = []
    for pmask in best_parts:
        part = [v for v in range(n) if (pmask >> v) & 1]
        part.sort()
        result.append(part)

    return result

# EVOLVE-BLOCK-END
