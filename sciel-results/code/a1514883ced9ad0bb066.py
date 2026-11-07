"""Cochromatic partition via branch-and-bound over maximal cliques/independent sets."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
DEADLINE = 75.0  # safety margin under 90s wall-clock


def _build_sets():
    """Build all maximal cliques and maximal independent sets as integer bitmasks."""
    adj = [0] * N
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    all_mask = (1 << N) - 1

    def maximal_sets(complement):
        """Bron-Kerbosch on adj if complement=False (cliques), on non-adj if True (independent)."""
        if complement:
            def nb(v):
                return all_mask ^ (1 << v) ^ adj[v]
        else:
            def nb(v):
                return adj[v]

        results = []

        def bronk(R, Pset, X):
            if Pset == 0 and X == 0:
                results.append(R)
                return
            # pivot: choose vertex in Pset|X with max |Pset & nb(u)|
            cand = Pset | X
            best_u = -1
            best_cnt = -1
            tmp = cand
            while tmp:
                lsb = tmp & (-tmp)
                u = lsb.bit_length() - 1
                c = Pset & nb(u)
                if c.bit_count() > best_cnt:
                    best_cnt = c.bit_count()
                    best_u = u
                tmp ^= lsb
            if best_u == -1:
                return
            # iterate over Pset \ nb(best_u)
            candidates = Pset & ~nb(best_u)
            tmp = candidates
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                bronk(R | lsb, Pset & nb(v), X & nb(v))
                Pset ^= lsb
                X |= lsb
                tmp ^= lsb

        bronk(0, all_mask, 0)
        return results

    cliques = maximal_sets(complement=False)
    indeps = maximal_sets(complement=True)
    return cliques, indeps


def _greedy_partition():
    """Greedy first-fit: returns a valid partition as list of lists."""
    es = set()
    for u, v in P['edges']:
        es.add((u, v))
        es.add((v, u))

    def is_clique(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if (lst[i], lst[j]) not in es:
                    return False
        return True

    def is_indep(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if (lst[i], lst[j]) in es:
                    return False
        return True

    parts = []
    for v in range(N):
        placed = False
        for part in parts:
            trial = part + [v]
            if is_clique(trial) or is_indep(trial):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts


def _greedy_partition_v2(candidates, all_mask):
    """Greedy set-cover: repeatedly pick the candidate covering most remaining vertices."""
    rem = all_mask
    chosen = []
    while rem:
        best_c = None
        best_cnt = 0
        for c in candidates:
            if c & ~rem:
                continue
            cnt = (c & rem).bit_count()
            if cnt > best_cnt:
                best_cnt = cnt
                best_c = c
        if best_c is None:
            # fallback: pick any single uncovered vertex
            lsb = rem & (-rem)
            best_c = lsb
            best_cnt = 1
        chosen.append(best_c)
        rem ^= best_c
    return chosen


def construct():
    start = time.time()
    cliques, indeps = _build_sets()

    # Candidate sets: all maximal cliques and maximal independent sets
    candidates = []
    for c in cliques:
        candidates.append(c)
    for c in indeps:
        candidates.append(c)

    # Remove duplicates (a set could be both a clique and independent only if size<=1)
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    candidates = uniq

    all_mask = (1 << N) - 1

    # Sort candidates by size descending for better pruning
    candidates.sort(key=lambda x: -x.bit_count())

    # Greedy initial upper bound (try both strategies, take the better)
    greedy_parts = _greedy_partition()
    greedy2 = _greedy_partition_v2(candidates, all_mask)
    greedy2_parts = []
    for s in greedy2:
        lst = [i for i in range(N) if (s >> i) & 1]
        greedy2_parts.append(lst)

    if len(greedy2_parts) < len(greedy_parts):
        best_k = len(greedy2_parts)
        best_solution = greedy2_parts
    else:
        best_k = len(greedy_parts)
        best_solution = greedy_parts

    # Precompute for each candidate its intersection with every other candidate (for pruning)
    # and a dominance relation: candidate A dominates B if A ⊇ B (A covers at least what B does)
    # We can safely remove dominated candidates (smaller sets contained in larger ones)
    # But only if the larger set is also a valid candidate. Since all are maximal cliques/indeps,
    # no maximal clique is contained in another, and same for independent sets.
    # However, a clique might be contained in an independent set? No, they're different types.
    # Actually a single vertex is both, but we already deduplicated.
    # So no dominance removal possible among maximal sets. Keep all.

    # For faster subset checks, pre-filter candidates by size
    # Group candidates by size for quick access
    max_cand_size = max(c.bit_count() for c in candidates) if candidates else 0

    found_better = False
    best_sets_found = None
    nodes_explored = [0]

    def search(rem_mask, chosen, count, max_count):
        nonlocal found_better, best_sets_found, best_k, best_solution

        if time.time() - start > DEADLINE:
            return False  # timeout

        nodes_explored[0] += 1

        if rem_mask == 0:
            if count < best_k:
                best_k = count
                best_sets_found = list(chosen)
                parts = []
                for s in chosen:
                    lst = [i for i in range(N) if (s >> i) & 1]
                    parts.append(lst)
                best_solution = parts
                found_better = True
            return True

        # Pruning: not enough sets left
        if count + 1 >= max_count:
            return False

        # Lower bound: each remaining set can cover at most max_cand_size vertices
        # Actually more precise: compute max coverage per remaining candidate
        rem_count = rem_mask.bit_count()
        # Find the largest candidate that is a subset of rem_mask
        max_cov = 0
        for c in candidates:
            if c & ~rem_mask:
                continue
            sz = c.bit_count()
            if sz > max_cov:
                max_cov = sz
        if max_cov == 0:
            return False
        min_sets_needed = (rem_count + max_cov - 1) // max_cov
        if count + min_sets_needed >= max_count:
            return False

        # Symmetry breaking: among the remaining vertices, pick the "hardest" one
        # (fewest candidate sets can cover it) and only consider candidates containing it
        # This reduces branching significantly
        # Find vertex in rem_mask with fewest covering candidates
        best_vertex = -1
        best_cnt = N + 1
        tmp = rem_mask
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            cnt = 0
            for c in candidates:
                if c & ~rem_mask:
                    continue
                if (c >> v) & 1:
                    cnt += 1
            if cnt < best_cnt:
                best_cnt = cnt
                best_vertex = v
            if cnt == 0:
                break
            tmp ^= lsb

        if best_vertex == -1 or best_cnt == 0:
            return False  # no candidate can cover this vertex

        # Only branch on candidates that contain best_vertex and are subsets of rem_mask
        for c in candidates:
            if time.time() - start > DEADLINE:
                return False
            if c & ~rem_mask:
                continue
            if c == 0:
                continue
            if not ((c >> best_vertex) & 1):
                continue
            chosen.append(c)
            if search(rem_mask ^ c, chosen, count + 1, max_count):
                chosen.pop()
                return True
            chosen.pop()
        return False

    # Try k = 1, 2, ..., best_k - 1
    for k in range(1, best_k):
        if time.time() - start > DEADLINE:
            break
        if search(all_mask, [], 0, k + 1):
            if found_better:
                break
        else:
            continue

    # Build final partition from best_solution
    result = []
    for part in best_solution:
        if part:
            result.append([int(v) for v in part])

    # Validate: every vertex exactly once
    flat = []
    for part in result:
        flat.extend(part)
    if sorted(flat) != list(range(N)):
        result = [[int(v) for v in p] for p in _greedy_partition()]

    # Ensure types: lists of Python ints
    final = []
    for part in result:
        final.append([int(v) for v in part])

    return final
