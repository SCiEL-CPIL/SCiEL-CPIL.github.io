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
    """Greedy first-fit with multiple orderings: returns best valid partition."""
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

    def greedy(order):
        parts = []
        for v in order:
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

    # Try several orderings
    best = greedy(list(range(N)))
    
    # Degree-based ordering (high degree first for cliques, low for independent)
    deg = [0] * N
    for u, v in P['edges']:
        deg[u] += 1
        deg[v] += 1
    best = min(best, greedy(sorted(range(N), key=lambda x: -deg[x])), key=len)
    best = min(best, greedy(sorted(range(N), key=lambda x: deg[x])), key=len)
    
    # Random shuffles
    import random
    for _ in range(50):
        order = list(range(N))
        random.shuffle(order)
        best = min(best, greedy(order), key=len)
    
    return best


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

    # Greedy initial upper bound
    greedy_parts = _greedy_partition()
    best_k = len(greedy_parts)
    best_solution = greedy_parts  # list of lists

    # Pre-filter candidates: only keep those that are subsets of all_mask
    # and have size >= 2 (singletons are always valid but less useful)
    filtered_candidates = [c for c in candidates if c.bit_count() >= 2]
    # Also keep singletons as fallback
    singleton_candidates = [1 << i for i in range(N)]
    all_candidates = filtered_candidates + singleton_candidates

    found_better = False
    best_sets_found = None
    node_count = [0]

    def search(rem_mask, chosen, count, max_count):
        nonlocal found_better, best_sets_found, best_k, best_solution, node_count

        if time.time() - start > DEADLINE:
            return False  # timeout

        node_count[0] += 1

        if rem_mask == 0:
            if count < best_k:
                best_k = count
                best_sets_found = list(chosen)
                # Convert to partition
                parts = []
                for s in chosen:
                    lst = [i for i in range(N) if (s >> i) & 1]
                    parts.append(lst)
                best_solution = parts
                found_better = True
            return True

        if count + 1 >= max_count:
            return False

        # Lower bound: need at least ceil(rem_mask.bit_count() / max_candidate_size) more sets
        rem_size = rem_mask.bit_count()
        max_cand_size = max((c.bit_count() for c in all_candidates if c & ~rem_mask == 0), default=1)
        if max_cand_size > 0:
            lb = (rem_size + max_cand_size - 1) // max_cand_size
            if count + lb >= max_count:
                return False

        # Try each candidate that is subset of rem_mask
        # Sort candidates by size descending for this rem_mask
        valid_cands = [c for c in all_candidates if c & ~rem_mask == 0 and c != 0]
        valid_cands.sort(key=lambda x: -x.bit_count())
        
        for c in valid_cands:
            if time.time() - start > DEADLINE:
                return False
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
            # No solution with k sets, try larger k
            continue

    # Build final partition from best_solution
    # Ensure vertices appear exactly once and parts are nonempty
    result = []
    for part in best_solution:
        if part:
            result.append([int(v) for v in part])

    # Validate: every vertex exactly once
    flat = []
    for part in result:
        flat.extend(part)
    if sorted(flat) != list(range(N)) or len(result) == 0:
        # Fallback to greedy
        result = [[int(v) for v in p] for p in _greedy_partition()]

    # Ensure types: lists of Python ints
    final = []
    for part in result:
        final.append([int(v) for v in part])

    return final
