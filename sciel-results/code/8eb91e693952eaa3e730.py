import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s hard kill

    N = 256
    elements = list(range(1, N + 1))

    # Build conflict graph as Python int bitmasks
    # conflict[e] is a bitmask of all vertices that conflict with e
    conflict = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a] |= (1 << b)
                conflict[b] |= (1 << a)

    def popcount(x):
        return bin(x).count('1')

    def is_valid(mask):
        """Check if mask is an independent set."""
        for e in range(1, N + 1):
            if mask & (1 << e):
                if conflict[e] & mask & ~(1 << e):
                    return False
        return True

    def can_add(e, mask):
        return (conflict[e] & mask) == 0

    def local_search_add(mask):
        """Repeatedly try to add any element that fits."""
        improved = True
        while improved:
            improved = False
            for e in range(1, N + 1):
                if not (mask & (1 << e)) and can_add(e, mask):
                    mask |= (1 << e)
                    improved = True
        return mask

    def greedy_build(order, start_mask=0):
        """Greedily add elements in order to start_mask."""
        result = start_mask
        for e in order:
            if can_add(e, result):
                result |= (1 << e)
        return result

    def branch_and_bound(candidates, current_size, best_size, time_limit):
        """
        Branch and bound for maximum independent set.
        candidates: bitmask of remaining candidate vertices
        current_size: size of independent set chosen so far
        Returns (best_mask, best_size) found.
        """
        best_mask = [0]
        best_size_val = [current_size]
        # Initialize with current solution if we have one
        # We'll track the best found globally via the closure

        def bound(cands):
            """Upper bound on additional vertices we can add from cands."""
            # Simple bound: popcount of cands
            return popcount(cands)

        def bbsolve(cands, chosen_mask, chosen_size):
            nonlocal best_size_val, best_mask
            if time.time() > time_limit:
                return
            if cands == 0:
                if chosen_size > best_size_val[0]:
                    best_size_val[0] = chosen_size
                    best_mask[0] = chosen_mask
                return

            # Upper bound: if even adding all candidates can't beat best, prune
            if chosen_size + popcount(cands) <= best_size_val[0]:
                return

            # Pick a vertex to branch on
            # Heuristic: pick vertex with minimum degree in the induced subgraph
            # (i.e., fewest conflicts within candidates)
            cands_list = []
            temp = cands
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                cands_list.append(v)
                temp ^= lb

            if not cands_list:
                return

            # Choose pivot: vertex with min conflicts in candidates
            pivot = None
            min_conf = N + 1
            for v in cands_list:
                conf_in_cands = popcount(conflict[v] & cands)
                if conf_in_cands < min_conf:
                    min_conf = conf_in_cands
                    pivot = v
                    if min_conf == 0:
                        break

            # Branch 1: include pivot
            new_cands = cands & ~(1 << pivot) & ~conflict[pivot]
            new_mask = chosen_mask | (1 << pivot)
            new_size = chosen_size + 1
            if new_size + popcount(new_cands) > best_size_val[0]:
                bbsolve(new_cands, new_mask, new_size)

            # Branch 2: exclude pivot
            new_cands2 = cands & ~(1 << pivot)
            if chosen_size + popcount(new_cands2) > best_size_val[0]:
                bbsolve(new_cands2, chosen_mask, chosen_size)

        bbsolve(candidates, 0, 0)
        return best_mask[0], best_size_val[0]

    def two_for_one_swap(mask, max_time):
        """Try to improve by removing one element and adding two others."""
        current_mask = mask
        current_size = popcount(current_mask)

        while time.time() < max_time:
            improved = False

            # Try pure additions
            for e in range(1, N + 1):
                if not (current_mask & (1 << e)) and can_add(e, current_mask):
                    current_mask |= (1 << e)
                    current_size += 1
                    improved = True

            # Try 2-for-1 swaps
            if not improved:
                for e in range(1, N + 1):
                    if current_mask & (1 << e):
                        candidate = current_mask & ~(1 << e)
                        addable = []
                        for f in range(1, N + 1):
                            if not (candidate & (1 << f)) and can_add(f, candidate):
                                addable.append(f)

                        for i in range(len(addable)):
                            for j in range(i + 1, len(addable)):
                                f, g = addable[i], addable[j]
                                if can_add(g, candidate | (1 << f)):
                                    new_mask = candidate | (1 << f) | (1 << g)
                                    if is_valid(new_mask):
                                        current_mask = new_mask
                                        current_size = popcount(current_mask)
                                        improved = True
                                        break
                            if improved:
                                break
                    if improved:
                        break

            if not improved:
                break

        return current_mask

    # Strategy 1: Start with all odds (known valid independent set)
    odds = [e for e in range(1, N + 1) if e % 2 == 1]
    evens = [e for e in range(1, N + 1) if e % 2 == 0]

    odds_mask = 0
    for e in odds:
        odds_mask |= (1 << e)

    # Greedily add evens in order of increasing conflict count
    evens_sorted = sorted(evens, key=lambda e: popcount(conflict[e]))
    mask = greedy_build(evens_sorted, odds_mask)
    mask = local_search_add(mask)

    best_mask = mask
    best_size = popcount(mask)

    # Strategy 2: Multiple randomized greedy restarts
    rng = random.Random(42)
    for trial in range(500):
        if time.time() > deadline - 30.0:
            break
        shuffled = evens[:]
        rng.shuffle(shuffled)
        m = greedy_build(shuffled, odds_mask)
        m = local_search_add(m)
        s = popcount(m)
        if s > best_size:
            best_mask = m
            best_size = s

    # Strategy 3: Eigenvector-guided greedy
    if time.time() < deadline - 25.0:
        # Compute eigenvector centrality-like scores
        # Use power iteration on adjacency (conflict) matrix
        # Actually, we want LOW conflict, so use non-conflict adjacency
        # Let's just use degree-based scoring: prefer vertices with fewer conflicts
        scores = [popcount(conflict[e]) for e in range(N + 1)]

        # Try greedy with different orderings based on scores
        for ordering_seed in range(20):
            if time.time() > deadline - 25.0:
                break
            rng2 = random.Random(ordering_seed)
            # Sort evens by score with some randomization
            evens_by_score = sorted(evens, key=lambda e: (scores[e], rng2.random()))
            m = greedy_build(evens_by_score, odds_mask)
            m = local_search_add(m)
            s = popcount(m)
            if s > best_size:
                best_mask = m
                best_size = s

    # Strategy 4: Branch and bound on the remaining candidates
    # Start from best_mask, find candidates that can be added
    if time.time() < deadline - 5.0:
        # Identify vertices not in best_mask
        not_in_mask = 0
        for e in range(1, N + 1):
            if not (best_mask & (1 << e)):
                not_in_mask |= (1 << e)

        # For B&B, we need to find the max independent set among candidates
        # that can be added to best_mask
        # A vertex e can be added if conflict[e] & best_mask == 0
        valid_candidates = 0
        for e in range(1, N + 1):
            if not (best_mask & (1 << e)) and (conflict[e] & best_mask) == 0:
                valid_candidates |= (1 << e)

        # Now find max independent set within valid_candidates
        # (conflicts only matter among the candidates themselves)
        bb_time_limit = deadline - 1.0
        bb_mask, bb_size = branch_and_bound(valid_candidates, best_size, best_size, bb_time_limit)
        if bb_size > best_size:
            best_mask = best_mask | bb_mask
            best_size = bb_size

    # Strategy 5: Final local search polish
    if time.time() < deadline - 2.0:
        polished = local_search_add(best_mask)
        if popcount(polished) > best_size:
            best_mask = polished
            best_size = popcount(polished)

    if time.time() < deadline - 1.0:
        polished = two_for_one_swap(best_mask, deadline - 0.5)
        if popcount(polished) > best_size:
            best_mask = polished
            best_size = popcount(polished)

    # Convert to list
    result = [e for e in range(1, N + 1) if best_mask & (1 << e)]

    if not result:
        result = [1]

    # Final validation
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                result.pop(j)
                j -= 1

    return result if result else [1]
