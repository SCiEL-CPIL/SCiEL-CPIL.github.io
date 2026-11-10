import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 600  # Safety margin

    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        sq = pow(x, 2, m)
        if sq != 0:
            forbidden.add(sq)

    # Build adjacency: conflict[i] is a set of vertices that conflict with i
    conflict = [set() for _ in range(m)]
    for i in range(m):
        for d in forbidden:
            j1 = (i + d) % m
            j2 = (i - d) % m
            conflict[i].add(j1)
            conflict[i].add(j2)

    # Precompute conflict masks as integers for fast bit operations
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in conflict[i]:
            mask |= (1 << j)
        conflict_mask[i] = mask

    # Degree of each vertex
    degree = [len(conflict[i]) for i in range(m)]

    def mask_to_list(mask):
        return [v for v in range(m) if (mask & (1 << v))]

    def list_to_mask(lst):
        mask = 0
        for v in lst:
            mask |= (1 << v)
        return mask

    def greedy_fill(mask, order):
        """Greedy fill: add vertices from order that don't conflict with mask."""
        for v in order:
            if (mask & (1 << v)) == 0 and (conflict_mask[v] & mask) == 0:
                mask |= (1 << v)
        return mask

    def greedy_construct(order):
        """Greedy construction given an order of candidates."""
        selected_mask = 0
        for candidate in order:
            if (conflict_mask[candidate] & selected_mask) == 0:
                selected_mask |= (1 << candidate)
        return mask_to_list(selected_mask), selected_mask

    def local_search(subset, mask, max_rounds=100, temperature=0.0):
        """
        Local search with 1-move and 2-move improvements.
        temperature > 0 allows occasional non-improving moves (simulated annealing).
        """
        subset = list(subset)
        mask = mask
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            n = len(subset)

            # Try removing 1 element and greedily adding back
            for i in range(n):
                if time.time() > deadline - 5:
                    return subset, mask
                u = subset[i]
                trial_mask = mask & ~(1 << u)
                # Greedy refill
                new_mask = trial_mask
                for v in range(m):
                    if (new_mask & (1 << v)) == 0 and (conflict_mask[v] & new_mask) == 0:
                        new_mask |= (1 << v)
                new_size = bin(new_mask).count('1')
                if new_size > n:
                    subset = mask_to_list(new_mask)
                    mask = new_mask
                    improved = True
                    break

            # Try removing 2 elements and greedily adding back
            if not improved:
                for i in range(n):
                    if time.time() > deadline - 5:
                        return subset, mask
                    for j in range(i + 1, n):
                        if time.time() > deadline - 5:
                            return subset, mask
                        u, v = subset[i], subset[j]
                        trial_mask = mask & ~(1 << u) & ~(1 << v)
                        new_mask = trial_mask
                        for v2 in range(m):
                            if (new_mask & (1 << v2)) == 0 and (conflict_mask[v2] & new_mask) == 0:
                                new_mask |= (1 << v2)
                        new_size = bin(new_mask).count('1')
                        if new_size > n:
                            subset = mask_to_list(new_mask)
                            mask = new_mask
                            improved = True
                            break
                    if improved:
                        break

        return subset, mask

    best_subset = []
    best_size = 0
    best_mask = 0

    random.seed(42)

    # Phase 1: Deterministic greedy (low-degree first)
    order = sorted(range(m), key=lambda x: degree[x])
    subset, mask = greedy_construct(order)
    subset, mask = local_search(subset, mask, max_rounds=50)
    if len(subset) > best_size:
        best_size = len(subset)
        best_subset = list(subset)
        best_mask = mask

    # Phase 2: Many random greedy trials with local search
    for trial in range(200000):
        if time.time() > deadline - 20:
            break

        # Vary the strategy
        r = random.random()
        if r < 0.3:
            # Random order
            order = list(range(m))
            random.shuffle(order)
        elif r < 0.5:
            # Degree-ordered with random tie-breaking
            order = sorted(range(m), key=lambda x: (degree[x], random.random()))
        elif r < 0.7:
            # Inverse degree (prefer high-degree vertices early to block them)
            order = sorted(range(m), key=lambda x: (-degree[x], random.random()))
        elif r < 0.85:
            # Weighted random: lower degree vertices more likely to appear early
            weights = [1.0 / (degree[x] + 1) ** 2 for x in range(m)]
            total_w = sum(weights)
            order = []
            remaining = list(range(m))
            while remaining:
                r_val = random.random() * sum(weights[v] for v in remaining)
                cum = 0
                for idx, v in enumerate(remaining):
                    cum += weights[v]
                    if cum >= r_val:
                        order.append(v)
                        remaining.pop(idx)
                        break
        else:
            # CRT-inspired: prefer vertices with small residues mod 5 and mod 41
            order = sorted(range(m), key=lambda x: (x % 5, x % 41, random.random()))

        subset, mask = greedy_construct(order)
        subset, mask = local_search(subset, mask, max_rounds=30)

        if len(subset) > best_size:
            best_size = len(subset)
            best_subset = list(subset)
            best_mask = mask

    # Phase 3: Final polish - more aggressive local search on best solution
    if time.time() < deadline - 30 and best_subset:
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            old_size = len(best_subset)
            best_subset, best_mask = local_search(best_subset, best_mask, max_rounds=100)
            if len(best_subset) == old_size:
                break

    # Final validation
    if not best_subset:
        subset, mask = greedy_construct(list(range(m)))
        best_subset = subset
        best_size = len(best_subset)

    # Verify no forbidden pairs
    best_set = set(best_subset)
    for a in best_subset:
        for b in best_subset:
            if a == b:
                continue
            diff = (a - b) % m
            if diff in forbidden:
                best_subset.remove(b)
                best_set.discard(b)
                break

    return np.array(best_subset, dtype=np.int64)
