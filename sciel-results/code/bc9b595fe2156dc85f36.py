"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def can_add(S_set, x):
        """Check if x can be added to sum-free set S without violation.
        Violation if: exists a in S such that a+x in S, or x-a in S, or 2x in S.
        """
        for a in S_set:
            if (a + x) in S_set:
                return False
            if (x - a) in S_set:
                return False
        if (2 * x) in S_set:
            return False
        return True

    def is_sum_free(S):
        """Verify S is sum-free."""
        S_set = set(S)
        for a in S:
            for b in S:
                if (a + b) in S_set:
                    return False
        return True

    def greedy_expand(S_list, V, order=None):
        """Add elements from V to S in the given order."""
        S_set = set(S_list)
        if order is None:
            order = sorted(V, key=lambda x: abs(x), reverse=True)
        for x in order:
            if x not in S_set and can_add(S_set, x):
                S_set.add(x)
        return list(S_set)

    def repair(S_list, V):
        """Remove elements to fix violations, then greedily expand."""
        S_set = set(S_list)
        V_set = set(V)

        # Phase 1: Remove elements causing violations
        changed = True
        while changed:
            changed = False
            S_list = list(S_set)
            # Find violations
            viol_counts = Counter()
            for i, a in enumerate(S_list):
                for b in S_list[i:]:
                    c = a + b
                    if c in S_set:
                        viol_counts[a] += 1
                        viol_counts[b] += 1
                        viol_counts[c] += 1
            if viol_counts:
                worst = max(viol_counts, key=viol_counts.get)
                S_set.discard(worst)
                changed = True

        # Phase 2: Greedy expand
        S_list = list(S_set)
        S_list = greedy_expand(S_list, V_set - S_set)
        return S_list

    def local_search(S_list, V, max_iters=300, time_limit=None):
        """Local search with 1-opt (add/remove) and 2-opt (swap) moves."""
        S_set = set(S_list)
        V_set = set(V)
        best_size = len(S_set)
        no_improve = 0

        for it in range(max_iters):
            if time_limit is not None and time.time() - time_limit > 30:
                break
            improved = False

            # Try 1-opt: add an element
            candidates = [x for x in V_set if x not in S_set]
            random.shuffle(candidates)
            for x in candidates[:30]:
                if can_add(S_set, x):
                    S_set.add(x)
                    best_size = len(S_set)
                    improved = True
                    break

            # Try 1-opt: remove an element that's causing issues
            if not improved:
                S_list = list(S_set)
                random.shuffle(S_list)
                for a in S_list[:15]:
                    S_set.discard(a)
                    # Try to add back something better
                    candidates2 = [x for x in V_set if x not in S_set]
                    random.shuffle(candidates2)
                    added = False
                    for x in candidates2[:40]:
                        if can_add(S_set, x):
                            S_set.add(x)
                            added = True
                            if len(S_set) > best_size:
                                best_size = len(S_set)
                            improved = True
                            break
                    if not added:
                        S_set.add(a)  # put back

            # Try 2-opt: swap two elements
            if not improved:
                S_list = list(S_set)
                if len(S_list) >= 2:
                    random.shuffle(S_list)
                    for i in range(min(10, len(S_list))):
                        a = S_list[i]
                        for j in range(i+1, min(i+10, len(S_list))):
                            b = S_list[j]
                            S_set.discard(a)
                            S_set.discard(b)
                            # Try to add two new elements
                            candidates3 = [x for x in V_set if x not in S_set]
                            random.shuffle(candidates3)
                            added_count = 0
                            temp_set = set(S_set)
                            for x in candidates3[:60]:
                                if can_add(temp_set, x):
                                    temp_set.add(x)
                                    added_count += 1
                                    if added_count == 2:
                                        break
                            if added_count == 2 and len(temp_set) > best_size:
                                S_set = temp_set
                                best_size = len(S_set)
                                improved = True
                                break
                            else:
                                S_set.add(a)
                                S_set.add(b)
                        if improved:
                            break

            if not improved:
                no_improve += 1
                if no_improve > 50:
                    break
            else:
                no_improve = 0

        return list(S_set)

    def search_instance(elements, seed_offset=0, time_budget=30):
        """Search for a large sum-free subset of elements."""
        rng = random.Random(seed_offset)
        best = []
        best_size = 0
        t_start = time.time()

        def try_candidate(cand):
            nonlocal best, best_size
            S = repair(cand, elements)
            S = local_search(S, elements, max_iters=200, time_limit=t_start)
            if len(S) > best_size:
                best_size = len(S)
                best = list(S)

        # Generate diverse initial candidates
        candidates = []

        # 1. Odds (guaranteed sum-free)
        odd = [x for x in elements if x % 2 != 0]
        candidates.append(odd)

        # 2. All positive
        pos = [x for x in elements if x > 0]
        candidates.append(pos)

        # 3. All negative
        neg = [x for x in elements if x < 0]
        candidates.append(neg)

        # 4. Top-k by value (largest)
        sorted_desc = sorted(elements, reverse=True)
        for frac in [0.3, 0.4, 0.5, 0.6]:
            k = int(len(elements) * frac)
            candidates.append(sorted_desc[:k])

        # 5. Top-k by magnitude
        sorted_mag = sorted(elements, key=lambda x: abs(x), reverse=True)
        for frac in [0.3, 0.4, 0.5]:
            k = int(len(elements) * frac)
            candidates.append(sorted_mag[:k])

        # 6. Interval-based: elements in [L, R]
        elements_sorted = sorted(elements)
        n = len(elements_sorted)
        for lo_frac in [0.0, 0.2, 0.4]:
            for hi_frac in [0.6, 0.8, 1.0]:
                lo = int(n * lo_frac)
                hi = int(n * hi_frac)
                if hi > lo:
                    candidates.append(elements_sorted[lo:hi])

        # 7. Random samples
        for _ in range(8):
            k = int(len(elements) * rng.uniform(0.3, 0.8))
            candidates.append(rng.sample(elements, k))

        # Process each candidate
        for cand in candidates:
            try_candidate(cand)

        # Additional random restarts if time permits
        restart_count = 0
        while time.time() - t_start < time_budget - 5:
            k = int(len(elements) * rng.uniform(0.3, 0.8))
            cand = rng.sample(elements, k)
            try_candidate(cand)
            restart_count += 1
            if restart_count > 50:
                break

        return best

    # Generate instances
    instances = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        instances.append(instance(size, seed))

    results = []
    # Allocate time budget proportionally to instance size, with a minimum
    # n=60 gets 30s, n=120 gets 40s, n=240 gets 50s (total 120s, leaving 25s margin)
    time_budgets = [30, 40, 50]
    for i, elems in enumerate(instances):
        if time.time() - start_time > DEADLINE:
            # Fallback: odd numbers
            results.append([x for x in elems if x % 2 != 0])
            continue
        result = search_instance(elems, seed_offset=801 + i, time_budget=time_budgets[i])
        results.append(result)

    return results
# EVOLVE-BLOCK-END
