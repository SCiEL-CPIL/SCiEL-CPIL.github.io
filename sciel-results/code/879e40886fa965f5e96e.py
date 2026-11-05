"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def can_add(S_set, x):
        """Check if x can be added to sum-free set S without violation."""
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

    def greedy_expand(S_set, candidates, order=None):
        """Add elements from candidates to S in the given order."""
        if order is None:
            order = sorted(candidates, key=lambda x: abs(x), reverse=True)
        for x in order:
            if x not in S_set and can_add(S_set, x):
                S_set.add(x)
        return S_set

    def repair(S_list, V):
        """Remove elements to fix violations using hypergraph-aware ordering, then greedily expand."""
        S_set = set(S_list)
        V_set = set(V)

        # Phase 1: Remove elements causing violations, prioritizing high-conflict elements
        changed = True
        while changed:
            changed = False
            S_list = list(S_set)
            # Find violations and count conflicts
            viol_counts = Counter()
            conflict_partners = {}  # element -> set of elements it conflicts with
            for i, a in enumerate(S_list):
                for j in range(i, len(S_list)):
                    b = S_list[j]
                    c = a + b
                    if c in S_set:
                        viol_counts[a] += 1
                        viol_counts[b] += 1
                        viol_counts[c] += 1
                        # Track conflict partners for hypergraph-aware ordering
                        if a not in conflict_partners:
                            conflict_partners[a] = set()
                        if b not in conflict_partners:
                            conflict_partners[b] = set()
                        if c not in conflict_partners:
                            conflict_partners[c] = set()
                        conflict_partners[a].add(b)
                        conflict_partners[a].add(c)
                        conflict_partners[b].add(a)
                        conflict_partners[b].add(c)
                        conflict_partners[c].add(a)
                        conflict_partners[c].add(b)

            if viol_counts:
                # Prioritize removing elements with highest conflict count
                # Tie-break by fewer conflict partners (lower degree) to preserve structure
                worst = max(viol_counts.keys(), key=lambda x: (viol_counts[x], -len(conflict_partners.get(x, set()))))
                S_set.discard(worst)
                changed = True

        # Phase 2: Greedy expand
        candidates = V_set - S_set
        S_set = greedy_expand(S_set, candidates)
        return list(S_set)

    def local_search(S_list, V, max_iters=300, time_limit=None):
        """Local search with 1-opt (add/remove) and 2-opt (swap) moves."""
        S_set = set(S_list)
        V_set = set(V)
        best_size = len(S_set)
        no_improve = 0
        rng = random.Random(hash(frozenset(S_set)) % (2**32))

        for it in range(max_iters):
            if time_limit is not None and time.time() - time_limit > 25:
                break
            improved = False

            # Try 1-opt: add an element
            candidates = [x for x in V_set if x not in S_set]
            rng.shuffle(candidates)
            for x in candidates[:25]:
                if can_add(S_set, x):
                    S_set.add(x)
                    best_size = len(S_set)
                    improved = True
                    break

            # Try 1-opt: remove an element, then try to add back something
            if not improved:
                S_list = list(S_set)
                rng.shuffle(S_list)
                for a in S_list[:12]:
                    S_set.discard(a)
                    candidates2 = [x for x in V_set if x not in S_set]
                    rng.shuffle(candidates2)
                    added = False
                    for x in candidates2[:30]:
                        if can_add(S_set, x):
                            S_set.add(x)
                            added = True
                            if len(S_set) > best_size:
                                best_size = len(S_set)
                            improved = True
                            break
                    if not added:
                        S_set.add(a)  # put back

            # Try 2-opt: swap two elements for two new ones
            if not improved:
                S_list = list(S_set)
                if len(S_list) >= 2:
                    rng.shuffle(S_list)
                    for i in range(min(8, len(S_list))):
                        a = S_list[i]
                        for j in range(i + 1, min(i + 8, len(S_list))):
                            b = S_list[j]
                            S_set.discard(a)
                            S_set.discard(b)
                            # Try to add two new elements
                            candidates3 = [x for x in V_set if x not in S_set]
                            rng.shuffle(candidates3)
                            temp_set = set(S_set)
                            added_count = 0
                            for x in candidates3[:50]:
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
                if no_improve > 40:
                    break
            else:
                no_improve = 0

        return list(S_set)

    def search_instance(elements, seed_offset=0, time_budget=25):
        """Search for a large sum-free subset of elements."""
        rng = random.Random(seed_offset)
        best = []
        best_size = 0
        t_start = time.time()

        def try_candidate(cand):
            nonlocal best, best_size
            if time.time() - t_start > time_budget:
                return
            S = repair(cand, elements)
            S = local_search(S, elements, max_iters=150, time_limit=t_start)
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
            if time.time() - t_start > time_budget:
                break
            try_candidate(cand)

        # Additional random restarts if time permits
        remaining = time_budget - (time.time() - t_start)
        if remaining > 2:
            for _ in range(30):
                if time.time() - t_start > time_budget:
                    break
                k = int(len(elements) * rng.uniform(0.3, 0.8))
                cand = rng.sample(elements, k)
                try_candidate(cand)

        # Final local search on best if time permits
        if time.time() - t_start < time_budget and best:
            improved = local_search(best, elements, max_iters=200, time_limit=t_start)
            if len(improved) > best_size:
                best = improved

        return best

    # Generate instances
    instances = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        instances.append(instance(size, seed))

    results = []
    for i, elems in enumerate(instances):
        if time.time() - start_time > DEADLINE:
            # Fallback: odd numbers
            results.append([x for x in elems if x % 2 != 0])
            continue

        # Allocate more time to larger instances
        if i == 0:
            time_budget = 30
        elif i == 1:
            time_budget = 35
        else:
            time_budget = 40

        result = search_instance(elems, seed_offset=801 + i, time_budget=time_budget)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
