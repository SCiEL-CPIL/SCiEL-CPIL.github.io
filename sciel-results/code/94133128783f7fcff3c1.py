def construct():
    import time
    import random
    from math import gcd
    from itertools import combinations

    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill

    def lcm(a, b):
        return a * b // gcd(a, b)

    # Precompute all bad triples: (a, b, c) with a < b < c where
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    # Represent as: for each element x, a set of (y, z) pairs that form bad triples with x
    # Also a set of all bad triples for quick lookup

    n = 128
    elements = list(range(1, n + 1))

    # bad_pairs_with[x] = set of frozenset({y,z}) such that {x,y,z} is a bad triple
    # bad_triples_set = set of frozenset({a,b,c}) for all bad triples
    bad_pairs_with = [set() for _ in range(n + 1)]
    bad_triples_set = set()

    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm(a, b)
            for c in range(b + 1, n + 1):
                if lab == lcm(a, c) == lcm(b, c):
                    triple = frozenset((a, b, c))
                    bad_triples_set.add(triple)
                    bad_pairs_with[a].add(frozenset((b, c)))
                    bad_pairs_with[b].add(frozenset((a, c)))
                    bad_pairs_with[c].add(frozenset((a, b)))

    def is_valid_addition(selected_set, x):
        """Check if adding x to selected_set creates any bad triple."""
        # For each pair (y,z) in selected_set, check if {x,y,z} is a bad triple
        sel_list = list(selected_set)
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                y, z = sel_list[i], sel_list[j]
                if frozenset((x, y, z)) in bad_triples_set:
                    return False
        return True

    def greedy_random(rng):
        """Greedy: add elements in random order if valid."""
        order = elements[:]
        rng.shuffle(order)
        selected = set()
        for x in order:
            if is_valid_addition(selected, x):
                selected.add(x)
        return selected

    def local_search(selected, rng, max_rounds=50):
        """Try remove-one/add-one moves to improve size."""
        current = set(selected)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            # Try removing each element
            for x in list(current):
                candidate = current - {x}
                # Try to add back other elements
                gains = 0
                added = set()
                for y in elements:
                    if y not in candidate:
                        if is_valid_addition(candidate | added, y):
                            added.add(y)
                            gains += 1
                if gains > 1:  # net gain of at least 1 (removed 1, added gains)
                    current = candidate | added
                    improved = True
                    break
        return current

    def local_search2(selected, rng, max_rounds=30):
        """More aggressive local search: try multiple removals."""
        current = set(selected)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            # Try removing pairs
            for x, y in combinations(sorted(current), 2):
                candidate = current - {x, y}
                added = set()
                gains = 0
                for z in elements:
                    if z not in candidate:
                        if is_valid_addition(candidate | added, z):
                            added.add(z)
                            gains += 1
                if gains > 2:
                    current = candidate | added
                    improved = True
                    break
        return current

    best = set()
    best_size = 0

    # Run multiple greedy + local search restarts
    rng = random.Random(42)

    # First, do a deterministic greedy to get a baseline
    baseline = set()
    for x in elements:
        if is_valid_addition(baseline, x):
            baseline.add(x)
    best = set(baseline)
    best_size = len(baseline)

    # Multi-restart with random greedy + local search
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break

        sel = greedy_random(rng)
        # Local search phase 1
        sel = local_search(sel, rng, max_rounds=30)
        # Local search phase 2 (more aggressive)
        if time.time() - start_time > DEADLINE:
            break
        sel = local_search2(sel, rng, max_rounds=20)

        if len(sel) > best_size:
            best = set(sel)
            best_size = len(sel)

    # Final check: try to add more elements to best
    if time.time() - start_time < DEADLINE:
        for x in elements:
            if x not in best:
                if is_valid_addition(best, x):
                    best.add(x)

    # Ensure nonempty
    if not best:
        best = {1}

    return sorted(best)
