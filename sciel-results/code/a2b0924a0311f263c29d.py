"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from math import gcd

    deadline = time.time() + 75  # safety margin before 90s kill

    # Precompute GCD table for 1..128
    n = 128
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    def _can_add(selected_set, x, gcd_table):
        """Check if x can be added to selected_set without creating a bad triple."""
        for a in selected_set:
            for b in selected_set:
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                g_ab = gcd_table[a][b]
                g_ax = gcd_table[a][x]
                g_bx = gcd_table[b][x]
                if g_ab == g_ax == g_bx:
                    return False
        return True

    def _count_conflicts(selected_set, x, gcd_table):
        """Count how many bad triples x would create."""
        count = 0
        for a in selected_set:
            for b in selected_set:
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                g_ab = gcd_table[a][b]
                g_ax = gcd_table[a][x]
                g_bx = gcd_table[b][x]
                if g_ab == g_ax == g_bx:
                    count += 1
        return count

    def _greedy(order, gcd_table):
        """Greedy add in given order."""
        selected = []
        selected_set = set()
        for x in order:
            if _can_add(selected_set, x, gcd_table):
                selected.append(x)
                selected_set.add(x)
        return selected

    def _local_search(selected, gcd_table):
        """Try to improve by swapping elements."""
        selected_set = set(selected)
        improved = True
        while improved:
            improved = False
            # Try removing one element and adding others
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                removed = selected[i]
                remaining = [x for j, x in enumerate(selected) if j != i]
                remaining_set = set(remaining)
                # Try adding candidates
                added_any = False
                for x in range(1, n + 1):
                    if x in remaining_set:
                        continue
                    if _can_add(remaining_set, x, gcd_table):
                        remaining.append(x)
                        remaining_set.add(x)
                        added_any = True
                if added_any and len(remaining) > len(selected):
                    selected = remaining
                    selected_set = remaining_set
                    improved = True
                    break
            if time.time() > deadline:
                break
        return selected

    # Strategy 1: Greedy with different orderings
    best = []

    # Natural order
    best = _greedy(list(range(1, n + 1)), gcd_table)

    # Descending order
    if time.time() < deadline:
        cand = _greedy(list(range(n, 0, -1)), gcd_table)
        if len(cand) > len(best):
            best = cand

    # Random orderings
    rng = random.Random(42)
    for _ in range(200):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        cand = _greedy(order, gcd_table)
        if len(cand) > len(best):
            best = cand

    # Local search on best
    if time.time() < deadline:
        best = _local_search(best, gcd_table)

    # Try local search on other good candidates
    if time.time() < deadline:
        for _ in range(50):
            if time.time() > deadline:
                break
            order = list(range(1, n + 1))
            rng.shuffle(order)
            cand = _greedy(order, gcd_table)
            if len(cand) >= len(best) - 2:
                cand = _local_search(cand, gcd_table)
                if len(cand) > len(best):
                    best = cand

    return best
# EVOLVE-BLOCK-END
