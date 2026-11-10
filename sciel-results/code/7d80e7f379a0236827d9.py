import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 128
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Build conflict structure: conflict_pairs[x] = set of frozenset({a,b})
    # where {a,b,x} is a bad triple
    conflict_pairs = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        if time.time() > deadline - 10:
            break
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    conflict_pairs[a].add(frozenset((b, c)))
                    conflict_pairs[b].add(frozenset((a, c)))
                    conflict_pairs[c].add(frozenset((a, b)))
    
    def count_conflicts(x, selected_set):
        for pair in conflict_pairs[x]:
            a, b = tuple(pair)
            if a in selected_set and b in selected_set:
                return 1
        return 0
    
    def count_conflicts_excluding(x, selected_set, excluded_y):
        for pair in conflict_pairs[x]:
            a, b = tuple(pair)
            if a in selected_set and b in selected_set:
                if a != excluded_y and b != excluded_y:
                    return 1
        return 0
    
    best_result = set()
    
    # Define various orderings
    orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        sorted(range(1, n + 1), key=lambda x: -x.bit_count()),
        sorted(range(1, n + 1), key=lambda x: x.bit_count()),
        list(range(1, n + 1, 2)) + list(range(2, n + 1, 2)),
        list(range(2, n + 1, 2)) + list(range(1, n + 1, 2)),
        sorted(range(1, n + 1), key=lambda x: len(conflict_pairs[x])),
        sorted(range(1, n + 1), key=lambda x: -len(conflict_pairs[x])),
    ]
    
    # Run greedy for each deterministic order
    for order in orders:
        if time.time() > deadline - 15:
            break
        selected = set()
        for x in order:
            if count_conflicts(x, selected) == 0:
                selected.add(x)
        if len(selected) > len(best_result):
            best_result = selected.copy()
    
    # Randomized greedy restarts
    import random
    rng = random.Random(42)
    for restart in range(30):
        if time.time() > deadline - 15:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        selected = set()
        for x in order:
            if count_conflicts(x, selected) == 0:
                selected.add(x)
        if len(selected) > len(best_result):
            best_result = selected.copy()
    
    # Local search improvement
    current = best_result.copy()
    
    # Phase 1: Try to add any element that fits (multiple passes)
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for x in range(1, n + 1):
            if x in current:
                continue
            if count_conflicts(x, current) == 0:
                current.add(x)
                improved = True
                break
    
    # Phase 2: 1-for-1 Swaps
    swap_attempts = 0
    max_swap_attempts = 10000
    while time.time() < deadline - 5 and swap_attempts < max_swap_attempts:
        swap_found = False
        for x in range(1, n + 1):
            if x in current:
                continue
            for y in list(current):
                swap_attempts += 1
                if swap_attempts > max_swap_attempts or time.time() > deadline - 5:
                    break
                if count_conflicts_excluding(x, current, y) == 0:
                    current.remove(y)
                    current.add(x)
                    swap_found = True
                    break
            if swap_found or swap_attempts > max_swap_attempts:
                break
        if not swap_found:
            break
    
    # Phase 3: Try 1-for-2 swaps (remove 1, add 2)
    if time.time() < deadline - 5:
        for x1 in range(1, n + 1):
            if x1 in current or time.time() > deadline - 5:
                continue
            for x2 in range(x1 + 1, n + 1):
                if x2 in current or time.time() > deadline - 5:
                    continue
                for y in list(current):
                    if count_conflicts_excluding(x1, current, y) == 0 and count_conflicts_excluding(x2, current - {y}, y) == 0:
                        current.remove(y)
                        current.add(x1)
                        current.add(x2)
                        break
    
    # Phase 4: Final add pass
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for x in range(1, n + 1):
            if x in current:
                continue
            if count_conflicts(x, current) == 0:
                current.add(x)
                improved = True
                break
    
    return list(current) if current else list(best_result)
