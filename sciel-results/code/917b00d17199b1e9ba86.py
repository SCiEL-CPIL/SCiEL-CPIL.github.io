import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute conflict pairs
    conflict_pairs = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    conflict_pairs[a].add((b, c))
                    conflict_pairs[b].add((a, c))
                    conflict_pairs[c].add((a, b))
    
    def count_conflicts(x, selected_set):
        for (a, b) in conflict_pairs[x]:
            if a in selected_set and b in selected_set:
                return 1
        return 0
    
    def is_valid(x, selected_set):
        return count_conflicts(x, selected_set) == 0
    
    best_result = set()
    
    # Multiple greedy passes
    orderings = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        sorted(range(1, n + 1), key=lambda x: -x.bit_count()),
        sorted(range(1, n + 1), key=lambda x: x.bit_count()),
        list(range(1, n + 1, 2)) + list(range(2, n + 1, 2)),
        list(range(2, n + 1, 2)) + list(range(1, n + 1, 2)),
        sorted(range(1, n + 1), key=lambda x: (x % 2, x)),
        sorted(range(1, n + 1), key=lambda x: (-x % 3, x)),
    ]
    
    for order in orderings:
        if time.time() > deadline - 15:
            break
        selected = set()
        for x in order:
            if is_valid(x, selected):
                selected.add(x)
        if len(selected) > len(best_result):
            best_result = selected.copy()
    
    # Local improvement
    if time.time() < deadline - 5:
        current = best_result.copy()
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding
            for x in range(1, n + 1):
                if x not in current and is_valid(x, current):
                    current.add(x)
                    improved = True
                    break
            if improved:
                continue
            # Try 1-for-1 swap
            for x in range(1, n + 1):
                if x in current:
                    continue
                for y in list(current):
                    if time.time() > deadline - 5:
                        break
                    tmp = current - {y}
                    if is_valid(x, tmp):
                        current = tmp
                        current.add(x)
                        improved = True
                        break
                if improved:
                    break
    
    return list(current) if current else list(best_result)
