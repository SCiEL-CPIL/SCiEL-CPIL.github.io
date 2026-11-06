import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0
    
    n = 256
    max_sum = 2 * n
    
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # Diagonal constraint
    valid_numbers = []
    for a in range(1, n + 1):
        if 2 * a not in squares:
            valid_numbers.append(a)
    
    num_valid = len(valid_numbers)
    
    # Build conflict sets
    conflicts = [set() for _ in range(num_valid)]
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b > a and b <= n:
                # Find index of b
                # We need a fast lookup
                pass
    
    # Better: build using the squares
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b > a and b in val_to_idx:
                j = val_to_idx[b]
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Min-degree greedy
    def min_degree_greedy():
        chosen = set()
        remaining = set(range(num_valid))
        while remaining:
            # Pick vertex with minimum degree in remaining
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                deg = len(conflicts[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            chosen.add(best_v)
            remaining.discard(best_v)
            # Remove all neighbors of best_v from remaining
            for nb in conflicts[best_v]:
                remaining.discard(nb)
        return chosen
    
    best_set = min_degree_greedy()
    
    # Local search: add any vertex that fits
    def add_fitting(s):
        changed = True
        while changed:
            changed = False
            for j in range(num_valid):
                if j not in s and not (conflicts[j] & s):
                    s.add(j)
                    changed = True
        return s
    
    best_set = add_fitting(best_set)
    
    # Try multiple random greedy passes
    for trial in range(500):
        if time.time() - start_time > deadline - 5:
            break
        random.seed(trial)
        order = list(range(num_valid))
        random.shuffle(order)
        s = set()
        for idx in order:
            if not (conflicts[idx] & s):
                s.add(idx)
        s = add_fitting(s)
        if len(s) > len(best_set):
            best_set = s
    
    # Local search with swaps
    def local_search(s):
        improved = True
        while improved and (time.time() - start_time) < deadline - 2:
            improved = False
            # Try adding
            for j in range(num_valid):
                if j not in s and not (conflicts[j] & s):
                    s.add(j)
                    improved = True
                    break
            # Try swaps
            if not improved:
                s_list = list(s)
                random.shuffle(s_list)
                for i in s_list:
                    for j in range(num_valid):
                        if j not in s:
                            rem = s - {i}
                            if not (conflicts[j] & rem):
                                s.discard(i)
                                s.add(j)
                                improved = True
                                break
                    if improved:
                        break
        return s
    
    best_set = local_search(best_set)
    
    result = sorted([valid_numbers[i] for i in best_set])
    if not result:
        return [1]
    return result
