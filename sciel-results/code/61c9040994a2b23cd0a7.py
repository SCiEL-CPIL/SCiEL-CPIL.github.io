import time
import random
from math import lcm
from itertools import combinations

def construct():
    start_time = time.time()
    limit = 88.0  # Leave 2s margin for safety
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute LCMs
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    def is_bad_triple(a, b, x):
        l_ab = lcm_table[a][b]
        l_ax = lcm_table[a][x]
        l_bx = lcm_table[b][x]
        return l_ab == l_ax == l_bx

    # Precompute conflicts for each x to speed up validity check?
    # Actually, checking against current set is better as set size grows.
    
    best_set = []
    best_len = 0
    
    def greedy_add(order):
        selected = set()
        # Always try to add 1 first if it's in the order, or just force it?
        # 1 is always safe.
        for x in order:
            valid = True
            # Check against all pairs in selected
            # To speed up, if selected is large, we can break early.
            # Using list(selected) for combinations is slightly faster than set?
            # Let's just use the set directly in combinations.
            for a, b in combinations(selected, 2):
                if is_bad_triple(a, b, x):
                    valid = False
                    break
            if valid:
                selected.add(x)
        return selected

    # Heuristic 1: Sort by number of conflicts in the full universe
    # This helps find a good starting point
    conflict_counts = [0] * (N + 1)
    for x in range(1, N + 1):
        count = 0
        for a in range(1, N + 1):
            if a == x: continue
            for b in range(a + 1, N + 1):
                if b == x: continue
                if is_bad_triple(a, b, x):
                    count += 1
        conflict_counts[x] = count
        
    order_by_conflicts = sorted(nums, key=lambda x: conflict_counts[x])
    # Force 1 to be first
    if 1 in order_by_conflicts:
        order_by_conflicts.remove(1)
        order_by_conflicts.insert(0, 1)
        
    s = greedy_add(order_by_conflicts)
    if len(s) > best_len:
        best_len = len(s)
        best_set = list(s)
        
    # Random restarts
    # Ensure 1 is always included by putting it at the start of the random order
    random_nums = [x for x in nums if x != 1]
    
    while time.time() - start_time < limit:
        random.shuffle(random_nums)
        order = [1] + random_nums
        s = greedy_add(order)
        if len(s) > best_len:
            best_len = len(s)
            best_set = list(s)
            
    if not best_set:
        return [1]
        
    return best_set
