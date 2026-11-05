import time
from itertools import combinations
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin
    
    # Precompute GCDs for all pairs in [1, 128]
    # This speeds up checks significantly
    max_val = 128
    gcd_table = [[0] * (max_val + 1) for _ in range(max_val + 1)]
    for i in range(1, max_val + 1):
        for j in range(i, max_val + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_bad_triple(a, b, c):
        return gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]

    def can_add(selected, x):
        # Check if x can be added to selected
        # selected is a list of ints
        n = len(selected)
        for i in range(n):
            a = selected[i]
            g_ax = gcd_table[a][x]
            for j in range(i + 1, n):
                b = selected[j]
                if g_ax == gcd_table[a][b] == gcd_table[b][x]:
                    return False
        return True

    # Initial set: 2^x * 3^y grid
    initial_set = []
    for x in range(8):
        pow2 = (1 << x)
        for y in range(5):
            val = pow2 * (3 ** y)
            if val <= 128:
                initial_set.append(val)
    
    selected = list(initial_set)
    
    # Greedy add
    for x in range(1, 129):
        if time.time() > deadline:
            break
        if x not in selected:
            if can_add(selected, x):
                selected.append(x)
                
    # Local search: try to swap
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(selected)
        for i in range(n):
            if time.time() > deadline:
                break
            y = selected[i]
            # Try removing y
            candidate = selected[:i] + selected[i+1:]
            # Try adding any x not in candidate
            for x in range(1, 129):
                if x in candidate:
                    continue
                if can_add(candidate, x):
                    new_set = candidate + [x]
                    # Check if this is actually an improvement (it is, size is same, but maybe allows more later? 
                    # Actually, if we just swap, size is same. We want to increase size.
                    # So this loop is for "stuck" situations.
                    # But if we can add x to candidate, we should have been able to add it to selected?
                    # No, because y might have blocked x.
                    # So if we remove y and add x, size is same.
                    # We need to check if we can add MORE after this swap.
                    # This is complex. Let's just do a simple "add" phase first.
                    pass
            
    # Better local search: 
    # 1. Try to add any number.
    # 2. If stuck, try removing one number and then adding two?
    
    # Let's refine the greedy add to be more thorough
    # Sort by "centrality" or just random?
    # The current greedy is deterministic.
    
    # Let's try a few random restarts if time permits
    import random
    best_size = len(selected)
    best_set = selected[:]
    
    # If we have time, try random shuffles for the greedy add
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            random.shuffle(list(range(1, 129)))
            order = list(range(1, 129))
            random.shuffle(order)
            
            curr_set = []
            # Start with a subset of the grid? Or empty?
            # Let's start with the grid, then add randoms
            curr_set = list(initial_set)
            
            for x in order:
                if x in curr_set:
                    continue
                if can_add(curr_set, x):
                    curr_set.append(x)
            
            if len(curr_set) > best_size:
                best_size = len(curr_set)
                best_set = curr_set[:]
                
    return best_set if best_set else [1]
