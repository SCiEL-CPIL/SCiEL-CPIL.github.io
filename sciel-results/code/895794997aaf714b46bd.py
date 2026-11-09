"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    N = 128
    numbers = list(range(1, N + 1))
    
    # 1. Precompute conflict pairs
    # conflicts[x] = list of (y, z) such that {x, y, z} is a bad triple
    conflicts = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                if gcd(b, c) == g_ab and gcd(c, a) == g_ab:
                    conflicts[a].append((b, c))
                    conflicts[b].append((a, c))
                    conflicts[c].append((a, b))

    # 2. Initial Greedy Solution
    # Order by ascending conflict count
    order_greedy = sorted(range(1, N + 1), key=lambda x: len(conflicts[x]))
    current_set = set()
    for x in order_greedy:
        can_add = True
        for y, z in conflicts[x]:
            if y in current_set and z in current_set:
                can_add = False
                break
        if can_add:
            current_set.add(x)
            
    best_size = len(current_set)
    best_set = current_set.copy()
    
    # 3. Backtracking Search with Bitsets
    # Order by descending conflict count for better pruning
    order_bt = sorted(range(1, N + 1), key=lambda x: len(conflicts[x]), reverse=True)
    
    # Precompute bit masks
    masks = {i: 1 << (i - 1) for i in range(1, N + 1)}
    
    start_time = time.time()
    timeout = 55.0
    nodes_visited = 0

    def can_include(x, included_mask):
        for y, z in conflicts[x]:
            if (included_mask & masks[y]) and (included_mask & masks[z]):
                return False
        return True

    def search(idx, included_mask, count):
        nonlocal best_size, best_set, nodes_visited
        
        if time.time() - start_time > timeout:
            return
            
        if count + (len(order_bt) - idx) <= best_size:
            return
            
        if idx == len(order_bt):
            if count > best_size:
                best_size = count
                s = set()
                m = included_mask
                while m:
                    lsb = m & -m
                    idx_bit = lsb.bit_length() - 1
                    s.add(idx_bit + 1)
                    m ^= lsb
                best_set = s
            return
            
        x = order_bt[idx]
        
        # Option 1: Include x
        if can_include(x, included_mask):
            nodes_visited += 1
            search(idx + 1, included_mask | masks[x], count + 1)
            
        # Option 2: Exclude x
        nodes_visited += 1
        search(idx + 1, included_mask, count)

    # Start search
    search(0, 0, 0)
    
    return sorted(list(best_set))

# EVOLVE-BLOCK-END
