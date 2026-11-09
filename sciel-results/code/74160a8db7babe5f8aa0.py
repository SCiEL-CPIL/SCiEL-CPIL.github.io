"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    max_n = 128
    nums = list(range(1, max_n + 1))
    
    # Precompute lcm table
    lcm_table = [[0] * (max_n + 1) for _ in range(max_n + 1)]
    for i in range(1, max_n + 1):
        for j in range(i, max_n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute conflict pairs
    # conflict_pairs[x] = list of (a, b) such that {a, b, x} is a bad triple
    conflict_pairs = [[] for _ in range(max_n + 1)]
    
    for a in range(1, max_n + 1):
        for b in range(a + 1, max_n + 1):
            ab = lcm_table[a][b]
            for c in range(b + 1, max_n + 1):
                if lcm_table[a][c] == ab and lcm_table[b][c] == ab:
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
    
    # Precompute bitmasks for conflict pairs for faster checking
    conflict_pairs_bits = [[] for _ in range(max_n + 1)]
    for x in range(1, max_n + 1):
        for a, b in conflict_pairs[x]:
            conflict_pairs_bits[x].append((1 << (a-1), 1 << (b-1)))
    
    def greedy(order, mask=0):
        for x in order:
            ok = True
            for bit_a, bit_b in conflict_pairs_bits[x]:
                if (mask & bit_a) and (mask & bit_b):
                    ok = False
                    break
            if ok:
                mask |= (1 << (x-1))
        return mask
    
    def mask_to_set(mask):
        return [i+1 for i in range(max_n) if (mask >> i) & 1]
    
    best_mask = 0
    best_size = 0
    
    rng = random.Random(42)
    
    # 1. Random Restarts with Greedy
    orders = [
        list(range(1, max_n + 1)),
        list(range(max_n, 0, -1)),
    ]
    
    for order in orders:
        m = greedy(order)
        size = bin(m).count("1")
        if size > best_size:
            best_size = size
            best_mask = m
    
    # Random shuffles
    for trial in range(20000):
        if time.time() > deadline - 5.0:
            break
        order = list(range(1, max_n + 1))
        rng.shuffle(order)
        m = greedy(order)
        size = bin(m).count("1")
        if size > best_size:
            best_size = size
            best_mask = m
    
    # 2. Local Search (Iterated Local Search)
    improved = True
    current_mask = best_mask
    current_size = best_size
    
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Try different perturbation sizes
        for k in [1, 2, 3, 5, 10]:
            if time.time() > deadline - 2.0:
                break
            
            # Pick k random elements to remove
            set_bits = [i+1 for i in range(max_n) if (current_mask >> i) & 1]
            if len(set_bits) < k:
                continue
                
            removed = rng.sample(set_bits, k)
            new_mask = current_mask
            for x in removed:
                new_mask &= ~(1 << (x-1))
            
            # Greedily add back
            candidates = [x for x in range(1, max_n + 1) if not (new_mask & (1 << (x-1)))]
            rng.shuffle(candidates)
            
            m = greedy(candidates, new_mask)
            size = bin(m).count("1")
            
            if size > current_size:
                current_size = size
                current_mask = m
                if size > best_size:
                    best_size = size
                    best_mask = m
                improved = True
                break # Break k loop, restart while loop with new current
    
    return mask_to_set(best_mask)
# EVOLVE-BLOCK-END
