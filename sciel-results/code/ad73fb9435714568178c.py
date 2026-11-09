"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import sys

sys.setrecursionlimit(2000)

DEADLINE = time.time() + 85

def construct():
    n = 128
    nums = list(range(1, n + 1))
    
    # Precompute gcds
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(nums[i], nums[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Precompute pair_bad[i][j] as a bitmask of k such that {i,j,k} is bad
    # bad if gcd(i,j) == gcd(i,k) == gcd(j,k)
    pair_bad = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g_ij = gcds[i][j]
            mask = 0
            for k in range(n):
                if k == i or k == j: continue
                if gcds[i][k] == g_ij and gcds[j][k] == g_ij:
                    mask |= (1 << k)
            pair_bad[i][j] = mask
            pair_bad[j][i] = mask
            
    # Calculate "degree" for ordering: how many bad triples does each element participate in?
    # Approximation: Sum of popcount(pair_bad[i][j]) for all j.
    # Or just count pairs (j,k) where i is in a bad triple.
    degrees = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j: continue
            degrees[i] += bin(pair_bad[i][j]).count('1')
            
    # Order by descending degree (most constrained first)
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # Map from ordered index to actual index
    # We will iterate through `order` in the B&B
    # idx in B&B corresponds to order[idx]
    
    best_size = 0
    best_mask = 0
    best_set = []
    
    # To speed up can_add, we precompute nothing more, but use the bitmask logic.
    
    def can_add(actual_idx, current_mask):
        # Check if actual_idx can be added to current_mask
        # For each i in current_mask, check if pair_bad[i][actual_idx] intersects current_mask
        m = current_mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            m ^= lsb
            # If any k in current_mask is in pair_bad[i][actual_idx], it's a conflict
            if pair_bad[i][actual_idx] & current_mask:
                return False
        return True

    def solve(pos, current_mask, current_size):
        nonlocal best_size, best_mask, best_set
        
        if time.time() > DEADLINE:
            return
            
        if pos == n:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
                # Extract set
                s = []
                m = current_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    s.append(nums[idx])
                    m ^= lsb
                best_set = sorted(s)
            return
            
        # Upper bound pruning
        # Max possible size is current_size + (n - pos)
        if current_size + (n - pos) <= best_size:
            return
            
        actual_idx = order[pos]
        bit = 1 << actual_idx
        
        # Option 1: Include actual_idx
        if can_add(actual_idx, current_mask):
            solve(pos + 1, current_mask | bit, current_size + 1)
            
        # Option 2: Exclude actual_idx
        solve(pos + 1, current_mask, current_size)
        
    # Start B&B
    # Initial state: empty set
    solve(0, 0, 0)
    
    if not best_set:
        return [1]
        
    return best_set

# EVOLVE-BLOCK-END
