import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Precompute LCMs for all pairs in [1, 128]
    # lcm(a, b) = a * b // gcd(a, b)
    N = 128
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = i * j // gcd(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Identify all bad triples in the full set
    # A triple (a,b,c) is bad if lcm(a,b)==lcm(a,c)==lcm(b,c)
    # We store for each number, the set of pairs that form a bad triple with it?
    # Or just check on the fly.
    
    # Heuristic: Count "risk" of each number in the full set
    # risk[n] = number of bad triples containing n
    risk = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                    risk[a] += 1
                    risk[b] += 1
                    risk[c] += 1
    
    best_set = set(range(1, N + 1)) # Placeholder, will be overwritten
    best_size = 0
    
    # Strategy 1: Greedy Add sorted by risk (ascending)
    # Strategy 2: Greedy Add sorted by omega (ascending)
    # Strategy 3: Random Greedy Add
    
    def omega(n):
        count = 0
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                count += 1
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count
    
    omega_vals = [0] * (N + 1)
    for i in range(1, N + 1):
        omega_vals[i] = omega(i)
        
    candidates = list(range(1, N + 1))
    
    def try_greedy(order):
        selected = []
        selected_set = set()
        for x in order:
            # Check if x can be added
            valid = True
            # We need to check if there exist a, b in selected such that
            # lcm(a,b) == lcm(a,x) == lcm(b,x)
            # Optimize: iterate pairs in selected
            # Since selected grows, this is O(K^2) per x
            # K is current size
            for i in range(len(selected)):
                a = selected[i]
                lax = lcm_table[a][x]
                for j in range(i + 1, len(selected)):
                    b = selected[j]
                    if lcm_table[a][b] == lax and lcm_table[b][x] == lax:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                selected.append(x)
                selected_set.add(x)
        return selected, len(selected)
    
    # Run multiple restarts with different orderings
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Randomize order within risk/omega groups
        # Or just random shuffle
        if num_restarts % 3 == 0:
            # Sort by risk
            order = sorted(candidates, key=lambda x: (risk[x], random.random()))
        elif num_restarts % 3 == 1:
            # Sort by omega
            order = sorted(candidates, key=lambda x: (omega_vals[x], random.random()))
        else:
            # Random
            order = candidates[:]
            random.shuffle(order)
            
        s, size = try_greedy(order)
        if size > best_size:
            best_size = size
            best_set = s
            # Early stop if we found a very large set?
            # Max possible is 128.
            if best_size > 100:
                break
                
    return best_set

# Fallback if something goes wrong (should not happen)
# But construct is the entry point.
