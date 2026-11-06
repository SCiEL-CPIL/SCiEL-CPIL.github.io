"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 128
    
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    # Precompute for each x and each gcd value d, the bitmask of elements y where gcd(x,y) = d
    # gcd_mask[x][d] = bitmask of y in [1,N] where gcd(x,y) = d
    max_gcd = N
    gcd_mask = [[0]*(max_gcd+1) for _ in range(N+1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            d = gcd_table[x][y]
            gcd_mask[x][d] |= (1 << y)
    
    # Precompute for each d, the list of pairs (a,b) with gcd(a,b) = d
    pairs_by_gcd = defaultdict(list)
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            d = gcd_table[a][b]
            pairs_by_gcd[d].append((a, b))
    
    # For fast checking: for each x and d, precompute which pairs from pairs_by_gcd[d]
    # have both elements in gcd_mask[x][d]
    # conflict_pairs[x][d] = list of (a,b) pairs where gcd(a,b)=d and gcd(x,a)=d and gcd(x,b)=d
    # But we can check on the fly using bitmasks
    
    def can_add_bitmask(x, selected_mask):
        """Check if x can be added to set represented by selected_mask."""
        for d in range(1, max_gcd+1):
            mask_x_d = gcd_mask[x][d]
            # Elements in selected set that have gcd(x, y) = d
            candidates = selected_mask & mask_x_d
            if candidates == 0:
                continue
            # Check if any pair (a,b) in pairs_by_gcd[d] has both a and b in candidates
            for a, b in pairs_by_gcd[d]:
                if (candidates & (1 << a)) and (candidates & (1 << b)):
                    return False
        return True
    
    def greedy_bitmask(order):
        """Run greedy algorithm with given order using bitmask."""
        selected_mask = 0
        for x in order:
            if can_add_bitmask(x, selected_mask):
                selected_mask |= (1 << x)
        return selected_mask
    
    def mask_to_list(mask):
        """Convert bitmask to list of elements."""
        result = []
        for i in range(1, N+1):
            if mask & (1 << i):
                result.append(i)
        return result
    
    def local_search_bitmask(selected_mask, time_limit):
        """Local search using bitmask representation."""
        best_mask = selected_mask
        best_size = bin(best_mask).count('1')
        
        while time.time() < time_limit:
            improved = False
            
            # Try adding elements not in the set
            for x in range(1, N+1):
                if best_mask & (1 << x):
                    continue
                if can_add_bitmask(x, best_mask):
                    best_mask |= (1 << x)
                    best_size += 1
                    improved = True
                    break
            if improved:
                continue
            
            # Try swapping: remove one, try to add two
            for i in range(1, N+1):
                if not (best_mask & (1 << i)):
                    continue
                temp_mask = best_mask & ~(1 << i)
                added = 0
                for x in range(1, N+1):
                    if temp_mask & (1 << x):
                        continue
                    if x == i:
                        continue
                    if can_add_bitmask(x, temp_mask):
                        temp_mask |= (1 << x)
                        added += 1
                        if added >= 2:
                            break
                if added > 1:
                    best_mask = temp_mask
                    best_size += added - 1
                    improved = True
                    break
            
            if not improved:
                break
        
        return best_mask
    
    # Try multiple orderings
    best_mask = 0
    
    # Ordering 1: ascending
    result_mask = greedy_bitmask(range(1, N+1))
    if bin(result_mask).count('1') > bin(best_mask).count('1'):
        best_mask = result_mask
    
    if time.time() > deadline - 20:
        return mask_to_list(best_mask) if best_mask else [1]
    
    # Ordering 2: descending
    result_mask = greedy_bitmask(range(N, 0, -1))
    if bin(result_mask).count('1') > bin(best_mask).count('1'):
        best_mask = result_mask
    
    if time.time() > deadline - 15:
        return mask_to_list(best_mask) if best_mask else [1]
    
    # Ordering 3: by number of prime factors (fewer first)
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    order3 = sorted(range(1, N+1), key=lambda x: (num_prime_factors(x), x))
    result_mask = greedy_bitmask(order3)
    if bin(result_mask).count('1') > bin(best_mask).count('1'):
        best_mask = result_mask
    
    if time.time() > deadline - 10:
        return mask_to_list(best_mask) if best_mask else [1]
    
    # Try several random orderings
    rng = random.Random(42)
    for trial in range(30):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N+1))
        rng.shuffle(order)
        result_mask = greedy_bitmask(order)
        if bin(result_mask).count('1') > bin(best_mask).count('1'):
            best_mask = result_mask
    
    if time.time() > deadline - 3:
        return mask_to_list(best_mask) if best_mask else [1]
    
    # Local search on best result
    best_mask = local_search_bitmask(best_mask, deadline - 1)
    
    return mask_to_list(best_mask) if best_mask else [1]
# EVOLVE-BLOCK-END
