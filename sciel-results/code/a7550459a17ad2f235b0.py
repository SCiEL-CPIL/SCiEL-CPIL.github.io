import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 128
    
    # Precompute LCM table for O(1) lookup
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            lcm_table[a][b] = a // gcd(a, b) * b
    
    # Identify all bad triples: (a, b, c) with a < b < c such that
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == L_ab and lcm_table[b][c] == L_ab:
                    bad_triples.append((a, b, c))
    
    # For each element x, store the set of conflicting pairs.
    # conflict_pairs[x] contains (u, v) such that {u, v, x} is a bad triple.
    # If both u and v are in the set, x cannot be added.
    conflict_pairs = [set() for _ in range(n + 1)]
    for a, b, c in bad_triples:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))
    
    # Precompute conflict list as lists for faster iteration
    conflict_lists = [list(conflict_pairs[i]) for i in range(n + 1)]
    
    # Precompute degrees
    degrees = [len(conflict_lists[i]) for i in range(n + 1)]
    
    def can_add(x, mask):
        """Check if element x can be added to the set represented by mask."""
        for u, v in conflict_lists[x]:
            if (mask >> (u - 1)) & 1 and (mask >> (v - 1)) & 1:
                return False
        return True
    
    def popcount(mask):
        return bin(mask).count('1')
    
    def greedy_add(order, mask):
        """Try to add elements from order to mask in sequence."""
        for x in order:
            if not (mask >> (x - 1)) & 1:
                if can_add(x, mask):
                    mask |= (1 << (x - 1))
        return mask
    
    def local_search(mask, max_iters=500):
        """
        Local search with 1-add, 1-remove, and 2-swap moves.
        """
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 3:
            improved = False
            iters += 1
            
            # 1. Greedy add pass
            for x in range(1, n + 1):
                if not (mask >> (x - 1)) & 1:
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        improved = True
            
            # 2. Try 1-remove + greedy refill
            if not improved:
                s_list = [i for i in range(1, n + 1) if (mask >> (i - 1)) & 1]
                if s_list:
                    # Try removing each element (or a random subset if too many)
                    if len(s_list) > 30:
                        candidates = random.sample(s_list, 30)
                    else:
                        candidates = s_list
                    
                    for x in candidates:
                        if time.time() > deadline - 3:
                            break
                        mask_no_x = mask & ~(1 << (x - 1))
                        new_mask = greedy_add(range(1, n + 1), mask_no_x)
                        if popcount(new_mask) > popcount(mask):
                            mask = new_mask
                            improved = True
                            break
            
            # 3. Try 2-swap: remove x, add y
            if not improved:
                s_list = [i for i in range(1, n + 1) if (mask >> (i - 1)) & 1]
                not_s_list = [i for i in range(1, n + 1) if not (mask >> (i - 1)) & 1]
                
                if s_list and not_s_list:
                    # Random sampling for swaps
                    for _ in range(50):
                        if time.time() > deadline - 3:
                            break
                        x = random.choice(s_list)
                        y = random.choice(not_s_list)
                        mask_swap = (mask & ~(1 << (x - 1))) | (1 << (y - 1))
                        # Verify validity: check all elements in mask_swap
                        valid = True
                        for elem in range(1, n + 1):
                            if (mask_swap >> (elem - 1)) & 1:
                                if not can_add(elem, mask_swap & ~(1 << (elem - 1))):
                                    valid = False
                                    break
                        if valid and popcount(mask_swap) == popcount(mask):
                            # Same size, but might enable further additions
                            new_mask = greedy_add(range(1, n + 1), mask_swap)
                            if popcount(new_mask) > popcount(mask):
                                mask = new_mask
                                improved = True
                                break
        
        return mask
    
    def two_opt(mask):
        """
        More aggressive local search: try all pairs of removals + refill.
        """
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            s_list = [i for i in range(1, n + 1) if (mask >> (i - 1)) & 1]
            
            # First do greedy add
            for x in range(1, n + 1):
                if not (mask >> (x - 1)) & 1:
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        improved = True
            
            if not improved and len(s_list) >= 2:
                # Try removing pairs
                if len(s_list) > 20:
                    # Sample pairs
                    for _ in range(100):
                        if time.time() > deadline - 3:
                            break
                        x, y = random.sample(s_list, 2)
                        mask_no_xy = mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
                        new_mask = greedy_add(range(1, n + 1), mask_no_xy)
                        if popcount(new_mask) > popcount(mask):
                            mask = new_mask
                            improved = True
                            break
                else:
                    for i in range(len(s_list)):
                        for j in range(i + 1, len(s_list)):
                            if time.time() > deadline - 3:
                                break
                            x, y = s_list[i], s_list[j]
                            mask_no_xy = mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
                            new_mask = greedy_add(range(1, n + 1), mask_no_xy)
                            if popcount(new_mask) > popcount(mask):
                                mask = new_mask
                                improved = True
                                break
                        if improved:
                            break
        
        return mask
    
    best_mask = 0
    best_size = 0
    
    # Multi-start with various strategies
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        r = random.random()
        
        if r < 0.3:
            # Random order
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif r < 0.5:
            # Low degree first
            order = sorted(range(1, n + 1), key=lambda x: degrees[x])
        elif r < 0.7:
            # High degree first
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x])
        elif r < 0.85:
            # Weighted random: favor low-degree elements
            weights = [1.0 / (1 + degrees[x]) for x in range(1, n + 1)]
            order = list(range(1, n + 1))
            for i in range(n - 1, 0, -1):
                total_w = sum(weights[j] for j in range(i + 1))
                r_val = random.random() * total_w
                cum = 0
                for j in range(i + 1):
                    cum += weights[j]
                    if cum >= r_val:
                        order[i], order[j] = order[j], order[i]
                        break
        else:
            # Start from a random subset and fill
            mask = 0
            for x in range(1, n + 1):
                if random.random() < 0.3:
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
            order = list(range(1, n + 1))
            random.shuffle(order)
        
        if r >= 0.85:
            mask = greedy_add(order, mask)
        else:
            mask = greedy_add(order, 0)
        
        # Local search
        mask = local_search(mask)
        
        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask
            
            # If we found a new best, do a more aggressive polish
            if size >= 100 and time.time() < deadline - 10:
                best_mask = two_opt(best_mask)
                best_size = popcount(best_mask)
    
    # Final polish with two_opt
    if time.time() < deadline - 2:
        best_mask = two_opt(best_mask)
        best_size = popcount(best_mask)
    
    result = [i for i in range(1, n + 1) if (best_mask >> (i - 1)) & 1]
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result


if __name__ == "__main__":
    res = construct()
    print(f"Result size: {len(res)}")
    print(res)
