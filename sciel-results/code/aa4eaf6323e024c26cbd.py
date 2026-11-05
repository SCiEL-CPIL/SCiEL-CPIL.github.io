import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    n = 512
    
    # Precompute all 4-APs
    # For each x, store a list of masks representing the other 3 elements of APs containing x
    conflict_triples = [[] for _ in range(n + 1)]
    
    # Generate all 4-APs (a, a+d, a+2d, a+3d)
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            term4 = a + 3 * d
            
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[a].append(mask_a)
            
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[b].append(mask_b)
            
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (term4 - 1))
            conflict_triples[c].append(mask_c)
            
            mask_d = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflict_triples[term4].append(mask_d)
    
    # Convert to tuples for faster iteration
    conflict_triples = [tuple(lst) for lst in conflict_triples]
    
    def can_add(x, S_mask):
        """Check if x can be added to S without forming a 4-AP."""
        for mask in conflict_triples[x]:
            if (mask & S_mask) == mask:
                return False
        return True
    
    def popcount(x):
        return x.bit_count()
    
    def try_add_all(S_mask, S_list):
        """Try to add all possible elements. Returns (new_mask, new_list, added_count)."""
        added = 0
        for x in range(1, n + 1):
            if not (S_mask & (1 << (x - 1))):
                if can_add(x, S_mask):
                    S_mask |= (1 << (x - 1))
                    S_list.append(x)
                    added += 1
        return S_mask, S_list, added
    
    best_mask = 0
    best_size = 0
    
    while time.time() < deadline:
        # 1. Greedy Construction with random shuffle
        S_mask = 0
        S_list = []
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S_mask):
                S_mask |= (1 << (x - 1))
                S_list.append(x)
        
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
        
        # 2. Exhaustive add pass
        S_mask, S_list, _ = try_add_all(S_mask, S_list)
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
        
        # 3. Local search with swaps
        # Maintain S_list for O(1) random element selection
        # After each swap, re-run add to exploit freed slots
        max_swap_rounds = 3000
        for _ in range(max_swap_rounds):
            if time.time() >= deadline:
                break
            
            if not S_list:
                break
            
            x = random.randint(1, n)
            if S_mask & (1 << (x - 1)):
                continue
            
            # Pick y in S uniformly
            y = random.choice(S_list)
            
            # Try swap: remove y, add x
            S_mask_without_y = S_mask & ~(1 << (y - 1))
            
            if can_add(x, S_mask_without_y):
                # Accept swap
                S_mask = S_mask_without_y | (1 << (x - 1))
                # Update S_list: replace y with x
                S_list[S_list.index(y)] = x
                
                # After swap, try to add more elements
                S_mask, S_list, added = try_add_all(S_mask, S_list)
                current_size = len(S_list)
                if current_size > best_size:
                    best_size = current_size
                    best_mask = S_mask
    
    # Convert best_mask to sorted list
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(i + 1)
    
    if not result:
        return [1, 2]
        
    return result
