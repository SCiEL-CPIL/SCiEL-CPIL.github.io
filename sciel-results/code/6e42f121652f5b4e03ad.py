import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    n = 512
    k = 4

    # --- Precomputation: Conflict Masks for 4-APs ---
    # conflict_masks[x] is a list of bitmasks.
    # If (S_mask & m) == m, then x cannot be added to S_mask without creating a 4-AP.
    conflict_masks = [[] for _ in range(n + 1)]
    
    # Generate all 4-term APs (a, a+d, a+2d, a+3d)
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # Element a is blocked if b, c, e are in S
            conflict_masks[a].append((1 << b) | (1 << c) | (1 << e))
            
            # Element b is blocked if a, c, e are in S
            conflict_masks[b].append((1 << a) | (1 << c) | (1 << e))
            
            # Element c is blocked if a, b, e are in S
            conflict_masks[c].append((1 << a) | (1 << b) | (1 << e))
            
            # Element e is blocked if a, b, c are in S
            conflict_masks[e].append((1 << a) | (1 << b) | (1 << c))

    def is_valid_add(x, S_mask):
        """Check if adding x to S_mask creates a 4-AP."""
        for m in conflict_masks[x]:
            if (S_mask & m) == m:
                return False
        return True

    def popcount(x):
        return bin(x).count('1')

    # --- Phase 1: Initial Construction using Generalized Cantor Sets ---
    # The standard Cantor set (base 3, digits 0,1) is 3-AP free, hence 4-AP free.
    # However, it is very sparse (~n^0.63).
    # We can try generalized Cantor sets:
    # 1. Base 4, digits 0,1. Is it 4-AP free?
    #    If x, x+d, x+2d, x+3d have base-4 digits 0,1, do they form a 4-AP?
    #    This is not guaranteed to be 4-AP free.
    # 2. A safer approach: Use the base-3 Cantor set as a seed, but also try
    #    a "greedy" construction that prioritizes elements with fewer conflicts.
    # 3. Actually, a known construction for 4-AP free sets is the set of numbers
    #    whose base-3 representation has no digit 2 AND no digit 1? No.
    
    # Let's stick to the proven base-3 Cantor set (digits 0,1) as one seed.
    # And also try a "density-weighted" greedy as another seed.
    
    seeds = []
    
    # Seed 1: Base-3 Cantor Set (digits 0,1)
    S_cantor = 0
    for i in range(1, n + 1):
        valid = True
        temp = i
        while temp > 0:
            if temp % 3 == 2:
                valid = False
                break
            temp //= 3
        if valid:
            S_cantor |= (1 << i)
    seeds.append(S_cantor)
    
    # Seed 2: Greedy Forward (deterministic)
    S_greedy_fwd = 0
    for x in range(1, n + 1):
        if is_valid_add(x, S_greedy_fwd):
            S_greedy_fwd |= (1 << x)
    seeds.append(S_greedy_fwd)
    
    # Seed 3: Greedy Reverse (deterministic)
    S_greedy_rev = 0
    for x in range(n, 0, -1):
        if is_valid_add(x, S_greedy_rev):
            S_greedy_rev |= (1 << x)
    seeds.append(S_greedy_rev)
    
    # Seed 4: Random Greedy (a few restarts)
    for _ in range(10):
        S_rand = 0
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        for x in candidates:
            if is_valid_add(x, S_rand):
                S_rand |= (1 << x)
        seeds.append(S_rand)
        
    # Keep the best seed
    best_seed = max(seeds, key=popcount)
    cur_mask = best_seed
    best_mask = cur_mask
    cur_size = popcount(cur_mask)
    best_size = cur_size

    # --- Phase 2: Local Search ---
    # We will try to add elements. If stuck, try swaps.
    
    def get_unselected(S_mask):
        return [i for i in range(1, n + 1) if not (S_mask & (1 << i))]
        
    def get_selected(S_mask):
        return [i for i in range(1, n + 1) if (S_mask & (1 << i))]
    
    # Local Search Loop
    while time.time() < deadline - 2:
        improved = False
        
        # Try to add any valid element
        unselected = get_unselected(cur_mask)
        random.shuffle(unselected)
        
        for x in unselected:
            if is_valid_add(x, cur_mask):
                cur_mask |= (1 << x)
                cur_size += 1
                if cur_size > best_size:
                    best_size = cur_size
                    best_mask = cur_mask
                improved = True
                break
        
        if improved:
            continue
            
        # If no additions, try 1-for-1 swaps (neutral) to escape local optimum
        selected = get_selected(cur_mask)
        if not selected:
            break
            
        found_swap = False
        # Limit the number of swap attempts per iteration to avoid spending too much time
        swap_attempts = 0
        while swap_attempts < 50 and time.time() < deadline - 2:
            swap_attempts += 1
            y = random.choice(selected)
            temp_mask = cur_mask & ~(1 << y)
            
            # Get unselected elements in temp_mask
            unselected_in_temp = [i for i in range(1, n + 1) if not (temp_mask & (1 << i)) and i != y]
            
            if not unselected_in_temp:
                continue
                
            x = random.choice(unselected_in_temp)
            
            if is_valid_add(x, temp_mask):
                cur_mask = temp_mask | (1 << x)
                # Size unchanged
                found_swap = True
                break
        
        if found_swap:
            continue
            
        # If stuck, try 1-for-2 swaps (increase size by 1)
        found_1for2 = False
        swap_attempts_1for2 = 0
        while swap_attempts_1for2 < 50 and time.time() < deadline - 2:
            swap_attempts_1for2 += 1
            y = random.choice(selected)
            temp_mask = cur_mask & ~(1 << y)
            
            unselected_in_temp = [i for i in range(1, n + 1) if not (temp_mask & (1 << i)) and i != y]
            
            if len(unselected_in_temp) < 2:
                continue
                
            x1, x2 = random.sample(unselected_in_temp, 2)
            
            if is_valid_add(x1, temp_mask):
                temp_mask1 = temp_mask | (1 << x1)
                if is_valid_add(x2, temp_mask1):
                    cur_mask = temp_mask1 | (1 << x2)
                    cur_size += 1
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur_mask
                    found_1for2 = True
                    break
        
        if found_1for2:
            continue
        
        # If still stuck, kick (remove a few random elements) to escape local optimum
        if selected:
            kick_size = min(3, len(selected))
            kicks = random.sample(selected, kick_size)
            for k in kicks:
                cur_mask &= ~(1 << k)
            cur_size -= kick_size

    # Final Verification
    result = [i for i in range(1, n + 1) if (best_mask & (1 << i))]
    
    # Double check validity
    S_set = set(result)
    valid = True
    for a in S_set:
        for d in range(1, (n - a) // 3 + 1):
            if (a+d) in S_set and (a+2*d) in S_set and (a+3*d) in S_set:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback: Greedy
        S_mask = 0
        for x in range(1, n+1):
            if is_valid_add(x, S_mask):
                S_mask |= (1 << x)
        result = [i for i in range(1, n + 1) if (S_mask & (1 << i))]
        
    return sorted(result)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    S = set(res)
    valid = True
    for a in S:
        for d in range(1, (512 - a) // 3 + 1):
            if (a+d) in S and (a+2*d) in S and (a+3*d) in S:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
