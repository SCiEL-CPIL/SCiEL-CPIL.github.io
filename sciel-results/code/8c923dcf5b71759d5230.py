import time
from math import gcd
import random

def construct():
    N = 128
    DEADLINE = time.time() + 55.0  # Safety margin
    
    # Precompute gcd
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_mat[i][j] = gcd(i, j)
            gcd_mat[j][i] = gcd_mat[i][j]
            
    # Precompute bad_pairs_with[x][u]
    # bad_pairs_with[x][u] is a bitmask of v such that {x, u, v} is a bad triple
    # Note: v must be distinct from x and u.
    bad_pairs_with = [[0] * (N + 1) for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        for u in range(1, N + 1):
            if x == u:
                continue
            mask = 0
            for v in range(1, N + 1):
                if v == x or v == u:
                    continue
                if gcd_mat[x][u] == gcd_mat[x][v] == gcd_mat[u][v]:
                    mask |= (1 << v)
            bad_pairs_with[x][u] = mask
            # Symmetry: bad_pairs_with[u][x] should be related?
            # The triple {x,u,v} is symmetric.
            # bad_pairs_with[u][x] is mask of w such that {u,x,w} is bad.
            # This is the same set of v's.
            # So bad_pairs_with[u][x] should have the same mask?
            # Wait. bad_pairs_with[x][u] is set of v.
            # bad_pairs_with[u][x] is set of w.
            # The set of third elements is the same.
            # So yes, bad_pairs_with[x][u] == bad_pairs_with[u][x].
            # Let's just compute it once for x < u and copy.
            
    # Optimization: Compute only for x < u and copy
    for x in range(1, N + 1):
        for u in range(x + 1, N + 1):
            mask = 0
            for v in range(1, N + 1):
                if v == x or v == u:
                    continue
                if gcd_mat[x][u] == gcd_mat[x][v] == gcd_mat[u][v]:
                    mask |= (1 << v)
            bad_pairs_with[x][u] = mask
            bad_pairs_with[u][x] = mask

    def can_add(x, S_mask):
        """Check if x can be added to S (represented by S_mask)."""
        # For each u in S, check if S has any v such that {x, u, v} is bad.
        # i.e., S_mask & bad_pairs_with[x][u] != 0
        temp = S_mask
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            if S_mask & bad_pairs_with[x][u]:
                return False
            temp -= lsb
        return True

    def greedy_build(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        S_mask = 0
        remaining = list(range(1, N + 1))
        random.shuffle(remaining)
        
        # Sort by degree (number of conflicts) to prefer low-degree nodes
        # Degree of x: number of pairs (u,v) such that {x,u,v} is bad.
        # This is roughly sum(popcount(bad_pairs_with[x][u]) for u in [1..N]) / 2?
        # Actually, just count how many v's are in bad_pairs_with[x][u] for all u.
        # Or simpler: just shuffle and add if possible.
        # Better: sort by global conflict count.
        
        conflict_count = [0] * (N + 1)
        for x in range(1, N + 1):
            count = 0
            for u in range(1, N + 1):
                if x != u:
                    count += bin(bad_pairs_with[x][u]).count('1')
            conflict_count[x] = count
            
        remaining.sort(key=lambda x: (conflict_count[x], random.random()))
        
        for x in remaining:
            if time.time() > DEADLINE:
                break
            if can_add(x, S_mask):
                S_mask |= (1 << x)
                
        return S_mask

    def local_search(S_mask):
        """
        Local search to maximize popcount(S_mask).
        1. Try to add any element.
        2. If stuck, try swaps that enable an addition.
        3. If stuck, try random swaps/removals to escape.
        """
        current = S_mask
        best = S_mask
        best_size = bin(current).count('1')
        
        all_elems_mask = (1 << (N + 1)) - 2 # Bits 1..N set
        
        while time.time() < DEADLINE - 2:
            improved = False
            
            # Phase 1: Try to add any element
            # We iterate through elements not in current
            not_in_S = all_elems_mask & ~current
            while not_in_S:
                lsb = not_in_S & -not_in_S
                x = lsb.bit_length() - 1
                if can_add(x, current):
                    current |= lsb
                    not_in_S &= ~lsb
                    improved = True
                    # Update best
                    if bin(current).count('1') > best_size:
                        best_size = bin(current).count('1')
                        best = current
                else:
                    not_in_S &= ~lsb
            
            if improved:
                continue # Restart addition phase with new set
                
            # Phase 2: Try swaps that lead to an addition
            # This is expensive. We try a limited number of random swaps.
            # Or systematic?
            # Let's try random swaps.
            
            # Get list of elements in S
            S_list = []
            temp = current
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                S_list.append(u)
                temp -= lsb
                
            # Get list of elements not in S
            not_in_S_list = []
            temp = not_in_S
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                not_in_S_list.append(v)
                temp -= lsb
                
            if not S_list or not not_in_S_list:
                break
                
            # Try a few random swaps
            swap_found = False
            for _ in range(20): # Limit swaps per step
                if time.time() > DEADLINE - 2:
                    break
                u = random.choice(S_list)
                v = random.choice(not_in_S_list)
                
                # Try removing u and adding v
                temp_current = current & ~(1 << u)
                if can_add(v, temp_current):
                    # Check if this swap allows an addition afterwards?
                    # Or just accept it as a move?
                    # If we accept it, size is same.
                    # We need to check if it leads to improvement.
                    # Let's tentatively make the swap and see if we can add anything.
                    new_current = temp_current | (1 << v)
                    
                    # Try to add from new_current
                    not_in_new = all_elems_mask & ~new_current
                    added = False
                    while not_in_new:
                        lsb = not_in_new & -not_in_new
                        w = lsb.bit_length() - 1
                        if can_add(w, new_current):
                            new_current |= lsb
                            not_in_new &= ~lsb
                            added = True
                        else:
                            not_in_new &= ~lsb
                            
                    if added:
                        current = new_current
                        if bin(current).count('1') > best_size:
                            best_size = bin(current).count('1')
                            best = current
                        swap_found = True
                        break # Restart main loop with new current
                        
            if swap_found:
                continue
                
            # Phase 3: Escape local optimum
            # Accept a random swap or removal with some probability
            # Since we want to maximize size, removal is bad.
            # Swap is neutral.
            # Let's just do a random swap to change the landscape.
            if random.random() < 0.5:
                u = random.choice(S_list)
                v = random.choice(not_in_S_list)
                current = (current & ~(1 << u)) | (1 << v)
            else:
                # Try 2-swap?
                pass
                
        return best

    # Run multiple trials
    best_global = 0
    best_global_mask = 0
    
    # Initial greedy
    initial_mask = greedy_build(123)
    best_global_mask = local_search(initial_mask)
    best_global = bin(best_global_mask).count('1')
    
    # More randomized starts
    for i in range(20):
        if time.time() > DEADLINE - 5:
            break
        mask = greedy_build(1000 + i)
        # Quick local search
        improved_mask = local_search(mask)
        size = bin(improved_mask).count('1')
        if size > best_global:
            best_global = size
            best_global_mask = improved_mask

    # Convert mask to list
    result = []
    temp = best_global_mask
    while temp:
        lsb = temp & -temp
        x = lsb.bit_length() - 1
        result.append(x)
        temp -= lsb
        
    if not result:
        result = [1]
        
    return sorted(result)
