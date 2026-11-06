import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    
    # Generate all 3-term APs (a, a+d, a+2d) with 1 <= a, a+2d <= n
    # Represent each AP as a bitmask of its 3 elements
    # Also need to check 4-term AP conflicts
    
    # First, let's build the set of all possible 3-APs
    # A 3-AP is (x, y, z) where x+y = 2z, x < y < z, all in [1, n]
    # d = y - x = z - y > 0
    
    ap_list = []  # List of (a, d, mask) where mask is bitmask of {a, a+d, a+2d}
    ap_index = {}  # (a, d) -> index in ap_list
    
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            if c <= n:
                # mask for elements a, b, c
                mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
                idx = len(ap_list)
                ap_list.append((a, d, mask))
                ap_index[(a, d)] = idx
    
    num_aps = len(ap_list)
    
    # Build conflict graph: two 3-APs conflict if:
    # 1. They share at least one element (can't both be in the set)
    # 2. OR they can be extended to form a 4-term AP together
    # 
    # Actually, the constraint is: the final set A must have NO 4-term AP.
    # A 4-term AP is (x, x+d, x+2d, x+3d).
    # If A contains all of x, x+d, x+2d, x+3d, that's a 4-AP.
    # 
    # The 3-APs in A are the 3-APs fully contained in A.
    # We want to maximize the number of 3-APs in A, subject to A having no 4-AP.
    #
    # Alternative view: Select a subset of elements A ⊆ [1,n] such that:
    # - A contains no 4-term AP
    # - Maximize the number of 3-term APs fully contained in A
    #
    # The branch-and-bound on 3-APs is tricky because selecting a 3-AP 
    # doesn't directly determine the element set. Let me reconsider.
    #
    # Better approach: Branch-and-bound on the ELEMENT SET, but use the 
    # 3-AP structure for fast evaluation and pruning.
    #
    # Actually, let's use a different structural approach:
    # Use a greedy construction with lookahead, then improve via local search
    # with bitmask-based delta evaluation.
    #
    # Let me implement a hybrid: 
    # 1. Greedy construction (fast, gives good starting point)
    # 2. Local search with bitmask-based delta evaluation (fast per step)
    # 3. Simulated annealing polish
    #
    # This is materially different from the seed's pure greedy forward pass.
    
    # Bitmask representation: element i (1-indexed) -> bit (i-1)
    # We'll use Python integers as bitmasks (arbitrary precision)
    
    # Precompute for each element x, the set of 3-APs that contain x
    # And for fast 4-AP checking
    
    # For 4-AP checking: A contains a 4-AP iff there exist x, d such that
    # x, x+d, x+2d, x+3d are all in A.
    
    # Let's use a set-based approach for clarity, with bitmask for speed
    
    # Greedy construction: add elements that maximize 3-APs while avoiding 4-APs
    selected = set()
    selected_mask = 0
    
    # Precompute: for each x, which 4-APs would be completed if we add x
    # A 4-AP (a, a+d, a+2d, a+3d) is "completed" when all 4 elements are in A
    
    # For speed, let's track counts of partial 4-APs
    
    # Actually, let's use a simpler but effective approach:
    # 1. Start with greedy
    # 2. Local search: try adding/removing/swapping elements
    # 3. Use bitmask for fast 3-AP counting and 4-AP checking
    
    def count_3aps(mask):
        """Count 3-term APs in the set represented by mask."""
        count = 0
        # For each pair (x, z) with x < z and (x+z) even, check if (x+z)//2 is in set
        # This is O(n^2) which is ~65k for n=256, acceptable
        for d in range(1, n // 2 + 1):
            for a in range(1, n - 2 * d + 1):
                b = a + d
                c = a + 2 * d
                # Check if a, b, c are all in mask
                if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1 and (mask >> (c - 1)) & 1:
                    count += 1
        return count
    
    def has_4ap(mask):
        """Check if the set represented by mask contains a 4-term AP."""
        for d in range(1, n // 3 + 1):
            for a in range(1, n - 3 * d + 1):
                b = a + d
                c = a + 2 * d
                e = a + 3 * d
                if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1 and (mask >> (c - 1)) & 1 and (mask >> (e - 1)) & 1:
                    return True
        return False
    
    # Greedy: try to add each element if it doesn't create a 4-AP
    # Order elements by potential 3-AP contribution
    # Element x participates in 3-APs as first, middle, or third term
    
    # For greedy, just go in order and add if no 4-AP is created
    for x in range(1, n + 1):
        new_mask = selected_mask | (1 << (x - 1))
        if not has_4ap(new_mask):
            selected.add(x)
            selected_mask = new_mask
    
    best_score = count_3aps(selected_mask)
    best_mask = selected_mask
    best_set = set(selected)
    
    # Local search: try swaps and adds/removes
    # For speed, precompute the effect of adding/removing each element
    
    # Let's do a few rounds of local search
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try adding each element not in set
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if x in best_set:
                continue
            new_mask = best_mask | (1 << (x - 1))
            if has_4ap(new_mask):
                continue
            new_score = count_3aps(new_mask)
            if new_score > best_score:
                best_mask = new_mask
                best_score = new_score
                best_set.add(x)
                improved = True
        
        # Try removing each element in set
        for x in list(best_set):
            if time.time() > deadline:
                break
            new_mask = best_mask & ~(1 << (x - 1))
            new_score = count_3aps(new_mask)
            # Removing always maintains no-4-AP property
            if new_score > best_score:  # This shouldn't happen for pure removal
                best_mask = new_mask
                best_score = new_score
                best_set.remove(x)
                improved = True
        
        # Try swaps: remove one, add another
        if not improved:
            for x_remove in list(best_set):
                if time.time() > deadline:
                    break
                for x_add in range(1, n + 1):
                    if x_add in best_set:
                        continue
                    new_mask = (best_mask & ~(1 << (x_remove - 1))) | (1 << (x_add - 1))
                    if has_4ap(new_mask):
                        continue
                    new_score = count_3aps(new_mask)
                    if new_score > best_score:
                        best_mask = new_mask
                        best_score = new_score
                        best_set.remove(x_remove)
                        best_set.add(x_add)
                        improved = True
                        break
        
        if not improved:
            break
    
    # Simulated annealing polish
    if time.time() < deadline - 10:
        current_mask = best_mask
        current_score = best_score
        current_set = set(best_set)
        
        T = 5.0
        cooling = 0.95
        
        while time.time() < deadline - 5 and T > 0.1:
            # Random move: add, remove, or swap
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                candidates = [x for x in range(1, n + 1) if x not in current_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                new_mask = current_mask | (1 << (x - 1))
                if has_4ap(new_mask):
                    continue
                new_score = count_3aps(new_mask)
            elif move_type == 'remove':
                if not current_set:
                    continue
                x = random.choice(list(current_set))
                new_mask = current_mask & ~(1 << (x - 1))
                new_score = count_3aps(new_mask)
            else:  # swap
                if not current_set:
                    continue
                x_remove = random.choice(list(current_set))
                candidates = [x for x in range(1, n + 1) if x not in current_set]
                if not candidates:
                    continue
                x_add = random.choice(candidates)
                new_mask = (current_mask & ~(1 << (x_remove - 1))) | (1 << (x_add - 1))
                if has_4ap(new_mask):
                    continue
                new_score = count_3aps(new_mask)
            
            delta = new_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_mask = new_mask
                current_score = new_score
                # Update set
                if move_type == 'add':
                    current_set.add(x)
                elif move_type == 'remove':
                    current_set.remove(x)
                else:
                    current_set.remove(x_remove)
                    current_set.add(x_add)
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    best_set = set(current_set)
            
            T *= cooling
        
        # Final local search from best SA result
        for _ in range(10):
            improved = False
            for x in range(1, n + 1):
                if time.time() > deadline:
                    break
                if x in best_set:
                    continue
                new_mask = best_mask | (1 << (x - 1))
                if has_4ap(new_mask):
                    continue
                new_score = count_3aps(new_mask)
                if new_score > best_score:
                    best_mask = new_mask
                    best_score = new_score
                    best_set.add(x)
                    improved = True
            if not improved:
                break
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
