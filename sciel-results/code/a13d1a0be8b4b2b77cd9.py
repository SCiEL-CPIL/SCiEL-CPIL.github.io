import time
import random
from itertools import combinations

def construct():
    n = 512
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # Precompute all 4-APs: list of tuples (a, b, c, d) where a < b < c < d and they form a 4-AP
    # A 4-AP is (a, a+d, a+2d, a+3d)
    all_4aps = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            if dd <= n:
                all_4aps.append((a, b, c, dd))
    
    # For each element, precompute which 4-APs it's in
    elem_to_aps = {x: [] for x in range(1, n + 1)}
    for idx, ap in enumerate(all_4aps):
        for x in ap:
            elem_to_aps[x].append(idx)
    
    # For each element, precompute the set of other elements that share a 4-AP with it
    # conflict_mask[x] = bitmask of elements y such that x and y are in some 4-AP
    # Actually, we need: if we add x, which existing elements would create a 4-AP?
    # Better: for each x, precompute the set of "forbidden pairs" - if x is in S,
    # then for each 4-AP containing x, the other 3 elements cannot all be in S.
    
    # Let's use a different approach: for each element x, precompute all 4-APs containing x.
    # When checking if we can add x to current set S, we check: for each 4-AP containing x,
    # are the other 3 elements already in S?
    
    # Precompute: for each x, list of (other1, other2, other3) for each 4-AP containing x
    x_to_other_triples = {x: [] for x in range(1, n + 1)}
    for ap in all_4aps:
        for x in ap:
            others = tuple(y for y in ap if y != x)
            x_to_other_triples[x].append(others)
    
    # Convert to 0-indexed for bitmask
    # Element x (1-indexed) -> bit (x-1)
    
    def check_can_add(bs, x):
        """Check if element x (1-indexed) can be added to set represented by bitmask bs."""
        for others in x_to_other_triples[x]:
            # Check if all 3 others are in bs
            if ((bs >> (others[0] - 1) & 1) and 
                (bs >> (others[1] - 1) & 1) and 
                (bs >> (others[2] - 1) & 1)):
                return False
        return True
    
    # Base-3 construction for initial lower bound
    def base3_construction():
        S = set()
        for x in range(1, n + 1):
            temp = x
            valid = True
            while temp > 0:
                if temp % 3 == 2:
                    valid = False
                    break
                temp //= 3
            if valid:
                S.add(x)
        return S
    
    # Get initial solution from base-3
    init_S = base3_construction()
    init_bs = 0
    for x in init_S:
        init_bs |= (1 << (x - 1))
    best_size = len(init_S)
    best_bs = init_bs
    
    # Order elements by degree (number of 4-APs they're in), descending
    # Elements with higher degree are more constraining, so decide on them first
    degree = {x: len(x_to_other_triples[x]) for x in range(1, n + 1)}
    order = sorted(range(1, n + 1), key=lambda x: -degree[x])
    
    # Branch and bound
    # State: current bitmask, current size, index in order
    # We process elements in 'order' one by one
    # For each element, we try: include it (if possible) or exclude it
    
    # Upper bound: current_size + number of remaining elements that could potentially be added
    # Simple bound: current_size + (total_remaining)
    # Better bound: current_size + (number of remaining elements not yet excluded)
    
    # For pruning, we can use: if current_size + remaining_count <= best_size, prune
    
    # Let's use a recursive approach with iterative deepening on time
    
    def branch_and_bound():
        nonlocal best_size, best_bs, deadline
        
        # Process elements in order
        # We'll use an iterative approach with a stack to avoid recursion depth issues
        # State: (bs, size, idx) where idx is index into order
        # But this could explode. Let's use a smarter approach.
        
        # Actually, let's do a greedy + local search hybrid with the branch-and-bound
        # for the most constrained elements first.
        
        # Phase 1: Greedy fill in order of decreasing degree
        bs = 0
        size = 0
        for x in order:
            if check_can_add(bs, x):
                bs |= (1 << (x - 1))
                size += 1
        
        if size > best_size:
            best_size = size
            best_bs = bs
        
        # Phase 2: Try to improve by swapping
        # For each element not in S, check if removing one element from S and adding it improves
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find all elements not in S
            not_in_S = [x for x in range(1, n + 1) if not (bs >> (x - 1) & 1)]
            random.shuffle(not_in_S)
            
            for x_new in not_in_S:
                if time.time() > deadline:
                    break
                # Find which elements in S conflict with x_new
                # i.e., for each 4-AP containing x_new, if the other 3 are in S, 
                # then we need to remove at least one of them
                # Actually, x_new can't be added because some 4-AP is fully satisfied
                # Let's find minimal set of elements to remove to allow x_new
                
                # For each 4-AP containing x_new, check if other 3 are in S
                blocking_aps = []
                for others in x_to_other_triples[x_new]:
                    if ((bs >> (others[0] - 1) & 1) and 
                        (bs >> (others[1] - 1) & 1) and 
                        (bs >> (others[2] - 1) & 1)):
                        blocking_aps.append(others)
                
                if not blocking_aps:
                    # Can add directly
                    bs |= (1 << (x_new - 1))
                    size += 1
                    if size > best_size:
                        best_size = size
                        best_bs = bs
                    improved = True
                    break
                
                # For each blocking AP, we need to remove at least one element
                # Try removing one element from the first blocking AP
                # Then check if x_new can be added, and if we can add more elements
                for others in blocking_aps:
                    if time.time() > deadline:
                        break
                    for elem_to_remove in others:
                        # Try removing elem_to_remove
                        temp_bs = bs & ~(1 << (elem_to_remove - 1))
                        if check_can_add(temp_bs, x_new):
                            temp_bs |= (1 << (x_new - 1))
                            # Now try to greedily add more elements
                            temp_size = size - 1 + 1  # removed 1, added 1
                            for x2 in order:
                                if not (temp_bs >> (x2 - 1) & 1) and check_can_add(temp_bs, x2):
                                    temp_bs |= (1 << (x2 - 1))
                                    temp_size += 1
                            if temp_size > size:
                                bs = temp_bs
                                size = temp_size
                                if size > best_size:
                                    best_size = size
                                    best_bs = bs
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        
        return best_size, best_bs
    
    # Run branch and bound
    best_size, best_bs = branch_and_bound()
    
    # Additional local search: try multiple random restarts with perturbation
    for restart in range(200):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 1000 + 42)
        
        # Start from best solution, remove k random elements
        k = 2 + (restart % 6)
        elems_in_S = [x for x in range(1, n + 1) if (best_bs >> (x - 1) & 1)]
        rng.shuffle(elems_in_S)
        
        temp_bs = best_bs
        temp_size = best_size
        for e in elems_in_S[:k]:
            temp_bs &= ~(1 << (e - 1))
            temp_size -= 1
        
        # Greedy refill in order of degree
        for x in order:
            if not (temp_bs >> (x - 1) & 1) and check_can_add(temp_bs, x):
                temp_bs |= (1 << (x - 1))
                temp_size += 1
        
        if temp_size > best_size:
            best_size = temp_size
            best_bs = temp_bs
    
    # Final local search: try to add any remaining elements
    for x in order:
        if not (best_bs >> (x - 1) & 1) and check_can_add(best_bs, x):
            best_bs |= (1 << (x - 1))
            best_size += 1
    
    # Extract result
    result = [x for x in range(1, n + 1) if (best_bs >> (x - 1) & 1)]
    
    return sorted(result)

# EVOLVE-BLOCK-START
# The construct function is defined above. This block is for structural clarity.
# The entry point is construct() which returns a sorted list of integers.
# EVOLVE-BLOCK-END
