# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses bitmask-based conflict detection for O(1) conflict checks, enabling
    significantly more local search iterations within the time budget.
    """
    m = 205
    
    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # For fast lookup, create a boolean array
    is_forbidden = np.zeros(m, dtype=bool)
    for d in forbidden:
        is_forbidden[d] = True
    
    # Build conflict masks: conflict_mask[i] is a bitmask of all j that conflict with i
    # Use Python int as arbitrary-precision bitmask
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if is_forbidden[d]:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Precompute: for each element, the set of elements it conflicts with (as a set for quick lookup)
    # Actually, we'll use the bitmask approach throughout
    
    def can_add_to_mask(selected_mask, candidate):
        """Check if candidate can be added to the set represented by selected_mask."""
        # candidate conflicts with any element in selected_mask if (conflict_mask[candidate] & selected_mask) != 0
        return (conflict_mask[candidate] & selected_mask) == 0
    
    def greedy_construct(order):
        """Greedy construction following a given order."""
        selected_mask = 0
        selected_list = []
        for candidate in order:
            if can_add_to_mask(selected_mask, candidate):
                selected_mask |= (1 << candidate)
                selected_list.append(candidate)
        return selected_list, selected_mask
    
    def local_search_improve(selected_list, selected_mask, max_iter=100000):
        """
        Local search with Add, Remove, and Swap moves using bitmask conflict checks.
        """
        selected_list = list(selected_list)
        selected_mask = selected_mask
        
        for _ in range(max_iter):
            improved = False
            
            # Try to add any element that doesn't conflict
            for a in range(m):
                if (selected_mask >> a) & 1:
                    continue
                if can_add_to_mask(selected_mask, a):
                    selected_mask |= (1 << a)
                    selected_list.append(a)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one, add one (only if it increases size or same size but different)
            # Actually, for maximizing size, we only want swaps that maintain or increase size
            # Let's try: for each r in selected, for each a not in selected, check if swap works
            # This is O(n^2) per iteration which is expensive. Let's be smarter.
            
            # Try 2-opt: remove r, add a where a was previously blocked by r
            # For each r in selected_list:
            #   For each a not in selected where a conflicts with r:
            #     Check if a can be added to selected_list minus r
            
            found_swap = False
            for r in selected_list:
                r_bit = 1 << r
                # Elements that conflict with r
                conflicting_with_r = conflict_mask[r]
                # Candidates that are not in selected and conflict with r
                candidates = conflicting_with_r & ~selected_mask
                # Try each candidate
                while candidates:
                    # Get lowest set bit
                    lsb = candidates & (-candidates)
                    a = lsb.bit_length() - 1
                    candidates ^= lsb
                    
                    # Check if a can be added to selected minus r
                    new_mask = selected_mask & ~r_bit
                    if can_add_to_mask(new_mask, a):
                        # Perform swap
                        selected_mask = new_mask | lsb
                        # Update list
                        selected_list.remove(r)
                        selected_list.append(a)
                        found_swap = True
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return selected_list, selected_mask
    
    def local_search_2swap(selected_list, selected_mask, max_iter=50000):
        """
        Local search with 2-swap moves: remove two, add two.
        """
        selected_list = list(selected_list)
        selected_mask = selected_mask
        n = len(selected_list)
        
        for _ in range(max_iter):
            improved = False
            
            # First try simple adds
            for a in range(m):
                if (selected_mask >> a) & 1:
                    continue
                if can_add_to_mask(selected_mask, a):
                    selected_mask |= (1 << a)
                    selected_list.append(a)
                    improved = True
                    break
            if improved:
                continue
            
            # Try 2-swap: remove r1, r2; add a1, a2
            # This is O(n^4) which is too expensive. Let's try a heuristic:
            # For each pair (r1, r2) in selected, try to find (a1, a2) not in selected
            # such that {a1, a2} can be added to selected - {r1, r2}
            
            # To make this tractable, we'll only try a limited number of pairs
            # and use a time check
            
            found_2swap = False
            sel_indices = list(range(n))
            rng_local = random.Random()
            
            # Try a limited number of random pairs to avoid O(n^4)
            num_pairs_to_try = min(200, len(sel_indices) * (len(sel_indices) - 1) // 2)
            pairs_tried = 0
            
            while pairs_tried < num_pairs_to_try:
                i1 = rng_local.randint(0, n - 1)
                i2 = rng_local.randint(0, n - 1)
                if i1 == i2:
                    continue
                r1 = selected_list[i1]
                r2 = selected_list[i2]
                if r1 == r2:
                    continue
                pairs_tried += 1
                
                r1_bit = 1 << r1
                r2_bit = 1 << r2
                new_mask = selected_mask & ~r1_bit & ~r2_bit
                
                # Find candidates that can be added to new_mask
                # We need two candidates a1, a2 such that both can be added to new_mask
                # and a1, a2 don't conflict with each other
                
                # Get all candidates that can be added to new_mask
                candidates = []
                for a in range(m):
                    if (selected_mask >> a) & 1:
                        continue
                    if can_add_to_mask(new_mask, a):
                        candidates.append(a)
                
                # Try pairs of candidates
                for idx1 in range(len(candidates)):
                    a1 = candidates[idx1]
                    a1_bit = 1 << a1
                    # Check if a1 conflicts with anything in new_mask (already checked)
                    # Now check if we can add a2 to new_mask | a1_bit
                    for idx2 in range(idx1 + 1, len(candidates)):
                        a2 = candidates[idx2]
                        a2_bit = 1 << a2
                        if can_add_to_mask(new_mask | a1_bit, a2):
                            # Perform 2-swap
                            selected_mask = new_mask | a1_bit | a2_bit
                            # Update list
                            selected_list.remove(r1)
                            selected_list.remove(r2)
                            selected_list.append(a1)
                            selected_list.append(a2)
                            found_2swap = True
                            improved = True
                            break
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return selected_list, selected_mask
    
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_solution = None
    best_size = 0
    best_mask = 0
    
    # Initialize with simple greedy
    initial_list, initial_mask = greedy_construct(list(range(m)))
    if len(initial_list) > best_size:
        best_size = len(initial_list)
        best_solution = initial_list[:]
        best_mask = initial_mask
    
    # Local search on initial
    improved_list, improved_mask = local_search_improve(initial_list, initial_mask)
    if len(improved_list) > best_size:
        best_size = len(improved_list)
        best_solution = improved_list[:]
        best_mask = improved_mask
    
    # Try 2-swap
    improved2_list, improved2_mask = local_search_2swap(improved_list, improved_mask)
    if len(improved2_list) > best_size:
        best_size = len(improved2_list)
        best_solution = improved2_list[:]
        best_mask = improved2_mask
    
    # Random restarts
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Random order
        order = list(range(m))
        rng.shuffle(order)
        solution_list, solution_mask = greedy_construct(order)
        
        # Local search
        solution_list, solution_mask = local_search_improve(solution_list, solution_mask)
        
        # 2-swap
        solution_list, solution_mask = local_search_2swap(solution_list, solution_mask)
        
        if len(solution_list) > best_size:
            best_size = len(solution_list)
            best_solution = solution_list[:]
            best_mask = solution_mask
        
        # Check time
        if time.time() > deadline - 30:
            break
    
    # Final local search on best solution
    if best_solution is not None and time.time() < deadline - 10:
        final_list, final_mask = local_search_improve(best_solution, best_mask)
        final_list, final_mask = local_search_2swap(final_list, final_mask)
        if len(final_list) > best_size:
            best_size = len(final_list)
            best_solution = final_list[:]
            best_mask = final_mask
    
    if best_solution is None:
        best_solution = [0]
    
    return np.array(best_solution, dtype=np.int64)

# EVOLVE-BLOCK-END
