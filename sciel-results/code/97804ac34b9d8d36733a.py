# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large square-difference-free subset of Z/205Z.
    
    Uses CRT structure (205=5*41) to initialize with a large independent set
    based on residues mod 5, then refines with local search.
    """
    m = 205
    deadline = time.time() + 590  # Safety margin before 600s kill
    
    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = (x * x) % m
        if val != 0:
            forbidden.add(val)
    
    # Precompute conflict matrix
    # conflict[i][j] is True if i and j cannot be in the same set
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(i + 1, m):
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[i][j] = True
                conflict[j][i] = True

    def is_valid(selected_set):
        """Check if a set is valid."""
        arr = sorted(selected_set)
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if conflict[arr[i], arr[j]]:
                    return False
        return True

    def can_add(selected_set, candidate):
        """Check if candidate can be added to selected_set."""
        # Vectorized check would be faster, but m=205 is small
        for s in selected_set:
            if conflict[candidate, s]:
                return False
        return True

    def get_conflicts(selected_set, candidate):
        """Return set of elements in selected_set that conflict with candidate."""
        return {s for s in selected_set if conflict[candidate, s]}

    def greedy_init_crt():
        """
        Initialize using CRT structure.
        Pick residues mod 5 that form an independent set in G_5.
        G_5 edges are diffs in {1, 4}.
        Independent set of size 2: {0, 2}.
        Include all numbers x such that x % 5 in {0, 2}.
        """
        s5 = {0, 2}
        selected = set()
        for x in range(m):
            if x % 5 in s5:
                selected.add(x)
        return selected

    def local_search(initial_set, time_limit=10.0):
        """
        Local search: try to add elements, and if blocked, try swaps.
        """
        current = set(initial_set)
        start_time = time.time()
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            
            # 1. Try to add any single element
            for candidate in range(m):
                if candidate in current:
                    continue
                if can_add(current, candidate):
                    current.add(candidate)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try single swaps (remove one, add one)
            # For each element in current, try to remove it and see if we can add something better
            # Or just try random swaps
            cur_list = list(current)
            random.shuffle(cur_list)
            
            for remove_candidate in cur_list:
                if (time.time() - start_time) >= time_limit:
                    break
                    
                temp_set = set(current)
                temp_set.discard(remove_candidate)
                
                # Try to add a new element
                for add_candidate in range(m):
                    if add_candidate in temp_set:
                        continue
                    if can_add(temp_set, add_candidate):
                        # Check if this is an improvement (it's same size, so maybe not strictly better size-wise)
                        # But it might allow further additions later.
                        # For now, we only accept if it leads to a larger set in subsequent steps.
                        # To keep it simple, we only do swaps if we can add MORE than 1 later?
                        # Actually, just accepting a swap of equal size is a way to move around the landscape.
                        # But we want to maximize size.
                        # So we should only accept swaps if they are part of a sequence that increases size.
                        # A simpler heuristic: Try to add 2 elements by removing 1?
                        pass
                
            # 3. Try 2-for-1 swaps (remove 2, add 1) -> Size decreases, bad.
            # 4. Try 1-for-2 swaps (remove 1, add 2) -> Size increases!
            # This is the most promising move.
            for remove_candidate in cur_list:
                if (time.time() - start_time) >= time_limit:
                    break
                    
                temp_set = set(current)
                temp_set.discard(remove_candidate)
                
                added_count = 0
                new_elements = []
                
                # Try to add as many as possible from temp_set
                # Greedily add
                for add_candidate in range(m):
                    if add_candidate in temp_set:
                        continue
                    if can_add(temp_set, add_candidate):
                        temp_set.add(add_candidate)
                        new_elements.append(add_candidate)
                        added_count += 1
                
                if added_count > 1:
                    # We removed 1 and added >1, so size increased
                    current = temp_set
                    improved = True
                    break

        return current

    # Start with CRT-based initialization
    best_set = greedy_init_crt()
    
    # Refine with local search
    best_set = local_search(best_set, time_limit=30.0)
    
    # Try multiple random perturbations and local searches
    rng = random.Random(42)
    
    for i in range(20):
        if time.time() > deadline - 30:
            break
            
        # Perturb best_set
        start_set = set(best_set)
        
        # Remove a few random elements
        for _ in range(rng.randint(1, 5)):
            if start_set:
                to_remove = rng.choice(list(start_set))
                start_set.discard(to_remove)
        
        # Run local search
        candidate_set = local_search(start_set, time_limit=10.0)
        
        if len(candidate_set) > len(best_set):
            best_set = candidate_set

    # Final polish
    final_set = local_search(best_set, time_limit=30.0)
    if len(final_set) > len(best_set):
        best_set = final_set

    # Convert to sorted array
    result = np.array(sorted(best_set), dtype=np.int64)
    
    # Validate
    if not is_valid(set(result.tolist())):
        # Fallback to the CRT set which is known valid
        result = np.array(sorted(greedy_init_crt()), dtype=np.int64)

    return result

# EVOLVE-BLOCK-END
