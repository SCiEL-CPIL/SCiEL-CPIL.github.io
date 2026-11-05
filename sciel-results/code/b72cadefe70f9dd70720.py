"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys
from collections import defaultdict

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    # Precompute range
    range_1_N = list(range(1, N + 1))
    
    def is_valid_sidon(selected):
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_sums(selected):
        sums = set()
        for i, a in enumerate(selected):
            for b in selected[i:]:
                sums.add(a + b)
        return sums

    def can_add(x, selected, sums):
        """Check if x can be added to selected without violating Sidon property."""
        if 2 * x in sums:
            return False
        for a in selected:
            if x + a in sums:
                return False
        return True

    def is_maximal(selected, sums):
        """Check if no element from [1, N] can be added."""
        sel_set = set(selected)
        for x in range_1_N:
            if x in sel_set:
                continue
            if can_add(x, selected, sums):
                return False
        return True

    def greedy_construct(order):
        """Construct a maximal Sidon set using a given candidate order."""
        selected = []
        sums = set()
        sel_set = set()
        
        for x in order:
            if x in sel_set:
                continue
            if can_add(x, selected, sums):
                selected.append(x)
                sel_set.add(x)
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(x + a)
        
        return selected, sums

    def local_search_minimize(selected, sums, time_limit):
        """Try to remove elements while maintaining maximality."""
        selected = list(selected)
        sums = get_sums(selected)
        sel_set = set(selected)
        
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(selected)):
                x = selected[i]
                # Tentatively remove x
                new_selected = selected[:i] + selected[i+1:]
                if not new_selected:
                    continue
                
                new_sums = get_sums(new_selected)
                
                # Check if maximal
                if is_maximal(new_selected, new_sums):
                    selected = new_selected
                    sums = new_sums
                    sel_set = set(selected)
                    changed = True
                    break
        
        return selected

    best_result = None
    best_size = float('inf')
    
    # Strategy 1: Ascending greedy (baseline)
    selected_orig, sums_orig = greedy_construct(range_1_N)
    if is_maximal(selected_orig, sums_orig):
        best_result = selected_orig
        best_size = len(selected_orig)
    
    # Strategy 2: Descending greedy
    selected_desc, sums_desc = greedy_construct(range(N, 0, -1))
    if is_maximal(selected_desc, sums_desc):
        if len(selected_desc) < best_size:
            best_result = selected_desc
            best_size = len(selected_desc)
    
    # Strategy 3: Randomized greedy with different orderings
    # Use multiple random seeds for diversity
    for seed in range(200):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        candidates = list(range_1_N)
        random.shuffle(candidates)
        
        selected, sums = greedy_construct(candidates)
        
        # It should be maximal by construction, but verify just in case
        if not is_maximal(selected, sums):
            # Fallback: extend it
            for x in range_1_N:
                if x not in set(selected) and can_add(x, selected, sums):
                    selected.append(x)
                    sums.add(2 * x)
                    for a in selected[:-1]:
                        sums.add(x + a)
        
        # Local search to minimize
        selected_min = local_search_minimize(selected, sums, deadline - 5)
        
        # Verify final result
        sums_final = get_sums(selected_min)
        if is_valid_sidon(selected_min) and is_maximal(selected_min, sums_final) and len(selected_min) < best_size:
            best_result = selected_min
            best_size = len(selected_min)
            
            # Early exit if we find a very small set
            if best_size <= 16:
                break
    
    # Strategy 4: Priority-based construction
    # Try to construct with elements that block the most others
    for strategy_seed in range(50):
        if time.time() > deadline:
            break
        
        random.seed(strategy_seed)
        
        # Calculate "blocking potential" for each element
        # An element blocks y if adding y would create a conflict with x
        # We want to pick elements that block many others
        
        # For each x, count how many y in [1,N] would conflict with x
        # Conflict: 2y == x+y (impossible) or x+y conflicts with existing sums
        # This is dynamic, so we use a heuristic: prefer elements with high "isolation"
        
        # Actually, let's try a different approach:
        # Start with a random small set, then extend greedily
        
        # Pick 5-10 random elements as a seed
        seed_size = random.randint(5, 10)
        seed_elements = random.sample(range_1_N, seed_size)
        
        # Try to find a valid Sidon set containing these elements
        # This is complex, so instead let's just do random greedy with bias
        
        # Bias towards elements that are "far" from each other
        candidates = list(range_1_N)
        
        # Sort by random priority but with some structure
        random.shuffle(candidates)
        
        selected, sums = greedy_construct(candidates)
        
        if not is_maximal(selected, sums):
            for x in range_1_N:
                if x not in set(selected) and can_add(x, selected, sums):
                    selected.append(x)
                    sums.add(2 * x)
                    for a in selected[:-1]:
                        sums.add(x + a)
        
        selected_min = local_search_minimize(selected, sums, deadline - 5)
        
        sums_final = get_sums(selected_min)
        if is_valid_sidon(selected_min) and is_maximal(selected_min, sums_final) and len(selected_min) < best_size:
            best_result = selected_min
            best_size = len(selected_min)
            
            if best_size <= 16:
                break
    
    if best_result is not None:
        return best_result
    
    return selected_orig
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(res)
