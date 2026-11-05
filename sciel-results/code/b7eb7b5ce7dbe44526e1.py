"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    # Precompute range
    range_1_N = range(1, N + 1)
    
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
        # 2*x must not be in sums
        if 2 * x in sums:
            return False
        # x + a must not be in sums for any a in selected
        # Also x + a must be distinct from other x + b (always true if a!=b)
        # And x + a must not be 2*x (impossible since a>0)
        for a in selected:
            if x + a in sums:
                return False
        return True

    def is_maximal(selected, sums):
        """Check if no element from [1, N] can be added."""
        for x in range_1_N:
            if x in selected:
                continue
            if can_add(x, selected, sums):
                return False
        return True

    def random_greedy(seed):
        """Construct a maximal Sidon set using randomized greedy."""
        random.seed(seed)
        candidates = list(range_1_N)
        random.shuffle(candidates)
        
        selected = []
        sums = set()
        
        for x in candidates:
            if can_add(x, selected, sums):
                selected.append(x)
                # Update sums
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(x + a)
        
        # Ensure maximality (greedy on random order might leave holes if we don't check all? 
        # Actually, if we process all candidates and add if possible, the result is maximal 
        # with respect to the order? No, it's maximal with respect to the set. 
        # If x was skipped, it was because it conflicted with current sums. 
        # Adding more elements only adds more sums, so it will still conflict. 
        # So the result is maximal.)
        
        return selected

    def local_search_minimize(selected, sums):
        """Try to remove elements while maintaining maximality."""
        # We want to remove elements. 
        # If we remove an element, we must ensure that any element that was previously blocked 
        # only by that element is now blocked by others, or was already blocked by others.
        # Actually, simpler: 
        # 1. Try removing an element x.
        # 2. Check if the new set is still Sidon (it is, subset of Sidon).
        # 3. Check if the new set is still Maximal.
        
        selected = list(selected)
        sums = get_sums(selected)
        
        # We iterate and try to remove elements. 
        # To be efficient, we can just try all elements and remove the first one that works?
        # Or try all and keep the one that leads to smallest set?
        # Since we want to minimize size, we should try to remove as many as possible.
        
        # A simple heuristic: 
        # While we can remove an element and remain maximal, do it.
        
        changed = True
        while changed:
            changed = False
            # Shuffle to avoid bias? Or try all?
            # Trying all is O(N * (N*K)) = O(N^2 K) per pass.
            # K is small (target < 20). N=256. 256^2 * 20 ~ 1.3M ops. 
            # This is fast enough for a few passes.
            
            for i in range(len(selected)):
                x = selected[i]
                # Tentatively remove x
                new_selected = selected[:i] + selected[i+1:]
                if not new_selected:
                    # Can't be empty if maximal? 
                    # If empty, not maximal. 
                    continue
                
                new_sums = get_sums(new_selected)
                
                # Check if maximal
                if is_maximal(new_selected, new_sums):
                    selected = new_selected
                    sums = new_sums
                    changed = True
                    break # Restart the loop with the new set
        
        return selected

    best_result = None
    best_size = float('inf')
    
    # Baseline: Ascending greedy
    selected_orig = []
    sums_orig = set()
    for x in range_1_N:
        if can_add(x, selected_orig, sums_orig):
            selected_orig.append(x)
            sums_orig.add(2 * x)
            for a in selected_orig[:-1]:
                sums_orig.add(x + a)
    
    if is_maximal(selected_orig, sums_orig):
        best_result = selected_orig
        best_size = len(selected_orig)
        
    # Randomized search
    # We run multiple random greedy constructions and then local search.
    num_trials = 1000
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        selected = random_greedy(trial)
        
        # It should be maximal by construction, but verify just in case
        sums = get_sums(selected)
        if not is_maximal(selected, sums):
            # Fallback: extend it
            for x in range_1_N:
                if x not in selected and can_add(x, selected, sums):
                    selected.append(x)
                    sums.add(2 * x)
                    for a in selected[:-1]:
                        sums.add(x + a)
                    
        # Local search to minimize
        selected_min = local_search_minimize(selected, sums)
        
        # Verify final result
        sums_final = get_sums(selected_min)
        if is_maximal(selected_min, sums_final) and len(selected_min) < best_size:
            best_result = selected_min
            best_size = len(selected_min)
            
            # Early exit if we find a very small set? 
            # Theoretical minimum is around sqrt(N) ~ 16. 
            # If we find 16, we might stop. But let's run a bit longer.
            if best_size <= 15:
                break

    if best_result is not None:
        return best_result
        
    return selected_orig
# EVOLVE-BLOCK-END
