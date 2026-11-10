# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    k = 2
    
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    forbidden_list = sorted(forbidden)
    forbidden_set = set(forbidden_list)
    
    # Build adjacency: adj[i] = set of j such that (i-j) mod m or (j-i) mod m is in forbidden
    # i.e., i and j cannot both be in the set
    adj = [set() for _ in range(m)]
    for i in range(m):
        for f in forbidden_list:
            j = (i + f) % m
            adj[i].add(j)
            adj[j].add(i)
        # Also (i - f) mod m = (i + (m-f)) % m, but since forbidden is symmetric (if f in F then m-f in F? 
        # Actually squares: if x^2 ≡ f mod m, then (-x)^2 ≡ f mod m, so f and m-f: 
        # (-x)^2 = x^2, so f is a square iff f is a square. m-f is a square iff there exists y with y^2 ≡ m-f.
        # Not necessarily symmetric. But we handle both directions via the loop.
    
    # Actually let me rebuild more carefully
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden_set:
                adj[i].add(j)
                adj[j].add(i)
    
    # Convert to bitmasks for fast operations
    adj_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask
    
    def is_valid_subset(subset):
        """Check if subset is valid (no forbidden differences)."""
        s = set(subset)
        for a in subset:
            for b in subset:
                if a == b:
                    continue
                d = (a - b) % m
                if d in forbidden_set:
                    return False
        return True
    
    def can_add(subset_set, candidate):
        """Check if candidate can be added to subset_set."""
        for s in subset_set:
            d = (candidate - s) % m
            if d in forbidden_set:
                return False
            d2 = (s - candidate) % m
            if d2 in forbidden_set:
                return False
        return True
    
    def greedy_init(order):
        """Greedy construction following given order."""
        selected = []
        selected_set = set()
        for candidate in order:
            ok = True
            for s in selected:
                d1 = (candidate - s) % m
                d2 = (s - candidate) % m
                if d1 in forbidden_set or d2 in forbidden_set:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
                selected_set.add(candidate)
        return selected
    
    def local_search(subset, max_iters=100000):
        """Local search to improve subset size."""
        subset = list(subset)
        subset_set = set(subset)
        best = list(subset)
        best_set = set(subset)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try adding a new element
            for candidate in range(m):
                if candidate in subset_set:
                    continue
                if can_add(subset_set, candidate):
                    subset.append(candidate)
                    subset_set.add(candidate)
                    improved = True
                    break
            
            if not improved:
                # Try swap: remove one, add one
                for remove_idx in range(len(subset)):
                    removed = subset[remove_idx]
                    subset_set.remove(removed)
                    
                    # Try to add something
                    added = False
                    for candidate in range(m):
                        if candidate in subset_set:
                            continue
                        if can_add(subset_set, candidate):
                            subset.append(candidate)
                            subset_set.add(candidate)
                            added = True
                            break
                    
                    if not added:
                        # Put it back
                        subset[remove_idx] = removed
                        subset_set.add(removed)
                    else:
                        # Remove the old one from list
                        subset.pop()  # remove the last added? No, we need to remove 'removed'
                        # Actually let me redo this
                        # We removed 'removed' from subset_set but it's still in subset list
                        # Let me fix the logic
                        pass
                    # This is getting messy, let me use a cleaner approach
                    break
            
            if not improved:
                # Try remove one, add two
                for remove_idx in range(len(subset)):
                    removed = subset[remove_idx]
                    temp_set = subset_set - {removed}
                    temp_list = [x for i, x in enumerate(subset) if i != remove_idx]
                    
                    # Try to add two elements
                    added_count = 0
                    added_elements = []
                    for candidate in range(m):
                        if candidate in temp_set:
                            continue
                        if can_add(temp_set, candidate):
                            temp_set.add(candidate)
                            temp_list.append(candidate)
                            added_elements.append(candidate)
                            added_count += 1
                            if added_count == 2:
                                break
                    
                    if added_count >= 2:
                        subset = temp_list
                        subset_set = temp_set
                        improved = True
                        break
                # If no improvement from remove-add-2, try single swap more carefully
                if not improved:
                    for remove_idx in range(len(subset)):
                        removed = subset[remove_idx]
                        temp_set = subset_set - {removed}
                        
                        for candidate in range(m):
                            if candidate in temp_set:
                                continue
                            if can_add(temp_set, candidate):
                                subset[remove_idx] = candidate
                                subset_set = temp_set | {candidate}
                                improved = True
                                break
                        if improved:
                            break
        
        return subset
    
    # Better local search implementation
    def local_search_v2(subset, max_iters=50000):
        """Cleaner local search."""
        subset = list(subset)
        subset_set = set(subset)
        
        for _ in range(max_iters):
            improved = False
            
            # Phase 1: Try to add any single element
            for candidate in range(m):
                if candidate in subset_set:
                    continue
                if can_add(subset_set, candidate):
                    subset.append(candidate)
                    subset_set.add(candidate)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Phase 2: Try swap (remove one, add one)
            swap_done = False
            for remove_idx in range(len(subset)):
                removed = subset[remove_idx]
                temp_set = subset_set - {removed}
                
                for candidate in range(m):
                    if candidate in temp_set:
                        continue
                    if can_add(temp_set, candidate):
                        subset[remove_idx] = candidate
                        subset_set = temp_set | {candidate}
                        swap_done = True
                        improved = True
                        break
                
                if swap_done:
                    break
            
            if improved:
                continue
            
            # Phase 3: Try remove one, add two
            for remove_idx in range(len(subset)):
                removed = subset[remove_idx]
                temp_set = subset_set - {removed}
                
                added = []
                for candidate in range(m):
                    if candidate in temp_set:
                        continue
                    if can_add(temp_set, candidate):
                        temp_set.add(candidate)
                        added.append(candidate)
                        if len(added) == 2:
                            break
                
                if len(added) >= 2:
                    new_subset = [x for i, x in enumerate(subset) if i != remove_idx] + added
                    subset = new_subset
                    subset_set = set(new_subset)
                    improved = True
                    break
            
            if not improved:
                break
        
        return subset
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin from 630s limit
    
    best_subset = None
    best_size = 0
    
    # Generate multiple initial solutions
    random.seed(42)
    
    # Deterministic greedy
    base_order = list(range(m))
    sol = greedy_init(base_order)
    if len(sol) > best_size:
        best_size = len(sol)
        best_subset = list(sol)
    
    # Try reverse order
    sol = greedy_init(list(range(m - 1, -1, -1)))
    if len(sol) > best_size:
        best_size = len(sol)
        best_subset = list(sol)
    
    # Try several random orderings
    for restart in range(50):
        if time.time() > deadline:
            break
        order = list(range(m))
        random.shuffle(order)
        sol = greedy_init(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_subset = list(sol)
    
    # Local search on best solution
    if time.time() < deadline:
        improved = local_search_v2(best_subset)
        if len(improved) > best_size:
            best_size = len(improved)
            best_subset = improved
    
    # More randomized local search
    while time.time() < deadline - 10:
        # Create a perturbed version of best solution
        if len(best_subset) < 2:
            break
        
        # Remove a random element and try to rebuild
        perturbed = list(best_subset)
        remove_idx = random.randint(0, len(perturbed) - 1)
        removed_elem = perturbed.pop(remove_idx)
        
        # Try to add back different elements
        perturbed_set = set(perturbed)
        for candidate in range(m):
            if candidate in perturbed_set:
                continue
            if can_add(perturbed_set, candidate):
                perturbed.append(candidate)
                perturbed_set.add(candidate)
                break
        
        # Local search from perturbed
        result = local_search_v2(perturbed, max_iters=10000)
        if len(result) > best_size:
            best_size = len(result)
            best_subset = result
    
    # Final polish
    if time.time() < deadline:
        final = local_search_v2(best_subset, max_iters=100000)
        if len(final) > best_size:
            best_size = len(final)
            best_subset = final
    
    return np.array(sorted(best_subset), dtype=np.int64)

# EVOLVE-BLOCK-END
