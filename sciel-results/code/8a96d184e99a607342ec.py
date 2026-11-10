import numpy as np
import itertools
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590
    
    def generate_candidates(norm_sq):
        candidates = []
        # Generate all integer vectors with squared norm == norm_sq
        # We can iterate over possible component values
        # Max component is sqrt(norm_sq)
        max_comp = int(np.sqrt(norm_sq))
        
        # Recursive generation or itertools.product
        # For efficiency, we generate tuples of squares that sum to norm_sq
        # Then assign signs and permutations
        
        # Simple approach for small norms:
        # Iterate over all combinations of absolute values
        # This is slow if norm_sq is large.
        # For norm_sq=9, max_comp=3.
        # Possible abs values: 0, 1, 2, 3
        
        # Let's use a helper to generate partitions of norm_sq into squares
        def gen_abs_comps(target, depth, current):
            if depth == 0:
                if sum(x*x for x in current) == target:
                    yield tuple(current)
                return
            if depth < 0:
                return
            # Pruning: if sum of squares of current > target, return
            # Also, we can restrict range
            
            # To avoid duplicates in generation, we can generate non-increasing sequences
            # But signs and permutations handle the rest.
            # Actually, just generating all tuples of length d from 0..max_comp is 4^11 = 4M.
            # 4M is acceptable.
            pass
        
        # Direct generation
        # For d=11, max_comp=3 (for norm 9), range is 0,1,2,3.
        # 4^11 = 4,194,304.
        # This is fast enough to generate and filter.
        
        seen = set()
        for vals in itertools.product(range(max_comp + 1), repeat=d):
            sq_sum = sum(x*x for x in vals)
            if sq_sum == norm_sq:
                # Generate all sign variations
                # To avoid duplicates, we can just add all sign variations
                # But we must be careful with zeros (sign doesn't matter)
                # And we must ensure we don't add the same vector multiple times
                # Using a set of tuples is safe.
                
                # Optimization: only generate unique sign patterns
                # If a component is 0, sign is irrelevant.
                # If a component is non-zero, sign can be + or -.
                
                # We'll just iterate 2^d sign patterns? 2^11 = 2048.
                # 4M * 2048 is too big.
                
                # Better: generate unique absolute value tuples first?
                # No, the product loop above generates ordered tuples.
                # So (1,2,0...) and (2,1,0...) are distinct in the loop.
                # We just need to handle signs.
                
                # For a specific vals tuple, generate all sign variations.
                # We can do this by iterating 0 to 2^d - 1.
                # But many will be duplicates if there are zeros.
                
                # Let's just add the vector with all positive signs to a list of "base" vectors?
                # No, we need all of them.
                
                # Efficient way:
                # Iterate over all sign masks.
                for mask in range(1 << d):
                    vec = np.zeros(d, dtype=np.int64)
                    for i in range(d):
                        if vals[i] != 0:
                            if (mask >> i) & 1:
                                vec[i] = -vals[i]
                            else:
                                vec[i] = vals[i]
                    # Add to seen to avoid duplicates from symmetries?
                    # The product loop generates unique absolute value sequences.
                    # The sign loop generates unique signs for that sequence.
                    # So no duplicates between different vals.
                    # Duplicates within same vals?
                    # If vals has 0, changing sign of 0 doesn't change vec.
                    # So we might generate the same vec multiple times for the same vals.
                    # Example: vals=(1,0). mask 0: (1,0). mask 1: (-1,0) (if bit 0 is for index 0? No, bit 0 for index 0).
                    # If vals[0]=1, vals[1]=0.
                    # mask 0: (1, 0)
                    # mask 1: (-1, 0) -> Different.
                    # mask 2: (1, 0) -> Same as mask 0 because vals[1]=0.
                    # So we have duplicates.
                    
                    # To avoid duplicates, we can skip masks that only flip zeros.
                    # Or just use a set.
                    seen.add(tuple(vec))
        
        return [np.array(v, dtype=np.int64) for v in seen]

    def solve_for_norm(norm_sq):
        if time.time() > deadline - 60:
            return None, 0
            
        candidates = generate_candidates(norm_sq)
        n = len(candidates)
        if n == 0:
            return None, 0
            
        # Build conflict graph
        # conflict[i][j] is True if dist(candidates[i], candidates[j])^2 < norm_sq
        # We can compute this using numpy
        # dist^2 = ||u||^2 + ||v||^2 - 2 u.v = 2*norm_sq - 2 u.v
        # So conflict if 2*norm_sq - 2 u.v < norm_sq
        # norm_sq < 2 u.v
        # u.v > norm_sq / 2
        
        # Compute dot products
        # C = candidates @ candidates.T
        # conflict = C > norm_sq / 2
        
        C = np.dot(candidates, candidates.T)
        conflict = C > (norm_sq / 2.0)
        np.fill_diagonal(conflict, False)
        
        # Greedy initialization
        # Sort by degree (number of conflicts)
        degrees = np.sum(conflict, axis=1)
        order = np.argsort(degrees)
        
        selected = []
        selected_set = set()
        forbidden = np.zeros(n, dtype=bool)
        
        for i in order:
            if not forbidden[i]:
                selected.append(i)
                selected_set.add(i)
                forbidden[conflict[i]] = True
                
        best_selected = selected[:]
        best_count = len(selected)
        
        # Local Search / ILS
        rng = np.random.default_rng(42)
        
        # Precompute neighbor lists for faster updates?
        # conflict is a boolean matrix.
        
        def local_search(current_selected, deadline_local):
            # current_selected is a list of indices
            # We maintain a count of how many selected nodes are adjacent to each node
            # count[j] = sum(1 for i in current_selected if conflict[i][j])
            
            count = np.zeros(n, dtype=np.int32)
            for i in current_selected:
                count[conflict[i]] += 1
            
            # Also maintain a set of selected indices for O(1) lookup
            sel_set = set(current_selected)
            
            # List of nodes with count 0 (candidates for addition)
            # We can just check all nodes each time?
            # Or maintain a list.
            
            improved = True
            while improved and time.time() < deadline_local:
                improved = False
                
                # 1. Try to add a node
                # Find all j not in sel_set with count[j] == 0
                # We can iterate through all j
                added = False
                for j in range(n):
                    if j not in sel_set and count[j] == 0:
                        # Add j
                        sel_set.add(j)
                        current_selected.append(j)
                        count[conflict[j]] += 1
                        improved = True
                        added = True
                        break # Restart loop to recalculate? Or continue?
                        # If we add one, we should restart to find more.
                        # But we can just continue checking?
                        # The count for other nodes might change.
                        # So restart is safer.
                        if time.time() > deadline_local:
                            return current_selected
                        break # Break to restart while loop?
                        # Actually, if we break the for loop, we go to next while iteration.
                        # This is good.
                
                if added:
                    continue
                    
                # 2. Try to swap
                # For each i in current_selected:
                #   Remove i
                #   Check if any j not in sel_set has count[j] == 0
                #   If yes, add j
                #   If no, restore i
                
                for idx, i in enumerate(current_selected):
                    if time.time() > deadline_local:
                        return current_selected
                    
                    # Remove i
                    sel_set.discard(i)
                    count[conflict[i]] -= 1
                    
                    # Find j
                    found_j = -1
                    for j in range(n):
                        if j not in sel_set and count[j] == 0:
                            found_j = j
                            break
                    
                    if found_j != -1:
                        # Swap i with found_j
                        current_selected[idx] = found_j
                        sel_set.discard(i) # Already discarded
                        sel_set.add(found_j)
                        # Update counts for new node
                        count[conflict[found_j]] += 1
                        # Note: count for i's neighbors was decreased.
                        # We don't need to restore i's counts because i is removed.
                        # But we need to ensure count is correct for the new set.
                        # The count array reflects the current selected set.
                        # When we removed i, we decremented counts for i's neighbors.
                        # When we add found_j, we increment counts for found_j's neighbors.
                        # This is correct.
                        
                        improved = True
                        break # Restart local search
                    
                    else:
                        # Restore i
                        sel_set.add(i)
                        count[conflict[i]] += 1
            
            return current_selected

        # Run ILS
        for iter in range(5000):
            if time.time() > deadline - 30:
                break
            
            # Perturb
            current = best_selected[:]
            # Remove 5 random nodes
            if len(current) > 10:
                remove_idx = rng.choice(len(current), size=5, replace=False)
                for idx in sorted(remove_idx, reverse=True):
                    current.pop(idx)
            
            # Local Search
            current = local_search(current, deadline - 30)
            
            if len(current) > best_count:
                best_count = len(current)
                best_selected = current[:]
                
        return [candidates[i] for i in best_selected], best_count

    best_points = None
    best_score = 0
    
    # Try Norm 3 (sq=9)
    if time.time() < deadline - 120:
        points, score = solve_for_norm(9)
        if points and score > best_score:
            best_points = points
            best_score = score
            
    # Try Norm 4 (sq=16)
    if time.time() < deadline - 60:
        points, score = solve_for_norm(16)
        if points and score > best_score:
            best_points = points
            best_score = score
            
    # Fallback to Norm 2 (sq=4) if nothing found (unlikely)
    if best_points is None:
        # D11 vectors
        points = []
        for i in range(d):
            for j in range(i+1, d):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = si
                        p[j] = sj
                        points.append(p)
        best_points = points
        best_score = len(points)
        
    return np.array(best_points, dtype=np.int64)
