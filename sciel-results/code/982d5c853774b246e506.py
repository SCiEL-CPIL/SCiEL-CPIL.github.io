import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_points = list(range(16))
    all_triples_list = list(itertools.combinations(all_points, 3))
    num_triples = len(all_triples_list)
    
    # Precompute masks and triple-to-index mapping
    triple_masks = {}
    triple_to_idx = {}
    for i, t in enumerate(all_triples_list):
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[t] = mask
        triple_to_idx[t] = i
        
    # Precompute conflict bitsets
    conflict_bits = [0] * num_triples
    for i in range(num_triples):
        mi = triple_masks[all_triples_list[i]]
        for j in range(i+1, num_triples):
            mj = triple_masks[all_triples_list[j]]
            inter = mi & mj
            if inter & (inter - 1):  # share > 1 point
                conflict_bits[i] |= (1 << j)
                conflict_bits[j] |= (1 << i)
    
    # Precompute which triples contain each pair of points
    # pair_to_triples[(p,q)] = list of triple indices containing both p and q
    pair_to_triples = {}
    for i, t in enumerate(all_triples_list):
        for a, b in itertools.combinations(t, 2):
            key = (min(a,b), max(a,b))
            if key not in pair_to_triples:
                pair_to_triples[key] = []
            pair_to_triples[key].append(i)
    
    def find_max_independent_set(active_mask):
        """
        Find the size of the largest subset of points that contains no triple.
        Uses branch-and-bound / backtracking approach.
        """
        if active_mask == 0:
            return 16
        
        # Get list of active triple indices
        active_indices = [i for i in range(num_triples) if (active_mask >> i) & 1]
        if not active_indices:
            return 16
        
        active_masks = [triple_masks[all_triples_list[i]] for i in active_indices]
        
        # Check from k=16 down to 1
        for k in range(16, 0, -1):
            for subset in itertools.combinations(all_points, k):
                smask = 0
                for p in subset:
                    smask |= (1 << p)
                
                is_ind = True
                for tm in active_masks:
                    if (tm & smask) == tm:
                        is_ind = False
                        break
                if is_ind:
                    return k
            if time.time() > deadline - 5:
                return k  # Conservative estimate
        return 0
    
    def greedy_dense_construction():
        """
        Greedily construct a dense linear triple system by covering uncovered point pairs.
        """
        active_mask = 0
        # Track which pairs are covered
        covered_pairs = set()
        
        # Start with random initialization
        indices = list(range(num_triples))
        random.shuffle(indices)
        
        # First pass: add non-conflicting triples randomly
        for idx in indices:
            if (conflict_bits[idx] & active_mask) == 0:
                active_mask |= (1 << idx)
                t = all_triples_list[idx]
                for a, b in itertools.combinations(t, 2):
                    covered_pairs.add((min(a,b), max(a,b)))
        
        # Second pass: try to add more triples that cover uncovered pairs
        remaining_indices = [i for i in range(num_triples) if not ((active_mask >> i) & 1)]
        random.shuffle(remaining_indices)
        
        for idx in remaining_indices:
            if (conflict_bits[idx] & active_mask) != 0:
                continue
            
            t = all_triples_list[idx]
            new_pairs = 0
            for a, b in itertools.combinations(t, 2):
                key = (min(a,b), max(a,b))
                if key not in covered_pairs:
                    new_pairs += 1
            
            # Add if it covers at least one new pair
            if new_pairs > 0:
                active_mask |= (1 << idx)
                for a, b in itertools.combinations(t, 2):
                    covered_pairs.add((min(a,b), max(a,b)))
        
        return active_mask
    
    def local_search(mask, max_iterations=100):
        """Local search to minimize independence number."""
        current_alpha = find_max_independent_set(mask)
        current_mask = mask
        
        for it in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Strategy: pick a random inactive triple, add it, remove conflicts, greedy refill
            inactive_indices = [i for i in range(num_triples) if not ((current_mask >> i) & 1)]
            if not inactive_indices:
                break
            
            target_idx = random.choice(inactive_indices)
            
            # Remove conflicts
            conflicts = conflict_bits[target_idx] & current_mask
            new_mask = current_mask & ~conflicts
            new_mask |= (1 << target_idx)
            
            # Greedy refill: prefer triples that cover more uncovered pairs
            refill_indices = list(range(num_triples))
            random.shuffle(refill_indices)
            added = 0
            for idx in refill_indices:
                if (new_mask >> idx) & 1:
                    continue
                if (conflict_bits[idx] & new_mask) == 0:
                    new_mask |= (1 << idx)
                    added += 1
                    if added > 50:  # Limit refill for speed
                        break
            
            new_alpha = find_max_independent_set(new_mask)
            
            if new_alpha <= current_alpha:
                current_alpha = new_alpha
                current_mask = new_mask
        
        return current_mask, current_alpha
    
    best_alpha = 16
    best_mask = 0
    
    # Multiple restarts with different strategies
    num_restarts = 25
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Alternate between random init and dense greedy
        if restart % 2 == 0:
            current_mask = greedy_dense_construction()
        else:
            # Random greedy
            active_mask = 0
            indices = list(range(num_triples))
            random.shuffle(indices)
            for idx in indices:
                if (conflict_bits[idx] & active_mask) == 0:
                    active_mask |= (1 << idx)
            current_mask = active_mask
        
        current_mask, current_alpha = local_search(current_mask, max_iterations=50)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_mask = current_mask
        
        # If we've reached a very good solution, do more polishing
        if best_alpha <= 6 and time.time() < deadline - 20:
            best_mask, best_alpha = local_search(best_mask, max_iterations=100)
    
    # Final polish
    if time.time() < deadline - 5:
        best_mask, best_alpha = local_search(best_mask, max_iterations=50)
    
    # Convert mask to list of triples
    result = []
    for i in range(num_triples):
        if (best_mask >> i) & 1:
            result.append(list(all_triples_list[i]))
    
    return result
