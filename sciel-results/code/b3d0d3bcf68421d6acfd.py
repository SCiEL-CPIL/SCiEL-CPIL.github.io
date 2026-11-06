import numpy as np
import itertools
import random
import time

def construct():
    n = 16
    all_points = list(range(n))
    
    # Generate all possible triples as tuples and bitmasks
    all_triples = list(itertools.combinations(range(n), 3))
    triple_bitmasks = []
    for t in all_triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_bitmasks.append(mask)
    
    # Precompute subset bitmasks for each size
    # subset_masks[size] is a list of bitmasks for all subsets of that size
    subset_masks = {}
    for size in range(n + 1):
        masks = []
        for subset in itertools.combinations(range(n), size):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            masks.append(mask)
        subset_masks[size] = np.array(masks, dtype=np.uint32)

    def get_alpha(triple_indices):
        """
        Calculate the independence number of the system defined by triple_indices.
        triple_indices: list of indices into all_triples
        """
        if not triple_indices:
            return n
        
        # Get the bitmasks for the active triples
        active_masks = np.array([triple_bitmasks[i] for i in triple_indices], dtype=np.uint32)
        
        # Check sizes from largest to smallest
        for size in range(n, 0, -1):
            subs = subset_masks[size]
            if len(subs) == 0:
                continue
            
            # Check if any subset is independent
            # A subset S is independent if no active triple T is a subset of S
            # i.e., (S & T) != T for all T
            # We want to know if there EXISTS a S such that for ALL T, (S & T) != T
            
            # Vectorized check:
            # For each S in subs, check against all T in active_masks
            # This creates a boolean matrix of shape (len(subs), len(active_masks))
            # entry [i, j] is True if (subs[i] & active_masks[j]) != active_masks[j]
            
            # To avoid large memory if len(subs) is huge, we can chunk or just do it.
            # Max len(subs) is C(16,8)=12870. active_masks ~ 40.
            # 12870 * 40 is small.
            
            # Compute intersections
            # subs[:, None] has shape (N_sub, 1)
            # active_masks[None, :] has shape (1, N_tri)
            # result has shape (N_sub, N_tri)
            
            # We need: ALL columns for a row are True
            # i.e., for row i, all (subs[i] & active_masks[j]) != active_masks[j]
            
            # If active_masks is empty, all are independent.
            if len(active_masks) == 0:
                return size

            # Check
            # (subs[:, None] & active_masks[None, :]) != active_masks[None, :]
            # This is a boolean array.
            # We need to check if any row has all True.
            
            # Optimization: If we find one independent subset, we can return size immediately.
            # But vectorized check gives all at once.
            # np.all(..., axis=1) gives a boolean array of length N_sub.
            # If any is True, return size.
            
            is_indep = np.all((subs[:, None] & active_masks[None, :]) != active_masks[None, :], axis=1)
            
            if np.any(is_indep):
                return size
        
        return 0

    # Greedy initialization
    def greedy_init():
        used_pairs = set()
        triples = []
        indices = list(range(len(all_triples)))
        random.shuffle(indices)
        
        for idx in indices:
            t = all_triples[idx]
            a, b, c = t
            pairs = {(a,b), (a,c), (b,c)}
            if any(p in used_pairs for p in pairs):
                continue
            triples.append(idx)
            used_pairs.update(pairs)
        return triples

    # Simulated Annealing
    best_triples = greedy_init()
    best_alpha = get_alpha(best_triples)
    
    current_triples = best_triples[:]
    current_alpha = best_alpha
    
    # Precompute pair sets for quick validity check
    triple_pairs = []
    for t in all_triples:
        a, b, c = t
        triple_pairs.append({(a,b), (a,c), (b,c)})
    
    # Map pair to list of triples containing it? 
    # For now, just check validity by iterating all triples.
    
    def is_valid_triple(idx, used_pairs):
        pairs = triple_pairs[idx]
        for p in pairs:
            if p in used_pairs:
                return False
        return True

    def get_used_pairs(triples):
        used = set()
        for idx in triples:
            used.update(triple_pairs[idx])
        return used

    current_used_pairs = get_used_pairs(current_triples)
    
    start_time = time.time()
    time_limit = 110.0 # Leave buffer for overhead
    
    T = 10.0
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        if T < 0.1:
            break
            
        # Pick a random triple to remove
        if not current_triples:
            break
        rem_idx_pos = random.randint(0, len(current_triples) - 1)
        rem_idx = current_triples[rem_idx_pos]
        
        # Remove it
        current_triples.pop(rem_idx_pos)
        # Update used_pairs
        rem_pairs = triple_pairs[rem_idx]
        for p in rem_pairs:
            current_used_pairs.discard(p)
            
        # Find valid triples to add
        # We can pick from all triples that are now valid
        valid_candidates = []
        for idx in range(len(all_triples)):
            if is_valid_triple(idx, current_used_pairs):
                valid_candidates.append(idx)
                
        if not valid_candidates:
            # Re-add the removed one
            current_triples.append(rem_idx)
            current_used_pairs.update(rem_pairs)
            continue
            
        add_idx = random.choice(valid_candidates)
        
        # Add it
        current_triples.append(add_idx)
        add_pairs = triple_pairs[add_idx]
        for p in add_pairs:
            current_used_pairs.add(p)
            
        new_alpha = get_alpha(current_triples)
        
        delta = new_alpha - current_alpha
        
        if delta < 0:
            # Better
            current_alpha = new_alpha
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = current_triples[:]
        else:
            # Worse or same
            prob = np.exp(-delta / T)
            if random.random() < prob:
                current_alpha = new_alpha
                
        T *= cooling_rate
        
    # Convert indices back to triples
    result = [list(all_triples[i]) for i in best_triples]
    return result
