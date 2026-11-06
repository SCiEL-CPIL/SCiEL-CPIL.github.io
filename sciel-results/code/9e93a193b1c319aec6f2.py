import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 118  # 2s margin for safety
    
    points = list(range(16))
    
    # Precompute all possible triples (combinations of 3 points)
    all_triples = list(itertools.combinations(points, 3))
    all_triple_masks = []
    for t in all_triples:
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        all_triple_masks.append(m)
        
    # Precompute conflict graph: 
    # Two triples conflict if they share 2 or more points.
    # We can represent the set of triples compatible with a given triple.
    # Actually, it's easier to just check conflicts on the fly or precompute a compatibility matrix.
    # There are 560 triples. A 560x560 boolean matrix is ~313k bytes, very small.
    # compat[i][j] = True if triple i and triple j can coexist (share <= 1 point).
    # Note: A triple is always compatible with itself? No, we add distinct triples.
    # But if we consider adding T_i to a set S, T_i must be compatible with all T_j in S.
    
    # Let's precompute for each triple index i, the list of indices j such that i and j are compatible.
    # Or better, a bitmask of compatible triples? 560 bits is 70 bytes. 
    # Let's use a list of lists of compatible indices.
    compatible_with = [[] for _ in range(len(all_triples))]
    for i in range(len(all_triples)):
        mi = all_triple_masks[i]
        for j in range(len(all_triples)):
            if i == j: continue
            mj = all_triple_masks[j]
            inter = mi & mj
            # Check if inter has at most 1 bit set
            if inter and (inter & (inter - 1)):
                # More than 1 bit set -> Conflict
                pass
            else:
                compatible_with[i].append(j)
                
    # Function to compute independence number (alpha) of a set of triples
    # We want the largest subset S of points such that no triple is fully contained in S.
    # Equivalently, S is an independent set in the hypergraph.
    # We can check subsets of size k from 16 down to 1.
    # To speed this up, we can use a recursive search or simple enumeration.
    # Given 16 points, C(16, 8) = 12870. 
    # If we have ~30 triples, checking each subset against 30 triples is 30 * 12870 * (avg depth) ~ manageable.
    
    def compute_alpha(triple_indices):
        if not triple_indices:
            return 16
            
        masks = [all_triple_masks[i] for i in triple_indices]
        
        # Check k from 16 down to 1
        # Optimization: If we find a large independent set, we stop.
        # We can prune: if a subset S of size k is not independent, no superset is.
        # But we are iterating by size, not building supersets.
        # Let's just iterate.
        
        for k in range(16, 0, -1):
            # Generate combinations of points of size k
            for subset in itertools.combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                
                # Check if any triple is contained in mask
                independent = True
                for m in masks:
                    if (mask & m) == m:
                        independent = False
                        break
                if independent:
                    return k
        return 1

    # Greedy Construction
    # Start with empty set.
    # At each step, pick the triple that maximizes the number of "new pairs" covered?
    # Or simply pick a random valid triple?
    # Let's try a greedy approach that prioritizes covering points evenly.
    # Actually, a random greedy start followed by local search is often robust.
    
    def random_greedy_start():
        current_indices = []
        current_masks = []
        # To ensure we don't get stuck in a bad local optimum immediately,
        # we can pick triples that are "central" or random.
        # Let's pick random valid triples.
        available = list(range(len(all_triples)))
        random.shuffle(available)
        
        for idx in available:
            # Check if idx is compatible with all current
            ok = True
            for m in current_masks:
                inter = all_triple_masks[idx] & m
                if inter and (inter & (inter - 1)):
                    ok = False
                    break
            if ok:
                current_indices.append(idx)
                current_masks.append(all_triple_masks[idx])
                
        return current_indices

    # Local Search
    # Move: Remove one triple, add another valid triple.
    # This maintains the property that the set is linear.
    
    best_indices = random_greedy_start()
    best_alpha = compute_alpha(best_indices)
    
    current_indices = best_indices[:]
    current_alpha = best_alpha
    
    T = 1.0
    cooling = 0.99
    
    # Precompute point counts for each triple to help with heuristic?
    # Not strictly necessary for basic SA.
    
    while time.time() < deadline:
        if len(current_indices) < 2:
            break
            
        # Pick a triple to remove
        remove_idx_pos = random.randint(0, len(current_indices) - 1)
        removed_idx = current_indices[remove_idx_pos]
        
        # Tentatively remove it
        current_indices.pop(remove_idx_pos)
        
        # Find a candidate to add
        # We can pick a random triple from the remaining available ones that is compatible.
        # Or pick from a precomputed list of compatible triples for the current set?
        # That's expensive. Let's just pick a random triple from all_triples and check.
        # To bias towards "good" triples, we could pick one that shares points with existing ones?
        # Let's just pick randomly from all_triples.
        
        added = False
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts and not added:
            candidate_idx = random.randint(0, len(all_triples) - 1)
            candidate_mask = all_triple_masks[candidate_idx]
            
            # Check compatibility with all current triples
            ok = True
            for i in current_indices:
                inter = candidate_mask & all_triple_masks[i]
                if inter and (inter & (inter - 1)):
                    ok = False
                    break
            
            if ok:
                current_indices.append(candidate_idx)
                added = True
            attempts += 1
            
        if not added:
            # Revert removal
            current_indices.insert(remove_idx_pos, removed_idx)
            continue
        
        # Evaluate new state
        new_alpha = compute_alpha(current_indices)
        
        # Acceptance
        if new_alpha < current_alpha:
            current_alpha = new_alpha
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_indices = current_indices[:]
        elif new_alpha == current_alpha:
            if random.random() < 0.5:
                current_alpha = new_alpha
            else:
                # Revert
                current_indices.pop()
                current_indices.insert(remove_idx_pos, removed_idx)
        else:
            delta = new_alpha - current_alpha
            if random.random() < pow(2.718, -delta / T):
                current_alpha = new_alpha
            else:
                # Revert
                current_indices.pop()
                current_indices.insert(remove_idx_pos, removed_idx)
                
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Convert indices back to triples
    result = [list(all_triples[i]) for i in best_indices]
    return result
