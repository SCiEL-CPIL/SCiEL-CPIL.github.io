import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n_points = 16
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n_points), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute conflict lists
    conflicts = [[] for _ in range(560)]
    for i in range(560):
        t1 = set(all_triples[i])
        for j in range(i+1, 560):
            t2 = set(all_triples[j])
            if len(t1 & t2) > 1:
                conflicts[i].append(j)
                conflicts[j].append(i)

    # Precompute subset masks for independence check
    subset_masks = {}
    for k in range(2, 17):
        combos = list(itertools.combinations(range(n_points), k))
        masks = np.array([sum(1 << p for p in c) for c in combos], dtype=np.int64)
        subset_masks[k] = masks

    def compute_alpha(triples):
        """
        Returns the independence number.
        """
        if not triples:
            return n_points
        
        triple_masks = np.array([sum(1 << p for p in t) for t in triples], dtype=np.int64)
        
        for k in range(n_points, 1, -1):
            masks = subset_masks[k]
            
            # Vectorized check: for each subset mask, check if any triple is contained in it
            contains = (masks[:, None] & triple_masks[None, :]) == triple_masks[None, :]
            any_contains = np.any(contains, axis=1)
            is_independent = ~any_contains
            
            if np.any(is_independent):
                return k
                
        return 1

    def find_max_independent_set(triples):
        """
        Returns the actual set of points forming a maximum independent set.
        """
        if not triples:
            return set(range(n_points))
        
        triple_masks = np.array([sum(1 << p for p in t) for t in triples], dtype=np.int64)
        
        for k in range(n_points, 1, -1):
            combos = list(itertools.combinations(range(n_points), k))
            for combo in combos:
                mask = sum(1 << p for p in combo)
                # Check if no triple is fully contained in this subset
                contained = (mask & triple_masks) == triple_masks
                if not np.any(contained):
                    return set(combo)
        
        return set()

    # Initial construction: STS(15) on 0..14
    vectors = [
        0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 
        0b0110, 0b0111, 0b1000, 0b1001, 0b1010, 
        0b1011, 0b1100, 0b1101, 0b1110, 0b1111
    ]
    vec_to_idx = {v: i for i, v in enumerate(vectors)}
    
    initial_triples = set()
    n_sts = 15
    for i in range(n_sts):
        for j in range(i+1, n_sts):
            v_k = vectors[i] ^ vectors[j]
            k = vec_to_idx[v_k]
            t = tuple(sorted([i, j, k]))
            initial_triples.add(t)
            
    current_triples = list(initial_triples)
    current_alpha = compute_alpha(current_triples)
    
    best_triples = current_triples[:]
    best_alpha = current_alpha
    
    random.seed(42)
    
    current_set = set(triple_to_idx[t] for t in current_triples)
    
    def is_linear_with(idx_new, current_indices, exclude_idx=None):
        """Check if triple idx_new is linear with all triples in current_indices except exclude_idx."""
        for idx_old in current_indices:
            if idx_old == exclude_idx:
                continue
            if idx_new in conflicts[idx_old]:
                return False
        return True

    # Main loop
    while time.time() < deadline - 5:
        # Get current maximum independent set
        current_triples_list = [all_triples[i] for i in current_set]
        max_ind_set = find_max_independent_set(current_triples_list)
        
        # Strategy 1: Targeted move - add a triple that hits the max independent set
        if max_ind_set and len(max_ind_set) >= 2:
            # Find triples that contain at least 2 points from max_ind_set
            # This will reduce the independence number
            candidates = []
            for idx in range(560):
                t = all_triples[idx]
                t_set = set(t)
                overlap = len(t_set & max_ind_set)
                if overlap >= 2 and idx not in current_set:
                    candidates.append((idx, overlap))
            
            if candidates:
                # Sort by overlap (prefer triples hitting more points in ind set)
                random.shuffle(candidates)
                
                # Try several candidates
                for idx_add, _ in candidates[:5]:
                    # Find a triple to remove that conflicts with idx_add
                    # Or just do a swap: remove one random, add this
                    if current_set:
                        # Try removing a triple that doesn't conflict with idx_add
                        # Actually, we need to check linearity after removal
                        idx_remove = random.choice(list(current_set))
                        
                        if is_linear_with(idx_add, current_set, exclude_idx=idx_remove):
                            new_set = set(current_set)
                            new_set.remove(idx_remove)
                            new_set.add(idx_add)
                            
                            new_triples = [all_triples[i] for i in new_set]
                            new_alpha = compute_alpha(new_triples)
                            
                            if new_alpha < best_alpha:
                                best_alpha = new_alpha
                                best_triples = new_triples
                                current_set = new_set
                                current_alpha = new_alpha
                                break
                            elif new_alpha <= current_alpha and random.random() < 0.3:
                                current_set = new_set
                                current_alpha = new_alpha
                                break
        
        # Strategy 2: Random swap (original approach)
        if current_set:
            idx_remove = random.choice(list(current_set))
            
            if random.random() < 0.5:
                a, b = random.sample(range(15), 2)
                t_add = tuple(sorted([a, b, 15]))
            else:
                t_add = random.choice(all_triples)
            
            idx_add = triple_to_idx[t_add]
            
            if idx_add in current_set:
                continue
            
            if is_linear_with(idx_add, current_set, exclude_idx=idx_remove):
                new_set = set(current_set)
                new_set.remove(idx_remove)
                new_set.add(idx_add)
                
                new_triples = [all_triples[i] for i in new_set]
                new_alpha = compute_alpha(new_triples)
                
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_triples = new_triples
                    current_set = new_set
                    current_alpha = new_alpha
                elif new_alpha <= current_alpha and random.random() < 0.5:
                    current_set = new_set
                    current_alpha = new_alpha
    
    return [list(t) for t in best_triples]
