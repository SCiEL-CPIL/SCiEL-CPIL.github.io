import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    n = 16
    points = list(range(n))
    all_triples = list(itertools.combinations(points, 3))
    num_all = len(all_triples)
    
    # Precompute bitmasks for each triple
    triple_masks = [0] * num_all
    triple_pairs = []
    for i, t in enumerate(all_triples):
        a, b, c = t
        mask = (1 << a) | (1 << b) | (1 << c)
        triple_masks[i] = mask
        triple_pairs.append([(a, b), (a, c), (b, c)])
    
    # Precompute pair masks for each triple (for fast conflict checking)
    triple_pair_masks = []
    for i in range(num_all):
        pm = 0
        for a, b in triple_pairs[i]:
            pm |= (1 << a) | (1 << b)
        triple_pair_masks.append(pm)
    
    # Precompute all subset masks by size for independence check
    # Use a smarter approach: precompute which triples hit each subset
    def get_alpha_fast(triple_idx_set):
        """Compute independence number using bit operations."""
        # Build a bitmask of all points covered by at least one triple
        # An independent set is a subset S where no triple is fully contained in S
        # i.e., for every triple t, t is NOT a subset of S
        
        # Try from largest to smallest
        for k in range(n, 0, -1):
            found = False
            for subset in itertools.combinations(points, k):
                s_mask = 0
                for p in subset:
                    s_mask |= (1 << p)
                # Check if any triple is fully contained in s_mask
                is_independent = True
                for ti in triple_idx_set:
                    if (triple_masks[ti] & s_mask) == triple_masks[ti]:
                        is_independent = False
                        break
                if is_independent:
                    found = True
                    break
            if found:
                return k
        return 1
    
    # Faster alpha using precomputed subset masks
    subset_masks_by_size = [[] for _ in range(n + 1)]
    for r in range(1, n + 1):
        for subset in itertools.combinations(points, r):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            subset_masks_by_size[r].append(mask)
    
    def get_alpha_v2(triple_idx_list):
        """Faster independence number computation."""
        if not triple_idx_list:
            return n
        masks = [triple_masks[ti] for ti in triple_idx_list]
        for k in range(n, 0, -1):
            for s_mask in subset_masks_by_size[k]:
                is_ind = True
                for m in masks:
                    if (m & s_mask) == m:
                        is_ind = False
                        break
                if is_ind:
                    return k
        return 1
    
    # Precompute conflict matrix: conflict[i][j] = True if triples i and j share >= 2 points
    # This is a 560x560 boolean matrix
    conflict = [[False] * num_all for _ in range(num_all)]
    for i in range(num_all):
        mi = triple_masks[i]
        for j in range(i + 1, num_all):
            mj = triple_masks[j]
            shared = mi & mj
            # Count bits in shared
            if bin(shared).count('1') >= 2:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # For each triple, precompute list of conflicting triple indices
    conflict_list = [[] for _ in range(num_all)]
    for i in range(num_all):
        for j in range(num_all):
            if i != j and conflict[i][j]:
                conflict_list[i].append(j)
    
    # Map from tuple to index for O(1) lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    def find_max_independent_set(triple_idx_list):
        """Find one maximum independent set (returns the set of points)."""
        if not triple_idx_list:
            return set(points)
        masks = [triple_masks[ti] for ti in triple_idx_list]
        for k in range(n, 0, -1):
            for s_mask in subset_masks_by_size[k]:
                is_ind = True
                for m in masks:
                    if (m & s_mask) == m:
                        is_ind = False
                        break
                if is_ind:
                    # Convert mask back to set
                    result = set()
                    for p in range(n):
                        if s_mask & (1 << p):
                            result.add(p)
                    return result
        return set([0])
    
    def try_targeted_local_search(triples_list, triple_indices, current_alpha, time_limit):
        """
        Local search that specifically targets the maximum independent set.
        Tries to replace triples or add structure to break up the independent set.
        """
        best_alpha = current_alpha
        best_triples = triples_list.copy()
        best_indices = triple_indices.copy()
        
        num_triples = len(triples_list)
        
        # Track used triple indices for O(1) lookup
        used_set = set(triple_indices)
        
        iteration = 0
        while time.time() < time_limit and iteration < 500:
            iteration += 1
            
            # Find current maximum independent set
            mis = find_max_independent_set(triple_indices)
            current_alpha = len(mis)
            
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = triples_list.copy()
                best_indices = triple_indices.copy()
                if best_alpha <= 5:
                    break
            
            # Try to find a triple that covers points in the MIS
            # and doesn't conflict with existing triples
            mis_mask = 0
            for p in mis:
                mis_mask |= (1 << p)
            
            # Score candidate replacements: prefer triples that cover more MIS points
            # and don't conflict
            best_candidate = None
            best_score = -1
            
            # Try replacing a random existing triple
            if num_triples > 0:
                pos = random.randint(0, num_triples - 1)
                old_idx = triple_indices[pos]
                
                # Temporarily remove
                remaining_indices = [ti for ti in triple_indices if ti != old_idx]
                
                # Find best replacement
                candidates = [ti for ti in range(num_all) if ti not in used_set]
                random.shuffle(candidates)
                
                for cand in candidates[:200]:  # Limit candidates
                    # Check conflicts with remaining
                    conflict_found = False
                    for rem in remaining_indices:
                        if conflict[cand][rem]:
                            conflict_found = True
                            break
                    if conflict_found:
                        continue
                    
                    # Score: how many MIS points does this triple cover?
                    cand_mask = triple_masks[cand]
                    covered_mis = bin(cand_mask & mis_mask).count('1')
                    
                    # Also consider: does it cover points not in MIS? (less important)
                    score = covered_mis * 10 + random.random()
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = (cand, pos)
                
                if best_candidate is not None:
                    cand, pos = best_candidate
                    triples_list[pos] = all_triples[cand]
                    triple_indices[pos] = cand
                    used_set.discard(old_idx)
                    used_set.add(cand)
                else:
                    # No good replacement, try a different position or skip
                    continue
            else:
                # No triples yet, just add a good one
                candidates = [ti for ti in range(num_all) if ti not in used_set]
                random.shuffle(candidates)
                for cand in candidates[:100]:
                    cand_mask = triple_masks[cand]
                    covered_mis = bin(cand_mask & mis_mask).count('1')
                    if covered_mis > 0:
                        triples_list.append(all_triples[cand])
                        triple_indices.append(cand)
                        used_set.add(cand)
                        break
        
        return best_triples, best_indices, best_alpha
    
    best_alpha = n + 1
    best_triples = []
    best_indices = []
    
    # Multiple randomized greedy constructions
    while time.time() < deadline:
        # Randomized greedy construction
        indices = list(range(num_all))
        random.shuffle(indices)
        
        used_pairs = set()
        current_triples = []
        current_indices = []
        
        for t_idx in indices:
            pairs = triple_pairs[t_idx]
            if pairs[0] in used_pairs or pairs[1] in used_pairs or pairs[2] in used_pairs:
                continue
            
            used_pairs.add(pairs[0])
            used_pairs.add(pairs[1])
            used_pairs.add(pairs[2])
            
            current_triples.append(all_triples[t_idx])
            current_indices.append(t_idx)
        
        # Evaluate
        alpha = get_alpha_v2(current_indices)
        
        # Targeted local search
        remaining_time = deadline - time.time()
        if remaining_time > 5 and len(current_triples) > 0:
            ls_time = min(remaining_time - 2, 20)
            current_triples, current_indices, alpha = try_targeted_local_search(
                current_triples.copy(), current_indices.copy(), alpha, 
                time.time() + ls_time
            )
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = current_triples.copy()
            best_indices = current_indices.copy()
            if best_alpha <= 5:
                break
    
    return best_triples
