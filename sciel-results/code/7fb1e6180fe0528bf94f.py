import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    n = 16
    points = list(range(n))
    all_triples = list(itertools.combinations(points, 3))
    num_triples = len(all_triples)  # 560
    
    # Precompute bitmasks and pairs for each triple
    triple_masks = [0] * num_triples
    triple_pairs = []
    for i, t in enumerate(all_triples):
        mask = 0
        a, b, c = t
        mask |= (1 << a)
        mask |= (1 << b)
        mask |= (1 << c)
        triple_masks[i] = mask
        triple_pairs.append([(a, b), (a, c), (b, c)])
    
    # Precompute conflict graph: for each triple, which other triples conflict with it
    # Two triples conflict if they share 2 or more points (i.e., share a pair)
    # We'll represent conflicts as a bitmask of triple indices for fast checking
    # Actually, 560 bits won't fit in a Python int efficiently for bitwise ops in tight loops
    # Instead, precompute for each triple the set of conflicting triple indices
    conflict_sets = [set() for _ in range(num_triples)]
    for i in range(num_triples):
        for j in range(i + 1, num_triples):
            mi = triple_masks[i]
            mj = triple_masks[j]
            if (mi & mj).bit_count() >= 2:
                conflict_sets[i].add(j)
                conflict_sets[j].add(i)
    
    # Precompute pair masks for fast pair-in-mask checking
    # For a given mask, we want to know if a specific pair (a,b) is contained
    # We can precompute for each pair the bitmask of points
    pair_point_mask = {}
    for a in range(n):
        for b in range(a + 1, n):
            pair_point_mask[(a, b)] = (1 << a) | (1 << b)
    
    # Precompute masks for subsets for independence number computation
    # We'll use a smarter approach: compute alpha by checking all subsets from large to small
    # But that's expensive. Instead, we'll use a randomized approach or a branch-and-bound.
    # For now, let's precompute all subset masks up to size 8 (since we expect alpha <= 8)
    # Actually, let's just compute alpha on the fly using a greedy + verification approach.
    
    def compute_alpha(triple_mask_list):
        """
        Compute the independence number: largest subset of points containing no triple.
        A subset S is independent if no triple is entirely contained in S.
        """
        if not triple_mask_list:
            return n
        
        # Build a set of triple masks for fast lookup
        triple_mask_set = set(triple_mask_list)
        
        # Try subsets from size n down to 1
        # Optimize: start from a reasonable upper bound
        # Upper bound: if we have k triples, each covers 3 points, so at most 16 - 3k/3 = 16-k points can be "covered"
        # But independence is about NOT containing any triple, so it's different.
        # Let's just iterate from high to low.
        
        for k in range(n, 0, -1):
            for subset in itertools.combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                # Check if any triple is contained in this subset
                independent = True
                for tm in triple_mask_list:
                    if (tm & mask) == tm:
                        independent = False
                        break
                if independent:
                    return k
        return 1
    
    def fast_compute_alpha(triple_mask_list, cached_result=None):
        """
        Faster alpha computation using bit operations.
        """
        if not triple_mask_list:
            return n
        
        # Convert to a set of masks for O(1) lookup
        triple_mask_set = set(triple_mask_list)
        
        # We can prune: if a subset of size k is independent, all subsets of size < k are also independent
        # So we just need to find the largest k.
        
        # Optimization: start from a high k and go down
        # Upper bound on alpha: 16 - floor(num_triples / 3) is a weak bound
        # Better: use the fact that each triple "hides" 3 points, but they overlap
        
        for k in range(n, 0, -1):
            for subset in itertools.combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                # Check if any triple is a subset of mask
                # We can iterate over triples and check if triple_mask is subset of mask
                independent = True
                for tm in triple_mask_list:
                    if (tm & mask) == tm:
                        independent = False
                        break
                if independent:
                    return k
        return 1
    
    # Even faster: precompute for each possible mask (2^16 = 65536) whether it's "covered" by any triple
    # This is too large. Instead, let's use a different approach.
    
    # Let's use a branch and bound or a simpler greedy approach for alpha.
    # Actually, for 16 points, iterating over all subsets up to size 8 is feasible:
    # C(16,8) = 12870, and for each we check against ~20-30 triples. That's ~400k ops per alpha call.
    # With ~1000 alpha calls, that's 400M ops, which might be too slow in Python.
    
    # Let's optimize by precomputing all subset masks and using bit operations.
    
    # Precompute all subset masks for sizes 1 to 16
    all_subset_masks = [[] for _ in range(n + 1)]
    for r in range(1, n + 1):
        for subset in itertools.combinations(points, r):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            all_subset_masks[r].append(mask)
    
    def compute_alpha_fast(triple_mask_list):
        """
        Compute alpha using precomputed subset masks.
        """
        if not triple_mask_list:
            return n
        
        triple_mask_set = set(triple_mask_list)
        
        for k in range(n, 0, -1):
            for s_mask in all_subset_masks[k]:
                # Check if any triple is contained in s_mask
                independent = True
                for tm in triple_mask_set:
                    if (tm & s_mask) == tm:
                        independent = False
                        break
                if independent:
                    return k
        return 1
    
    # Now, the main search algorithm:
    # 1. Greedy construction: add triples that cover the most uncovered point pairs
    # 2. Local search: swap triples to reduce alpha
    
    def greedy_construct():
        """
        Greedy construction: repeatedly add the triple that covers the most 
        currently uncovered point pairs, ensuring linearity.
        """
        used_pairs = set()
        current_triples = []
        current_masks = []
        
        # Track which pairs are covered
        all_pairs = set()
        for a in range(n):
            for b in range(a + 1, n):
                all_pairs.add((a, b))
        
        uncovered_pairs = set(all_pairs)
        
        # For each triple, precompute which pairs it covers
        # We already have triple_pairs
        
        while True:
            if time.time() > deadline:
                break
            
            # Find the triple that covers the most uncovered pairs
            best_triple_idx = -1
            best_count = 0
            best_pairs = None
            
            for i in range(num_triples):
                pairs = triple_pairs[i]
                # Check if all pairs are uncovered
                if all(p in uncovered_pairs for p in pairs):
                    count = 3  # Each triple covers exactly 3 pairs
                    # All triples cover 3 pairs, so we need a better criterion
                    # Let's use: maximize the number of points that are in uncovered pairs
                    # Or: minimize the number of points already "used" by other triples
                    # Actually, for a linear triple system, all valid triples cover 3 uncovered pairs.
                    # So we need a tie-breaker. Let's use: maximize the spread of points.
                    # Or: prefer triples that involve points with fewer connections.
                    
                    # For now, just pick the first one that works
                    if best_triple_idx == -1:
                        best_triple_idx = i
                        best_pairs = pairs
                        best_count = 3
                    # Tie-breaker: prefer triples with points that have lower degree
                    # Count how many times each point appears in current_triples
                    # This is expensive, so let's use a simpler heuristic
                    
            if best_triple_idx == -1:
                break
            
            # Add the triple
            t = all_triples[best_triple_idx]
            current_triples.append(t)
            current_masks.append(triple_masks[best_triple_idx])
            
            # Mark its pairs as covered
            for p in best_pairs:
                uncovered_pairs.discard(p)
                used_pairs.add(p)
        
        return current_triples, current_masks
    
    def local_search(triples_list, masks_list, current_alpha, time_limit):
        """
        Local search: try swapping triples to reduce alpha.
        Key insight: focus on triples that intersect the current maximum independent set.
        """
        best_alpha = current_alpha
        best_triples = triples_list.copy()
        best_masks = masks_list.copy()
        
        num_current = len(triples_list)
        
        # Build used triple indices
        used_indices = set()
        for t in triples_list:
            idx = all_triples.index(t)
            used_indices.add(idx)
        
        # Find the current maximum independent set
        def find_max_independent_set(triple_mask_list):
            """Find one maximum independent set."""
            if not triple_mask_list:
                return set(points)
            
            triple_mask_set = set(triple_mask_list)
            
            for k in range(n, 0, -1):
                for subset in itertools.combinations(points, k):
                    mask = 0
                    for p in subset:
                        mask |= (1 << p)
                    independent = True
                    for tm in triple_mask_set:
                        if (tm & mask) == tm:
                            independent = False
                            break
                    if independent:
                        return set(subset)
            return set()
        
        # Iteratively try to improve
        iteration = 0
        while time.time() < time_limit:
            iteration += 1
            
            # Find current max independent set
            max_indep = find_max_independent_set(best_masks)
            
            if len(max_indep) <= best_alpha:
                # Try to improve further
                pass
            
            # Try swapping: for each triple in the current solution, try replacing it
            # with a triple that covers points in the max independent set
            improved = False
            
            # Shuffle positions for randomness
            positions = list(range(num_current))
            random.shuffle(positions)
            
            for pos in positions:
                if time.time() > time_limit:
                    break
                
                old_triple = best_triples[pos]
                old_mask = best_masks[pos]
                old_idx = all_triples.index(old_triple)
                
                # Remove old triple
                temp_triples = best_triples.copy()
                temp_masks = best_masks.copy()
                temp_triples[pos] = None
                temp_masks[pos] = None
                used_indices.discard(old_idx)
                
                # Try to find a replacement that covers points in max_indep
                # Prefer triples that have 2 or 3 points in max_indep
                candidates = []
                for i in range(num_triples):
                    if i in used_indices:
                        continue
                    
                    # Check if candidate conflicts with other triples
                    cand_pairs = triple_pairs[i]
                    conflict = False
                    for j in range(num_current):
                        if j == pos:
                            continue
                        other_mask = temp_masks[j]
                        if other_mask is None:
                            continue
                        for p in cand_pairs:
                            pm = pair_point_mask[p]
                            if (other_mask & pm) == pm:
                                conflict = True
                                break
                        if conflict:
                            break
                    
                    if conflict:
                        continue
                    
                    # Count how many points of candidate are in max_indep
                    cand_mask = triple_masks[i]
                    count_in_indep = (cand_mask & sum((1 << p) for p in max_indep)).bit_count()
                    candidates.append((count_in_indep, i))
                
                if not candidates:
                    # Restore
                    best_triples = temp_triples.copy()
                    best_masks = temp_masks.copy()
                    used_indices.add(old_idx)
                    continue
                
                # Sort by count_in_indep descending
                candidates.sort(reverse=True)
                
                # Try the best candidate
                found = False
                for count, cand_idx in candidates[:5]:  # Try top 5
                    if count < 2:
                        break  # Not worth trying
                    
                    cand_triple = all_triples[cand_idx]
                    
                    temp_triples[pos] = cand_triple
                    temp_masks[pos] = triple_masks[cand_idx]
                    used_indices.add(cand_idx)
                    
                    # Evaluate new alpha
                    valid_masks = [m for m in temp_masks if m is not None]
                    new_alpha = compute_alpha_fast(valid_masks)
                    
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_triples = temp_triples.copy()
                        best_masks = temp_masks.copy()
                        improved = True
                        found = True
                        if best_alpha <= 5:
                            break
                    else:
                        # Restore
                        used_indices.discard(cand_idx)
                
                if not found:
                    # Restore
                    temp_triples[pos] = old_triple
                    temp_masks[pos] = old_mask
                    used_indices.add(old_idx)
                
                if best_alpha <= 5:
                    break
            
            if not improved and iteration > 50:
                break
        
        return best_triples, best_masks, best_alpha
    
    # Main search loop
    best_alpha = n + 1
    best_triples = []
    
    # Try multiple random restarts with greedy construction
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Randomize the greedy order
        random.seed(restart * 12345 + 67890)
        
        # Greedy construction with randomization
        used_pairs = set()
        current_triples = []
        current_masks = []
        
        all_pairs = set()
        for a in range(n):
            for b in range(a + 1, n):
                all_pairs.add((a, b))
        
        uncovered_pairs = set(all_pairs)
        
        # Build a list of candidate triples, shuffled
        triple_order = list(range(num_triples))
        random.shuffle(triple_order)
        
        for t_idx in triple_order:
            if time.time() > deadline - 10:
                break
            
            pairs = triple_pairs[t_idx]
            if all(p in uncovered_pairs for p in pairs):
                t = all_triples[t_idx]
                current_triples.append(t)
                current_masks.append(triple_masks[t_idx])
                for p in pairs:
                    uncovered_pairs.discard(p)
                    used_pairs.add(p)
        
        if not current_triples:
            continue
        
        # Evaluate alpha
        alpha = compute_alpha_fast(current_masks)
        
        # Local search
        remaining_time = deadline - 10 - time.time()
        if remaining_time > 2:
            ls_triples, ls_masks, ls_alpha = local_search(
                current_triples.copy(), current_masks.copy(), alpha,
                time.time() + min(remaining_time - 1, 20)
            )
            if ls_alpha < alpha:
                current_triples = ls_triples
                current_masks = ls_masks
                alpha = ls_alpha
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = current_triples.copy()
            if best_alpha <= 5:
                break
    
    return best_triples
