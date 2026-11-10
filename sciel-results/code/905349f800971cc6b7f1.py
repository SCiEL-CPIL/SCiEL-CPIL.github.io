import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    n = 16
    all_points = list(range(n))
    
    # Precompute all triples as masks
    # There are C(16,3) = 560 triples
    triple_list = []
    mask_to_idx = {}
    idx_to_mask = {}
    idx_to_pts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                mask = (1 << i) | (1 << j) | (1 << k)
                idx = len(triple_list)
                triple_list.append(mask)
                mask_to_idx[mask] = idx
                idx_to_mask[idx] = mask
                idx_to_pts[idx] = (i, j, k)
                
    num_triples = len(triple_list)
    
    # Precompute conflicts: two triples conflict if they share >= 2 points
    # conflicts[i] is a list of indices j such that triple i and j share >= 2 points
    # For speed, we store as a set for O(1) membership checks
    conflicts = [set() for _ in range(num_triples)]
    for i in range(num_triples):
        mi = triple_list[i]
        for j in range(i + 1, num_triples):
            mj = triple_list[j]
            common = mi & mj
            # Check if common has >= 2 bits set
            if common & (common - 1):  # Has at least 2 bits set
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Precompute point_to_triples: for each point, which triples contain it
    # This helps in local search
    point_to_triples = [[] for _ in range(n)]
    for idx, mask in enumerate(triple_list):
        for i in range(n):
            if mask & (1 << i):
                point_to_triples[i].append(idx)
                
    # Precompute masks by size for alpha calculation
    # We need to check independent sets of size k
    # Independent set S: no edge is subset of S
    # Equivalent: for all e in edges, (S & e) != e
    
    # We'll use a different approach for alpha calculation:
    # For a given set of edges, alpha is the max size of a set with no edge fully contained.
    # We can compute this by checking subsets of size 5, 4, 3, 2, 1 in descending order.
    
    # Precompute all masks of size 5 (4368), size 4 (1820), size 3 (560), size 2 (120)
    # We only need to check these for the current edge set.
    
    # For faster alpha check, we can precompute for each mask of size 5,
    # which triples are subsets of it.
    
    # Actually, let's use a smarter approach:
    # For a given edge set E, a subset S of size k is independent iff
    # no triple in E is fully contained in S.
    
    # We can precompute for each mask of size 5 (or 4, etc.) the list of triples that are subsets.
    # Then for a given edge set, we check if any of those triples are in the edge set.
    
    # Let's precompute for size 5:
    # For each mask of size 5, store the list of triple indices that are subsets of it.
    
    masks_size_5 = []
    mask_5_to_triples = {}
    for mask in range(1 << n):
        if bin(mask).count('1') == 5:
            masks_size_5.append(mask)
            # Find all triples that are subsets of this mask
            triples_in_mask = []
            for t_idx in range(num_triples):
                t_mask = triple_list[t_idx]
                if (t_mask & mask) == t_mask:
                    triples_in_mask.append(t_idx)
            mask_5_to_triples[mask] = triples_in_mask
            
    # Similarly for size 4
    masks_size_4 = []
    mask_4_to_triples = {}
    for mask in range(1 << n):
        if bin(mask).count('1') == 4:
            masks_size_4.append(mask)
            triples_in_mask = []
            for t_idx in range(num_triples):
                t_mask = triple_list[t_idx]
                if (t_mask & mask) == t_mask:
                    triples_in_mask.append(t_idx)
            mask_4_to_triples[mask] = triples_in_mask
            
    # For size 3, the only triple subset is the mask itself (if it's a valid triple)
    # For size 2 and 1, no triple can be a subset, so they're always independent (if we only have triples of size 3)
    
    def calc_alpha_fast(edges_set):
        """
        Calculate the independence number of the hypergraph with edges in edges_set.
        Returns the maximum size of a subset with no edge fully contained.
        """
        if not edges_set:
            return n
            
        # Check size 5
        for mask in masks_size_5:
            # Check if any triple in mask_5_to_triples[mask] is in edges_set
            found = False
            for t_idx in mask_5_to_triples[mask]:
                if t_idx in edges_set:
                    found = True
                    break
            if not found:
                return 5
                
        # Check size 4
        for mask in masks_size_4:
            found = False
            for t_idx in mask_4_to_triples[mask]:
                if t_idx in edges_set:
                    found = True
                    break
            if not found:
                return 4
                
        # Check size 3: a set of size 3 is independent iff it doesn't contain any edge
        # But wait, if the set itself is an edge, then it's not independent.
        # If the set contains an edge, it's not independent.
        # For size 3, the only way it contains an edge is if the set IS an edge (since edges have size 3).
        # So a size 3 set is independent iff it is not in edges_set.
        
        # We need to check if there exists a size 3 set not in edges_set.
        # There are 560 size 3 sets. If edges_set has fewer than 560 edges, there might be one.
        # But we want the MAX size. So if no size 5 or 4 is independent, check size 3.
        
        # Actually, we should check all size 3 sets? That's 560 checks.
        # Or we can check if the number of edges is less than 560, then there's a size 3 set not in edges_set.
        # But that's not quite right: we need a specific size 3 set that is not an edge.
        
        # Let's just iterate through all size 3 masks (which are exactly the triples)
        # and check if any is not in edges_set.
        
        for t_idx in range(num_triples):
            if t_idx not in edges_set:
                return 3
                
        # If all triples are in edges_set, then no size 3 set is independent.
        # Then check size 2: any size 2 set is independent (since no edge of size 3 can be subset of size 2)
        return 2
        
    # Actually, the above calc_alpha_fast has a bug:
    # For size 5, if we find one independent, we return 5. That's correct.
    # For size 4, if we find one independent, we return 4. That's correct.
    # For size 3, if we find one not in edges_set, we return 3. That's correct.
    # But we need to be careful: we should check size 5 first, then 4, then 3, then 2.
    # The logic above does that.
    
    # However, the loop over masks_size_5 (4368 masks) and for each, checking up to 10 triples
    # (C(5,3)=10 triples per 5-set) might be slow if done many times.
    # 4368 * 10 = 43680 checks per alpha calculation.
    # If we do this 10000 times, that's 436 million checks, which might be too slow in Python.
    
    # Let's optimize: instead of checking all 5-sets, we can use a different approach.
    
    # Alternative: for each edge, it "covers" all 5-sets that contain it.
    # A 5-set is independent iff it is not covered by any edge.
    # We can maintain a count of how many edges cover each 5-set.
    # Then a 5-set is independent iff its count is 0.
    
    # Let's precompute for each 5-set, which edges cover it (i.e., are subsets of it).
    # We already have mask_5_to_triples for this.
    
    # Let's use a different alpha calculation that is faster on average:
    # We'll sample random 5-sets and check if they're independent.
    # If we find one, alpha >= 5.
    # If we don't find one after many samples, we check 4-sets, etc.
    
    # But for correctness, we need to find the EXACT alpha, not just an approximation.
    # The evaluator checks over all subsets, so we need the true alpha.
    
    # Let's stick with the exact calculation but optimize the inner loop.
    
    # Actually, let's precompute for each 5-set a bitmask of the triples that cover it.
    # Then for a given edge set, we can check if the intersection is empty.
    # But edge set is a set of indices, not a bitmask.
    
    # Let's use a different representation: maintain a boolean array or set of active edges.
    
    # For now, let's use the calc_alpha_fast above and hope it's fast enough.
    # 4368 * 10 = 43680 operations per call. In Python, that's ~1ms.
    # If we do 10000 calls, that's 10 seconds. That's acceptable within 130s.
    
    # Let's refine calc_alpha_fast to be more efficient:
    
    def calc_alpha(edges_set):
        if not edges_set:
            return n
            
        # Check size 5
        # For each 5-set, check if any of its 10 triples is in edges_set
        for mask in masks_size_5:
            # Unroll the check for speed?
            # mask_5_to_triples[mask] is a list of up to 10 indices
            for t_idx in mask_5_to_triples[mask]:
                if t_idx in edges_set:
                    break
            else:
                # No triple found, so this 5-set is independent
                return 5
                
        # Check size 4
        for mask in masks_size_4:
            for t_idx in mask_4_to_triples[mask]:
                if t_idx in edges_set:
                    break
            else:
                return 4
                
        # Check size 3
        # A 3-set is independent iff it is not an edge
        # We can check if len(edges_set) < 560, then there must be a 3-set not in edges_set
        if len(edges_set) < num_triples:
            return 3
            
        # Check size 2
        return 2
        
    def is_valid_add(triple_idx, edges_set):
        """Check if triple_idx can be added to edges_set without conflict."""
        for c in conflicts[triple_idx]:
            if c in edges_set:
                return False
        return True
        
    def greedy_pack():
        """Greedy construction of a linear triple system."""
        current_triples = set()
        available = list(range(num_triples))
        random.shuffle(available)
        
        for idx in available:
            if is_valid_add(idx, current_triples):
                current_triples.add(idx)
                
        return current_triples
        
    def random_swap(current_triples):
        """
        Perform a random swap: remove one edge, add one valid edge.
        Returns new set or None if no move possible.
        """
        if not current_triples:
            return None
            
        to_remove = random.choice(list(current_triples))
        new_triples = set(current_triples)
        new_triples.remove(to_remove)
        
        # Find a valid add
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        
        for idx in candidates:
            if idx in new_triples:
                continue
            if is_valid_add(idx, new_triples):
                new_triples.add(idx)
                return new_triples
        return None
        
    def targeted_swap(current_triples, witness_mask):
        """
        Try to fix the independent set 'witness_mask' by adding an edge inside it.
        """
        # Get all triples inside witness_mask
        points = []
        for i in range(n):
            if witness_mask & (1 << i):
                points.append(i)
                
        # Generate all triples from these points
        local_triples = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                for k in range(j + 1, len(points)):
                    mask = (1 << points[i]) | (1 << points[j]) | (1 << points[k])
                    idx = mask_to_idx.get(mask)
                    if idx is not None:
                        local_triples.append(idx)
                        
        random.shuffle(local_triples)
        
        for t_idx in local_triples:
            if t_idx in current_triples:
                continue
                
            # Check if t_idx can be added directly
            if is_valid_add(t_idx, current_triples):
                new_triples = set(current_triples)
                new_triples.add(t_idx)
                return new_triples
            else:
                # Find conflicts
                conflicts_in_set = [c for c in conflicts[t_idx] if c in current_triples]
                
                for c_idx in conflicts_in_set:
                    new_triples = set(current_triples)
                    new_triples.remove(c_idx)
                    if is_valid_add(t_idx, new_triples):
                        new_triples.add(t_idx)
                        return new_triples
        return None
        
    best_alpha = n
    best_triples = None
    
    # Start with a greedy pack
    current_triples = greedy_pack()
    current_alpha = calc_alpha(current_triples)
    
    if current_alpha < best_alpha:
        best_alpha = current_alpha
        best_triples = set(current_triples)
        
    # Local Search with simulated annealing
    T = 1.0
    cooling_rate = 0.995
    min_T = 0.05
    
    while time.time() < deadline:
        current_alpha = calc_alpha(current_triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = set(current_triples)
            if best_alpha <= 4:
                # Good solution found, continue to see if we can improve
                pass
        
        # Generate Move
        move_type = random.random()
        new_triples = None
        
        if current_alpha > 4 and move_type < 0.7:
            # Targeted move on a witness
            target_size = 5 if current_alpha >= 5 else current_alpha
            # Find one witness
            witness = None
            if target_size == 5:
                candidates = random.sample(masks_size_5, min(50, len(masks_size_5)))
            elif target_size == 4:
                candidates = random.sample(masks_size_4, min(50, len(masks_size_4)))
            else:
                candidates = []
                
            for m in candidates:
                # Check if m is independent
                if target_size == 5:
                    found = False
                    for t_idx in mask_5_to_triples[m]:
                        if t_idx in current_triples:
                            found = True
                            break
                    if not found:
                        witness = m
                        break
                elif target_size == 4:
                    found = False
                    for t_idx in mask_4_to_triples[m]:
                        if t_idx in current_triples:
                            found = True
                            break
                    if not found:
                        witness = m
                        break
                        
            if witness:
                new_triples = targeted_swap(current_triples, witness)
                
        if new_triples is None:
            new_triples = random_swap(current_triples)
            
        if new_triples is None:
            # Perturb
            for _ in range(random.randint(2, 3)):
                if current_triples:
                    current_triples.remove(random.choice(list(current_triples)))
            continue
            
        new_alpha = calc_alpha(new_triples)
        
        # Acceptance
        delta = new_alpha - current_alpha
        if delta <= 0:
            current_triples = new_triples
            current_alpha = new_alpha
            if delta < 0:
                T = max(T, 0.5)
        else:
            if random.random() < (2.71828 ** (-delta / T)):
                current_triples = new_triples
                current_alpha = new_alpha
                
        T *= cooling_rate
        if T < min_T:
            T = min_T
            if random.random() < 0.1:
                for _ in range(2):
                    if current_triples:
                        current_triples.remove(random.choice(list(current_triples)))
                        
    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    result = []
    for idx in best_triples:
        pts = list(idx_to_pts[idx])
        result.append(pts)
        
    return result
