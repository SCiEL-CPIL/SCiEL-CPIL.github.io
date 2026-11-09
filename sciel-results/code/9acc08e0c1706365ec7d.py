"""Optimized construction of large 4-uniform sunflower-free family using min-degree greedy on 3-uniform hypergraph."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin before 90s kill
    
    # Generate all 4-subsets as bitmasks
    all_masks = []
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_masks.append(mask)
        all_sets.append(combo)
    
    n_total = len(all_masks)  # 2184
    
    # Precompute sunflower triples: for each pair (i,j), find all k that form a sunflower with i,j
    # A sunflower triple (a,b,c) means a∩b = a∩c = b∩c
    # For fixed pair (i,j) with intersection S = mask_i & mask_j:
    #   C must contain S, and C\S must be disjoint from (mask_i\S) and (mask_j\S)
    #   So C = S ∪ T where T is a (4-|S|)-subset of [16]\(mask_i | mask_j)
    
    # Build conflict structure: for each vertex v, store the set of other vertices
    # such that there exists a third vertex forming a sunflower.
    # Actually, we need to model this as a 3-uniform hypergraph where hyperedges are sunflower triples.
    # For min-degree greedy on independent set in hypergraph:
    #   - Select vertex with minimum "degree" (fewest hyperedges it's in)
    #   - Remove all hyperedges containing that vertex
    #   - Repeat
    
    # Alternative efficient approach: 
    # For each pair (i,j), compute the set of k that form a sunflower with i,j.
    # Then when building the family greedily, if we've already picked i and j, we cannot pick any k in that set.
    
    # Precompute: sunflower_partners[i][j] = set of k such that (i,j,k) is a sunflower triple
    # But this is O(n^3) memory which is ~10^10, too much.
    
    # Better: For each pair (i,j), compute the forbidden set on the fly.
    # For fixed i,j: S = mask_i & mask_j, |S| = s
    # Forbidden k: k must contain S, and k\S ⊆ complement of (mask_i | mask_j)
    # The available elements for k\S are [16] \ (mask_i | mask_j), which has 16 - |mask_i ∪ mask_j| elements
    # |mask_i ∪ mask_j| = 8 - s (since both have 4 elements, intersection is s)
    # So available = 16 - (8 - s) = 8 + s elements
    # We need to choose (4-s) elements from these (8+s) elements
    # Number of such k = C(8+s, 4-s)
    # For s=0: C(8,4)=70, s=1: C(9,3)=84, s=2: C(10,2)=45, s=3: C(11,1)=11, s=4: C(12,0)=1
    
    # Strategy: Greedy with random restarts, but use a smarter selection order.
    # At each step, among all remaining valid candidates, pick the one that "conflicts" with the fewest other candidates.
    # This is the min-degree heuristic.
    
    # For efficiency, we'll use a different approach:
    # 1. Start with all 2184 sets as candidates
    # 2. Repeatedly:
    #    a. For each candidate, compute its "conflict count" with already-chosen sets
    #    b. Pick the candidate with minimum conflict count (tie-break randomly)
    #    c. Add it to the family
    #    d. Remove candidates that would form a sunflower with any pair of chosen sets
    
    # Actually, let's use the hypergraph approach more directly:
    # Maintain a set of "active" hyperedges (sunflower triples not yet blocked).
    # A vertex is "free" if it's not in any active hyperedge... no, that's not right either.
    # 
    # Let me think differently. The constraint is: no three chosen sets form a sunflower.
    # So when we pick a new set c, we need to check that for every pair (a,b) already chosen,
    # (a,b,c) is NOT a sunflower.
    #
    # Efficient check: For new candidate c and existing pair (a,b):
    #   (a,b,c) is sunflower iff a∩b = a∩c = b∩c
    #   Let S = a∩b. Then we need a∩c = S and b∩c = S.
    #   This means c contains S, and c contains no elements from (a\S) ∪ (b\S).
    #
    # Precompute for each pair (a,b): the set of "forbidden patterns" for c.
    # Specifically, c is forbidden if S ⊆ c and c ∩ ((a\S) ∪ (b\S)) = ∅.
    # Equivalently, c ⊆ S ∪ ([16] \ (a∪b)) and S ⊆ c.
    # So c = S ∪ T where T ⊆ [16] \ (a∪b), |T| = 4 - |S|.
    
    # Let's precompute for each pair (i,j) the list of k that form a sunflower with i,j.
    # There are C(2184,2) ≈ 2.4M pairs, and for each we can compute the forbidden k's.
    # But storing all this is expensive. Instead, let's compute on the fly.
    
    # New approach: Use a greedy algorithm with random restarts.
    # At each step, pick the candidate that minimizes the number of "future conflicts".
    # Approximate future conflicts by the number of pairs in the current family that would
    # forbid this candidate.
    
    best_family = []
    best_size = 0
    
    max_restarts = 30
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        random.seed(restart * 104729 + 3)
        
        # Start with all sets as candidates
        candidates = list(range(n_total))
        random.shuffle(candidates)
        
        chosen = []
        chosen_masks = []
        
        # We'll use a priority-based greedy: at each step, pick the candidate
        # that has the fewest conflicts with the current family.
        
        for step in range(n_total):
            if time.time() > deadline - 5:
                break
            if not candidates:
                break
            
            # Find the candidate with minimum conflict count
            best_idx = -1
            best_conflicts = float('inf')
            
            # Sample a subset of candidates for efficiency if list is large
            if len(candidates) > 200:
                sample_size = 200
                sample = random.sample(candidates, sample_size)
            else:
                sample = candidates
            
            for idx in sample:
                mask = all_masks[idx]
                conflicts = 0
                
                for i in range(len(chosen)):
                    mask_i = chosen_masks[i]
                    for j in range(i):
                        mask_j = chosen_masks[j]
                        # Check if (mask, mask_i, mask_j) is a sunflower
                        ab = mask_i & mask_j
                        ac = mask & mask_j
                        bc = mask & mask_i
                        if ab == ac == bc:
                            conflicts += 1
                            break
                    if conflicts > 0:
                        break
                
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_idx = idx
            
            if best_idx == -1 or best_conflicts > 0:
                # All sampled candidates have conflicts; try to find any valid one
                for idx in candidates:
                    mask = all_masks[idx]
                    valid = True
                    for i in range(len(chosen)):
                        mask_i = chosen_masks[i]
                        for j in range(i):
                            mask_j = chosen_masks[j]
                            ab = mask_i & mask_j
                            ac = mask & mask_j
                            bc = mask & mask_i
                            if ab == ac == bc:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        best_idx = idx
                        best_conflicts = 0
                        break
            
            if best_idx == -1:
                break
            
            # Add best_idx to chosen
            chosen.append(best_idx)
            chosen_masks.append(all_masks[best_idx])
            
            # Remove best_idx from candidates
            candidates.remove(best_idx)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
            if best_size >= 250:
                break
    
    # Return the best family found
    result = [sorted([all_sets[i][k]+1 for k in range(4)]) for i in best_family]
    return result
