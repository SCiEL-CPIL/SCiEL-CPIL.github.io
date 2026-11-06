import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    # Elements are 1..16, so bits 1..16
    candidates = []
    candidate_tuples = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        candidates.append(mask)
        candidate_tuples.append(comb)
        
    N = len(candidates)
    
    # Precompute pairwise intersections for speed
    # pair_int[i][j] = candidates[i] & candidates[j]
    pair_int = [[0] * N for _ in range(N)]
    for i in range(N):
        ci = candidates[i]
        for j in range(i + 1, N):
            val = ci & candidates[j]
            pair_int[i][j] = val
            pair_int[j][i] = val

    def can_add(family_set, k_idx):
        """
        Check if candidate k_idx can be added to family (set of indices)
        without forming a sunflower.
        Uses O(|F|^2) checks but with set membership for speed.
        """
        ck = candidates[k_idx]
        
        # For each pair in the family, check if adding k creates a sunflower
        # A sunflower exists if there are a, b in family such that:
        # |a ∩ k| = |b ∩ k| = |a ∩ b|
        # i.e., pair_int[a][k] == pair_int[b][k] == pair_int[a][b]
        
        # We can optimize by grouping elements by their intersection with k
        # For each possible intersection value, count how many elements in family have that intersection
        # If any value appears 2+ times AND those two elements also have that same intersection with each other,
        # then we have a sunflower.
        
        # Actually, simpler: for each pair (a,b) in family, check the condition.
        # But that's O(|F|^2). With |F| up to ~200, that's 40000 checks per can_add call.
        # With N=1820 candidates, that's 1820 * 40000 = 72.8M operations per greedy pass.
        # That might be too slow.
        
        # Better approach: For candidate k, compute its intersection with every element in family.
        # Then, for each pair of elements in family that have the same intersection value with k,
        # check if their mutual intersection equals that value.
        
        # Let's use a dictionary: intersection_value -> list of indices in family
        int_map = {}
        for idx in family_set:
            val = pair_int[idx][k_idx]
            if val not in int_map:
                int_map[val] = []
            int_map[val].append(idx)
        
        # For each group with 2+ elements, check if any pair in the group has mutual intersection equal to the group value
        for val, indices in int_map.items():
            if len(indices) < 2:
                continue
            # Check all pairs in this group
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    a = indices[i]
                    b = indices[j]
                    if pair_int[a][b] == val:
                        return False
        return True

    def greedy_fill(family_set, order):
        """
        Greedily add candidates from 'order' to 'family_set' if valid.
        Returns the number of added elements.
        """
        added = 0
        for idx in order:
            if idx in family_set:
                continue
            if can_add(family_set, idx):
                family_set.add(idx)
                added += 1
        return added

    best_family = set()
    best_size = 0
    
    # List of all indices
    all_indices = list(range(N))
    
    # Phase 1: Multiple randomized greedy starts
    while time.time() < deadline - 10.0:
        random.shuffle(all_indices)
        family_set = set()
        
        # Greedy phase
        for idx in all_indices:
            if can_add(family_set, idx):
                family_set.add(idx)
        
        if len(family_set) > best_size:
            best_size = len(family_set)
            best_family = family_set.copy()
            
        # Try to add any remaining valid candidates (in case order mattered)
        remaining = [i for i in all_indices if i not in family_set]
        random.shuffle(remaining)
        for idx in remaining:
            if can_add(family_set, idx):
                family_set.add(idx)
        
        if len(family_set) > best_size:
            best_size = len(family_set)
            best_family = family_set.copy()
        
        # Phase 2: Local search with kicks
        kicks = 0
        max_kicks = 100
        while time.time() < deadline - 5.0 and kicks < max_kicks:
            kicks += 1
            
            if not family_set:
                break
            
            # Pick a random element to remove
            victim_idx = random.choice(list(family_set))
            family_set.remove(victim_idx)
            
            old_size = len(family_set)
            
            # Try to greedily fill with any candidates not in family
            candidates_to_try = [i for i in all_indices if i not in family_set]
            random.shuffle(candidates_to_try)
            
            for idx in candidates_to_try:
                if can_add(family_set, idx):
                    family_set.add(idx)
            
            new_size = len(family_set)
            
            if new_size > old_size:
                # Improvement!
                if new_size > best_size:
                    best_size = new_size
                    best_family = family_set.copy()
                # Reset kicks to allow more search on this good family
                kicks = 0 
                break 
            elif new_size == old_size:
                # Same size. Accept with low probability to randomize
                if random.random() < 0.1:
                    if new_size > best_size:
                        best_size = new_size
                        best_family = family_set.copy()
                    # Continue kicking
                    pass
                else:
                    # Restore victim
                    family_set.add(victim_idx)
            else:
                # Worse size
                family_set.add(victim_idx)

    # Phase 3: If we have time, do more sophisticated local search
    # Try removing 2 elements at a time and refilling
    if time.time() < deadline - 5.0 and best_family:
        family_set = best_family.copy()
        improvements = 0
        while time.time() < deadline - 5.0 and improvements < 50:
            improvements += 1
            # Try removing 2 random elements
            if len(family_set) < 2:
                break
            victim1, victim2 = random.sample(list(family_set), 2)
            family_set.remove(victim1)
            family_set.remove(victim2)
            
            old_size = len(family_set)
            
            # Try to greedily fill
            candidates_to_try = [i for i in all_indices if i not in family_set]
            random.shuffle(candidates_to_try)
            
            for idx in candidates_to_try:
                if can_add(family_set, idx):
                    family_set.add(idx)
            
            new_size = len(family_set)
            
            if new_size > old_size:
                if new_size > best_size:
                    best_size = new_size
                    best_family = family_set.copy()
                    improvements = 0
            elif new_size == old_size:
                if random.random() < 0.05:
                    if new_size > best_size:
                        best_size = new_size
                        best_family = family_set.copy()
                else:
                    family_set.add(victim1)
                    family_set.add(victim2)
            else:
                family_set.add(victim1)
                family_set.add(victim2)

    # Convert best_family (set of indices) to a list of tuples
    result = []
    for idx in best_family:
        result.append(candidate_tuples[idx])
        
    return result
