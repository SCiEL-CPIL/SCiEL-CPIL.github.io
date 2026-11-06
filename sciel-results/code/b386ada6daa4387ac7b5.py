import random
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # Precompute all 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    n_sets = len(all_sets)  # 1820
    
    def popcount(x):
        return bin(x).count('1')
    
    def is_sunflower(a, b, c):
        ab = a & b
        ac = a & c
        bc = b & c
        return ab == ac == bc
    
    def can_add(mask, chosen_masks):
        """Check if mask can be added to chosen_masks without forming a sunflower."""
        for i in range(len(chosen_masks)):
            for j in range(i + 1, len(chosen_masks)):
                if is_sunflower(mask, chosen_masks[i], chosen_masks[j]):
                    return False
        return True
    
    def greedy_with_conflict_score(seed):
        """Greedy that prioritizes sets with fewer conflicts."""
        rng = random.Random(seed)
        indices = list(range(n_sets))
        rng.shuffle(indices)
        
        chosen = []
        chosen_masks = []
        
        # For each candidate, compute how many pairs it conflicts with
        for idx in indices:
            mask = all_sets[idx]
            # Count conflicts
            conflicts = 0
            for i in range(len(chosen_masks)):
                for j in range(i + 1, len(chosen_masks)):
                    if is_sunflower(mask, chosen_masks[i], chosen_masks[j]):
                        conflicts += 1
            # Add if no conflicts
            if conflicts == 0:
                chosen.append(idx)
                chosen_masks.append(mask)
        
        return chosen
    
    def local_search_improve(chosen):
        """Try to improve by swapping or adding."""
        chosen_set = set(chosen)
        chosen_masks = [all_sets[i] for i in chosen]
        improved = True
        while improved:
            improved = False
            # Try to add any set
            for idx in range(n_sets):
                if idx in chosen_set:
                    continue
                if can_add(all_sets[idx], chosen_masks):
                    chosen.append(idx)
                    chosen_masks.append(all_sets[idx])
                    chosen_set.add(idx)
                    improved = True
                    break
            if improved:
                continue
            
            # Try 1-for-2 swaps
            for i in range(len(chosen)):
                if time.time() > deadline:
                    return chosen
                temp_chosen = [x for j, x in enumerate(chosen) if j != i]
                temp_set = set(temp_chosen)
                temp_masks = [all_sets[x] for x in temp_chosen]
                
                added_count = 0
                added_indices = []
                for idx in range(n_sets):
                    if idx in temp_set or idx == chosen[i]:
                        continue
                    if can_add(all_sets[idx], temp_masks):
                        added_indices.append(idx)
                        temp_set.add(idx)
                        temp_masks.append(all_sets[idx])
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if added_count >= 2:
                    chosen = temp_chosen + added_indices
                    chosen_set = set(chosen)
                    improved = True
                    break
        
        return chosen
    
    best_chosen = []
    best_size = 0
    
    # Multiple random restarts with different strategies
    for seed in range(500):
        if time.time() > deadline:
            break
        
        # Alternate between different greedy strategies
        if seed % 3 == 0:
            # Standard random greedy
            rng = random.Random(seed)
            indices = list(range(n_sets))
            rng.shuffle(indices)
            chosen = []
            chosen_masks = []
            for idx in indices:
                mask = all_sets[idx]
                if can_add(mask, chosen_masks):
                    chosen.append(idx)
                    chosen_masks.append(mask)
        elif seed % 3 == 1:
            # Greedy with conflict scoring (prefer low-conflict sets)
            chosen = greedy_with_conflict_score(seed)
        else:
            # Reverse greedy (start from end of shuffled list)
            rng = random.Random(seed)
            indices = list(range(n_sets))
            rng.shuffle(indices)
            indices.reverse()
            chosen = []
            chosen_masks = []
            for idx in indices:
                mask = all_sets[idx]
                if can_add(mask, chosen_masks):
                    chosen.append(idx)
                    chosen_masks.append(mask)
        
        if len(chosen) > best_size:
            best_chosen = chosen
            best_size = len(chosen)
    
    # Local search on best
    if time.time() < deadline - 10:
        best_chosen = local_search_improve(best_chosen)
        best_size = len(best_chosen)
    
    # Convert back to sorted lists
    return [sorted([i+1 for i in range(16) if all_sets[idx] & (1 << i)]) for idx in best_chosen]
