import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # 1. Generate all 4-subsets of [1, 16] as bitmasks
    all_sets = []
    for combo in combinations(range(1, 17), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append((mask, combo))
        
    N = len(all_sets)
    masks = [m for m, _ in all_sets]
    combos = [c for _, c in all_sets]
    
    # 2. Precompute intersection sizes matrix I[i][j]
    I = [[0] * N for _ in range(N)]
    for i in range(N):
        mi = masks[i]
        for j in range(i, N):
            inter = bin(mi & masks[j]).count('1')
            I[i][j] = inter
            I[j][i] = inter

    # Helper to check if adding c to chosen_indices creates a sunflower
    # Returns True if conflict, False otherwise
    # Also returns the conflicting pair if conflict exists
    def check_conflict(c, chosen_indices):
        groups = {}
        for idx in chosen_indices:
            s = I[idx][c]
            if s not in groups:
                groups[s] = []
            groups[s].append(idx)
        
        for s, group in groups.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                a = group[i]
                for j in range(i + 1, len(group)):
                    b = group[j]
                    if I[a][b] == s:
                        return True, (a, b)
        return False, None

    best_family_indices = []
    random.seed(42)
    
    while time.time() < deadline:
        # 1. Greedy Phase
        indices = list(range(N))
        random.shuffle(indices)
        chosen_indices = []
        
        for idx in indices:
            conflict, _ = check_conflict(idx, chosen_indices)
            if not conflict:
                chosen_indices.append(idx)
                
        # 2. Local Search Phase
        # Try to improve chosen_indices
        # We use a set for O(1) lookup of membership
        chosen_set = set(chosen_indices)
        
        # We will try to add sets or swap sets
        # To increase size, we need to add a set.
        # If we can't add, we swap to potentially enable future additions.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add a random set
            # Pick a random candidate not in chosen_set
            candidates = [i for i in range(N) if i not in chosen_set]
            if not candidates:
                break
                
            # Shuffle candidates to pick randomly
            random.shuffle(candidates)
            
            added = False
            for c in candidates:
                conflict, pair = check_conflict(c, chosen_indices)
                if not conflict:
                    # Add it
                    chosen_indices.append(c)
                    chosen_set.add(c)
                    improved = True
                    added = True
                    break # Restart the improvement loop with the new set
                    
            if added:
                continue
                
            # If no set could be added, try a swap
            # Pick a random set x not in family
            # Find a conflicting pair (a, b)
            # Try removing a and adding x
            # Try removing b and adding x
            
            # To make this efficient, we just pick one random candidate
            c = candidates[0]
            conflict, pair = check_conflict(c, chosen_indices)
            
            if conflict and pair:
                a, b = pair
                # Try swap a -> c
                # Remove a, add c
                new_indices = [idx for idx in chosen_indices if idx != a]
                # Check if c conflicts with new_indices
                conflict2, _ = check_conflict(c, new_indices)
                if not conflict2:
                    # Accept swap
                    chosen_indices = new_indices + [c]
                    chosen_set.remove(a)
                    chosen_set.add(c)
                    improved = True
                    continue
                
                # Try swap b -> c
                new_indices = [idx for idx in chosen_indices if idx != b]
                conflict3, _ = check_conflict(c, new_indices)
                if not conflict3:
                    # Accept swap
                    chosen_indices = new_indices + [c]
                    chosen_set.remove(b)
                    chosen_set.add(c)
                    improved = True
                    continue
        
        # Update best
        if len(chosen_indices) > len(best_family_indices):
            best_family_indices = chosen_indices[:]

    return [sorted(combos[i]) for i in best_family_indices]
