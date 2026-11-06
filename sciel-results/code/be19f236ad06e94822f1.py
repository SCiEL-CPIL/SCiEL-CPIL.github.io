import itertools
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 16
    k = 4
    
    # Generate all 4-element subsets
    all_subsets = list(itertools.combinations(range(1, n + 1), k))
    num_subsets = len(all_subsets)
    
    # Bitmasks
    masks = [0] * num_subsets
    for i, s in enumerate(all_subsets):
        m = 0
        for bit in s:
            m |= (1 << (bit - 1))
        masks[i] = m
    
    # Precompute intersection matrix
    # inter[i][j] = masks[i] & masks[j]
    # Using numpy for fast access and potential vectorization if needed, 
    # but standard list of lists or array of arrays is also fine.
    # 1820x1820 is ~3.3M entries. uint16 is 6.6MB.
    inter = np.zeros((num_subsets, num_subsets), dtype=np.uint16)
    for i in range(num_subsets):
        mi = masks[i]
        for j in range(i + 1, num_subsets):
            val = mi & masks[j]
            inter[i, j] = val
            inter[j, i] = val
            
    def can_add(C, F):
        """
        Check if subset C can be added to family F without forming a sunflower.
        F is a list of indices.
        """
        len_F = len(F)
        for i in range(len_F):
            Ai = F[i]
            ic = inter[Ai, C]
            for j in range(i + 1, len_F):
                Bj = F[j]
                ij = inter[Ai, Bj]
                if ij == ic and ij == inter[Bj, C]:
                    return False
        return True

    def is_sunflower_triple(B, C, x):
        """
        Check if B, C, x form a sunflower.
        """
        return inter[B, C] == inter[B, x] == inter[C, x]

    def greedy_construct(order):
        family = []
        for idx in order:
            if time.time() > deadline:
                break
            if can_add(idx, family):
                family.append(idx)
        return family

    # 1. Initial Greedy Phase
    best_family = []
    random.seed(42)
    
    # Try a few random orderings
    for _ in range(5):
        order = list(range(num_subsets))
        random.shuffle(order)
        fam = greedy_construct(order)
        if len(fam) > len(best_family):
            best_family = fam[:]
            
    # 2. Local Search Phase (1-out, 2-in)
    # We try to increase the size of best_family
    current_family = best_family[:]
    current_set = set(current_family)
    all_indices = set(range(num_subsets))
    
    # To speed up, we can maintain a list of candidates not in family
    # But checking can_add is the bottleneck.
    
    improvements = 0
    no_improve_streak = 0
    
    while time.time() < deadline:
        if not current_family:
            break
            
        # Pick a random element to remove
        # We might want to pick elements that are "critical" (conflict with many others)
        # but random is a good start.
        A = random.choice(current_family)
        
        # Remove A
        F_temp = [x for x in current_family if x != A]
        F_temp_set = set(F_temp)
        
        # Find candidates that can be added to F_temp
        # We iterate through all indices not in F_temp
        # This is O(N) * can_add cost
        # can_add is O(|F|^2)
        
        valid_candidates = []
        for C in all_indices:
            if C in F_temp_set:
                continue
            if time.time() > deadline:
                break
            if can_add(C, F_temp):
                valid_candidates.append(C)
                
        if len(valid_candidates) < 2:
            no_improve_streak += 1
            if no_improve_streak > 50:
                # Reset streak, maybe try a different A or restart?
                # For now, just continue
                no_improve_streak = 0
            continue
            
        # Try to find a pair B, C from valid_candidates
        # such that they don't form a sunflower with any x in F_temp
        found_pair = False
        # Shuffle to avoid bias
        random.shuffle(valid_candidates)
        
        # Limit the number of pairs to check to avoid timeout
        # Check first, say, 200 pairs
        checked = 0
        for i in range(len(valid_candidates)):
            if time.time() > deadline:
                break
            B = valid_candidates[i]
            for j in range(i + 1, len(valid_candidates)):
                if checked > 200:
                    break
                C = valid_candidates[j]
                checked += 1
                
                # Check if B, C form a sunflower with any x in F_temp
                ok = True
                for x in F_temp:
                    if is_sunflower_triple(B, C, x):
                        ok = False
                        break
                if ok:
                    # Success
                    new_family = F_temp + [B, C]
                    # Update current family
                    current_family = new_family
                    current_set = set(current_family)
                    improvements += 1
                    no_improve_streak = 0
                    found_pair = True
                    break
            if found_pair:
                break
                
        if not found_pair:
            no_improve_streak += 1
            # If we fail many times, maybe try a different strategy?
            # For now, just keep trying.
            
    # Convert indices back to subsets
    result = [sorted(all_subsets[i]) for i in current_family]
    
    return result
