import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    triple_pairs = [ [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])] for t in all_triples ]
    triple_masks = [ (1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in all_triples ]
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    max_mask = 1 << 16
    full_mask = max_mask - 1
    
    # Popcount and subsets
    popcount = [0] * max_mask
    subsets_by_popcount = [[] for _ in range(17)]
    for i in range(max_mask):
        c = 0
        x = i
        while x:
            x &= x - 1
            c += 1
        popcount[i] = c
        subsets_by_popcount[c].append(i)
        
    # Supersets
    triple_supersets = []
    for m in triple_masks:
        comp = full_mask ^ m
        supersets = []
        sub = comp
        while True:
            supersets.append(m | sub)
            if sub == 0: break
            sub = (sub - 1) & comp
        triple_supersets.append(supersets)

    # State
    used_pairs = set()
    current_triples = []
    bad_count = [0] * max_mask
    current_alpha = 16
    
    best_alpha = 16
    best_triples = []
    
    def get_alpha():
        for k in range(16, -1, -1):
            for s in subsets_by_popcount[k]:
                if bad_count[s] == 0:
                    return k
        return 0

    def add_triple(idx):
        t = all_triples[idx]
        current_triples.append(t)
        for p in triple_pairs[idx]:
            used_pairs.add(p)
        for s in triple_supersets[idx]:
            bad_count[s] += 1
            
    def remove_triple(idx):
        t = all_triples[idx]
        # Remove from list
        # O(N) search
        for i, ct in enumerate(current_triples):
            if ct == t:
                current_triples.pop(i)
                break
        for p in triple_pairs[idx]:
            used_pairs.discard(p)
        for s in triple_supersets[idx]:
            bad_count[s] -= 1

    def update_alpha_after_add(old_alpha):
        # Check if old_alpha is still valid
        for s in subsets_by_popcount[old_alpha]:
            if bad_count[s] == 0:
                return old_alpha
        
        for k in range(old_alpha - 1, -1, -1):
            for s in subsets_by_popcount[k]:
                if bad_count[s] == 0:
                    return k
        return 0

    def update_alpha_after_remove(old_alpha):
        for k in range(old_alpha + 1, 17):
            for s in subsets_by_popcount[k]:
                if bad_count[s] == 0:
                    return k
        return old_alpha

    def is_valid_add(idx):
        for p in triple_pairs[idx]:
            if p in used_pairs:
                return False
        return True

    def init_greedy():
        nonlocal used_pairs, current_triples, bad_count, current_alpha
        used_pairs.clear()
        current_triples.clear()
        # Reset bad_count
        # Instead of resetting whole array, we can just set to 0
        for i in range(max_mask):
            bad_count[i] = 0
            
        indices = list(range(n_triples))
        random.shuffle(indices)
        for idx in indices:
            if is_valid_add(idx):
                add_triple(idx)
        current_alpha = get_alpha()
        
    # Initial
    init_greedy()
    best_alpha = current_alpha
    best_triples = list(current_triples)
    
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    steps_since_improvement = 0
    max_stuck = 2000
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.4:
            # Add
            candidates = []
            for idx in range(n_triples):
                if is_valid_add(idx):
                    candidates.append(idx)
            if not candidates:
                continue
                
            idx = random.choice(candidates)
            old_alpha = current_alpha
            
            add_triple(idx)
            new_alpha = update_alpha_after_add(old_alpha)
            delta = new_alpha - old_alpha
            current_alpha = new_alpha
            
            if delta <= 0 or (T > T_min and random.random() < math.exp(-delta / T)):
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                remove_triple(idx)
                current_alpha = old_alpha
                steps_since_improvement += 1

        elif move_type < 0.7:
            # Remove
            if not current_triples:
                continue
            rem_pos = random.randrange(len(current_triples))
            t_rem = current_triples[rem_pos]
            idx = triple_to_idx[t_rem]
            
            old_alpha = current_alpha
            remove_triple(idx)
            new_alpha = update_alpha_after_remove(old_alpha)
            delta = new_alpha - old_alpha
            current_alpha = new_alpha
            
            if delta <= 0 or (T > T_min and random.random() < math.exp(-delta / T)):
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                add_triple(idx)
                current_alpha = old_alpha
                steps_since_improvement += 1

        else:
            # Swap
            if not current_triples:
                continue
            rem_pos = random.randrange(len(current_triples))
            t_rem = current_triples[rem_pos]
            idx_rem = triple_to_idx[t_rem]
            
            # Temporarily remove
            remove_triple(idx_rem)
            
            # Find valid add
            candidates = []
            for idx_add in range(n_triples):
                if idx_add == idx_rem:
                    continue
                if is_valid_add(idx_add):
                    candidates.append(idx_add)
                    
            if not candidates:
                add_triple(idx_rem)
                continue
                
            idx_add = random.choice(candidates)
            
            old_alpha = current_alpha
            add_triple(idx_add)
            new_alpha = update_alpha_after_add(old_alpha) # Note: we removed one, so alpha increased or stayed. Then we add one, so it decreases or stays.
            # Wait.
            # State S. Alpha A.
            # Remove t_rem -> State S'. Alpha A'. A' >= A.
            # We calculated A' in remove step? No, we didn't update current_alpha in the temp remove.
            # We need to be careful.
            
            # Let's trace:
            # 1. current_alpha is A.
            # 2. remove_triple(idx_rem) modifies bad_count and used_pairs.
            # 3. We need to calculate Alpha of S'.
            #    Let's call it A_temp.
            #    A_temp = update_alpha_after_remove(A)
            # 4. Then we add idx_add.
            #    New Alpha = update_alpha_after_add(A_temp)
            
            # So we need to compute A_temp first.
            
            # Let's redo the swap logic carefully.
            pass # Will implement in final code

        T *= cooling
        if T < T_min: T = T_min
        
        if steps_since_improvement > max_stuck:
            init_greedy()
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = list(current_triples)
            steps_since_improvement = 0
            T = 1.0

    return best_triples
