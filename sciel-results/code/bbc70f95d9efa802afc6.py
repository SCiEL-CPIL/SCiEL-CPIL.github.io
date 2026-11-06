import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    n = 16
    points = list(range(n))
    all_triples = list(combinations(points, 3))
    triple_to_mask = {t: (1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in all_triples}
    
    # Precompute conflicts
    # conflict[t] = list of triples that conflict with t
    conflict = {t: [] for t in all_triples}
    for i, t1 in enumerate(all_triples):
        for t2 in all_triples[i+1:]:
            inter = len(set(t1) & set(t2))
            if inter >= 2:
                conflict[t1].append(t2)
                conflict[t2].append(t1)
    
    # Precompute subsets by size for alpha calculation
    subsets_by_size = [[] for _ in range(n+1)]
    for mask in range(1 << n):
        k = bin(mask).count('1')
        subsets_by_size[k].append(mask)
        
    # Function to compute alpha
    def get_alpha(triple_masks):
        if not triple_masks:
            return n
        # Check from largest size down
        for k in range(n, 0, -1):
            for S in subsets_by_size[k]:
                independent = True
                for m in triple_masks:
                    if (S & m) == m:
                        independent = False
                        break
                if independent:
                    return k
        return 0 # Should not happen for k>=1
        
    # Greedy initialization
    def greedy_init():
        triples = []
        used = set()
        # Shuffle to randomize
        shuffled_triples = all_triples[:]
        random.shuffle(shuffled_triples)
        for t in shuffled_triples:
            if t in used:
                continue
            # Check if t conflicts with any existing triple
            ok = True
            for u in triples:
                if t in conflict[u]:
                    ok = False
                    break
            if ok:
                triples.append(t)
                used.add(t)
        return triples

    best_alpha = n
    best_triples = []
    
    # Multiple restarts
    while time.time() < deadline:
        # Init
        current_triples = greedy_init()
        current_masks = [triple_to_mask[t] for t in current_triples]
        current_alpha = get_alpha(current_masks)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = current_triples[:]
            if best_alpha <= 5: # Good enough?
                break
                
        # Local Search
        t_start = time.time()
        while time.time() < deadline and time.time() - t_start < 5.0: # 5s per restart
            # Generate move
            move_type = random.random()
            new_triples = current_triples[:]
            new_masks = current_masks[:]
            
            if move_type < 0.6: # Add
                # Pick a random triple not in current
                # To be efficient, pick from all_triples
                t_add = random.choice(all_triples)
                if t_add in new_triples:
                    continue
                # Check conflict
                ok = True
                for u in new_triples:
                    if t_add in conflict[u]:
                        ok = False
                        break
                if not ok:
                    continue
                new_triples.append(t_add)
                new_masks.append(triple_to_mask[t_add])
                
            elif move_type < 0.8: # Remove
                if not new_triples:
                    continue
                idx = random.randint(0, len(new_triples) - 1)
                new_masks.pop(idx)
                new_triples.pop(idx)
                
            else: # Swap
                if not new_triples:
                    continue
                idx_remove = random.randint(0, len(new_triples) - 1)
                t_remove = new_triples[idx_remove]
                new_masks.pop(idx_remove)
                new_triples.pop(idx_remove)
                
                t_add = random.choice(all_triples)
                if t_add in new_triples:
                    continue
                ok = True
                for u in new_triples:
                    if t_add in conflict[u]:
                        ok = False
                        break
                if not ok:
                    continue
                new_triples.append(t_add)
                new_masks.append(triple_to_mask[t_add])
            
            new_alpha = get_alpha(new_masks)
            
            # Acceptance
            if new_alpha < current_alpha:
                current_triples = new_triples
                current_masks = new_masks
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples[:]
                    if best_alpha <= 5:
                        break
            elif new_alpha == current_alpha:
                # Accept to explore
                if random.random() < 0.5:
                    current_triples = new_triples
                    current_masks = new_masks
            else:
                # Reject (or accept with very low prob, but keep it simple)
                pass
                
        if best_alpha <= 5:
            break

    # Convert best_triples to list of lists
    return [list(t) for t in best_triples]
