import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    
    # Precompute masks and pairs for speed
    triple_info = {}
    for t in all_triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        pairs = frozenset([(t[0], t[1]), (t[0], t[2]), (t[1], t[2])])
        triple_info[t] = (mask, pairs)

    def get_pairs(t):
        return triple_info[t][1]
        
    def get_mask(t):
        return triple_info[t][0]

    def calc_alpha_and_set(triples):
        if not triples:
            return 16, set(points)
        
        masks = [get_mask(t) for t in triples]
        
        # Check from 16 down to 1
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                    
                is_ind = True
                for m in masks:
                    if (m & mask) == m:
                        is_ind = False
                        break
                if is_ind:
                    return k, set(subset)
        return 0, set()

    def is_compatible(triple, used_pairs):
        for p in get_pairs(triple):
            if p in used_pairs:
                return False
        return True

    def build_greedy_system():
        indices = list(range(len(all_triples)))
        random.shuffle(indices)
        current = []
        used_pairs = set()
        for idx in indices:
            t = all_triples[idx]
            if is_compatible(t, used_pairs):
                current.append(t)
                for p in get_pairs(t):
                    used_pairs.add(p)
        return current

    best_alpha = 16
    best_triples = []
    
    # Run multiple local searches
    while time.time() < deadline:
        # 1. Initialize
        current_triples = build_greedy_system()
        if not current_triples:
            continue
            
        current_alpha, _ = calc_alpha_and_set(current_triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = [list(t) for t in current_triples]
            
        # 2. Local Search
        used_pairs = set()
        for t in current_triples:
            for p in get_pairs(t):
                used_pairs.add(p)
                
        no_improve_counter = 0
        
        while time.time() < deadline - 5: # Keep 5s for final check
            current_alpha, S = calc_alpha_and_set(current_triples)
            
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = [list(t) for t in current_triples]
                no_improve_counter = 0
            else:
                no_improve_counter += 1
                
            if S is None or len(S) < 3:
                break
                
            # Generate candidates in S using masks
            mask_S = 0
            for p in S:
                mask_S |= (1 << p)
            candidates = [t for t in all_triples if (get_mask(t) & ~mask_S) == 0]
            
            if not candidates:
                break
                
            random.shuffle(candidates)
            
            # Try a few candidates per iteration? No, one move per iteration.
            t_new = candidates[0]
            
            if t_new in current_triples:
                continue
                
            # Find conflicts
            pairs_new = get_pairs(t_new)
            conflicts = []
            for t_old in current_triples:
                if get_pairs(t_old) & pairs_new:
                    conflicts.append(t_old)
                    
            # If no conflicts, just add
            # If conflicts, remove them and add
            # This is always a valid move.
            
            new_triples = [t for t in current_triples if t not in conflicts]
            new_triples.append(t_new)
            
            # Quick check: did we improve?
            # We can estimate alpha change.
            # But let's calculate exact alpha.
            new_alpha, _ = calc_alpha_and_set(new_triples)
            
            # Acceptance
            if new_alpha < current_alpha:
                current_triples = new_triples
                # Rebuild used_pairs
                used_pairs = set()
                for t in current_triples:
                    for p in get_pairs(t):
                        used_pairs.add(p)
                no_improve_counter = 0
            elif new_alpha == current_alpha:
                # Accept if size increases or randomly
                if len(new_triples) > len(current_triples) or random.random() < 0.5:
                    current_triples = new_triples
                    used_pairs = set()
                    for t in current_triples:
                        for p in get_pairs(t):
                            used_pairs.add(p)
                    no_improve_counter = 0
            else:
                # Reject
                pass
                
            if no_improve_counter > 200:
                # Restart this local search
                break
                
    return best_triples
