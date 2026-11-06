import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Initial Construction
    vectors = [
        0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 
        0b0110, 0b0111, 0b1000, 0b1001, 0b1010, 
        0b1011, 0b1100, 0b1101, 0b1110, 0b1111
    ]
    vec_to_idx = {v: i for i, v in enumerate(vectors)}
    
    triples = []
    n_pg = 15
    for i in range(n_pg):
        for j in range(i+1, n_pg):
            v_k = vectors[i] ^ vectors[j]
            k = vec_to_idx[v_k]
            t = tuple(sorted([i, j, k]))
            triples.append(t)
            
    random.seed(42)
    points_0_14 = list(range(15))
    random.shuffle(points_0_14)
    for i in range(0, 14, 2):
        a = points_0_14[i]
        b = points_0_14[i+1]
        triples.append(tuple(sorted([a, b, 15])))
        
    # 2. Data Structures
    def get_masks(triples):
        masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            masks.append(m)
        return masks
        
    def build_pair_map(triples):
        pair_map = {}
        for t in triples:
            a, b, c = t
            pair_map[(a, b)] = True
            pair_map[(a, c)] = True
            pair_map[(b, c)] = True
        return pair_map

    current_masks = get_masks(triples)
    pair_map = build_pair_map(triples)
    
    def compute_alpha_fast(triples, lower_bound=1):
        n_points = 16
        triple_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            triple_masks.append(m)
        
        for k in range(16, lower_bound - 1, -1):
            for combo in itertools.combinations(range(n_points), k):
                mask = 0
                for p in combo:
                    mask |= (1 << p)
                
                is_independent = True
                for tm in triple_masks:
                    if (mask & tm) == tm:
                        is_independent = False
                        break
                if is_independent:
                    return k
        return lower_bound

    current_alpha = compute_alpha_fast(triples)
    best_triples = triples[:]
    best_alpha = current_alpha
    
    # 3. Local Search
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    
    for iteration in range(100000):
        if time.time() > deadline - 5:
            break
            
        idx = random.randint(0, len(triples) - 1)
        removed = triples[idx]
        remaining_triples = triples[:idx] + triples[idx+1:]
        
        # Update pair_map for remaining
        # Since we remove one triple, we just need to check candidates against remaining
        # For speed, we can just check candidates against the full pair_map, 
        # but allow the removed triple's pairs if they are the only conflict?
        # No, simpler: rebuild pair_map for remaining? 
        # 42 triples, rebuilding is fast.
        remaining_pair_map = build_pair_map(remaining_triples)
        
        candidates = []
        for a in range(15):
            for b in range(a+1, 15):
                candidates.append((a, b, 15))
        for _ in range(20):
            a, b, c = random.sample(range(16), 3)
            candidates.append(tuple(sorted([a, b, c])))
            
        random.shuffle(candidates)
        
        found_move = False
        for cand in candidates:
            if time.time() > deadline - 5:
                break
            
            cand = tuple(sorted(cand))
            if cand in remaining_triples:
                continue
                
            a, b, c = cand
            # Check linearity using pair_map
            if remaining_pair_map.get((a, b), False) or remaining_pair_map.get((a, c), False) or remaining_pair_map.get((b, c), False):
                continue
                
            new_triples = remaining_triples + [cand]
            new_alpha = compute_alpha_fast(new_triples, lower_bound=1)
            
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = new_triples[:]
                triples = new_triples[:]
                current_masks = get_masks(triples)
                pair_map = build_pair_map(triples)
                found_move = True
                break
            elif new_alpha == best_alpha:
                if random.random() < 0.1:
                    triples = new_triples[:]
                    current_masks = get_masks(triples)
                    pair_map = build_pair_map(triples)
                    found_move = True
                    break
            
        if not found_move and random.random() < 0.05:
            # Try adding
            all_candidates = []
            for a in range(16):
                for b in range(a+1, 16):
                    for c in range(b+1, 16):
                        all_candidates.append((a, b, c))
            random.shuffle(all_candidates)
            for cand in all_candidates[:50]:
                cand = tuple(sorted(cand))
                if cand in triples:
                    continue
                a, b, c = cand
                if pair_map.get((a, b), False) or pair_map.get((a, c), False) or pair_map.get((b, c), False):
                    continue
                new_triples = triples + [cand]
                new_alpha = compute_alpha_fast(new_triples, lower_bound=1)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_triples = new_triples[:]
                    triples = new_triples[:]
                    current_masks = get_masks(triples)
                    pair_map = build_pair_map(triples)
                break
        
        T *= alpha
        if T < T_min:
            T = T_min

    return best_triples
