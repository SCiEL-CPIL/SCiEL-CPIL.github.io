import numpy as np
import time
from itertools import combinations, product

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_vectors = []
    best_score = 0
    
    # Strategy 1: All norm-sqrt(2) vectors
    vectors1 = []
    for i, j in combinations(range(D), 2):
        for si, sj in product((1, -1), repeat=2):
            v = [0] * D
            v[i], v[j] = si, sj
            vectors1.append(v)
    vectors1 = np.array(vectors1, dtype=np.int64)
    best_score = len(vectors1)
    best_vectors = vectors1
    
    # Strategy 2: Greedy on norm-sqrt(3) vectors using bitmask representation
    # Represent each vector as (pos_mask, neg_mask)
    candidates = []
    for i, j, k in combinations(range(D), 3):
        for si, sj, sk in product((1, -1), repeat=3):
            pos_mask = 0
            neg_mask = 0
            for coord, val in zip((i, j, k), (si, sj, sk)):
                if val == 1:
                    pos_mask |= (1 << coord)
                else:
                    neg_mask |= (1 << coord)
            candidates.append((pos_mask, neg_mask))
    
    # Greedy selection
    selected_masks = []
    
    for pos_m, neg_m in candidates:
        if time.time() > deadline:
            break
        
        compatible = True
        for pos_s, neg_s in selected_masks:
            # Compute dist_sq
            # v - w: coordinate i contributes:
            # if v_i=1, w_i=1: 0
            # if v_i=1, w_i=-1: 4
            # if v_i=1, w_i=0: 1
            # if v_i=-1, w_i=1: 4
            # if v_i=-1, w_i=-1: 0
            # if v_i=-1, w_i=0: 1
            # if v_i=0, w_i=1: 1
            # if v_i=0, w_i=-1: 1
            # if v_i=0, w_i=0: 0
            
            # v_i=1: pos_m & ~(pos_s | neg_s) -> w_i=0, count * 1
            # v_i=1: pos_m & neg_s -> w_i=-1, count * 4
            # v_i=-1: neg_m & ~(pos_s | neg_s) -> w_i=0, count * 1
            # v_i=-1: neg_m & pos_s -> w_i=1, count * 4
            # v_i=0: ~(pos_m | neg_m) & pos_s -> w_i=1, count * 1
            # v_i=0: ~(pos_m | neg_m) & neg_s -> w_i=-1, count * 1
            
            v1_w0 = (pos_m & ~(pos_s | neg_s)).bit_count()
            v1_wm1 = (pos_m & neg_s).bit_count()
            vm1_w0 = (neg_m & ~(pos_s | neg_s)).bit_count()
            vm1_w1 = (neg_m & pos_s).bit_count()
            v0_w1 = (~(pos_m | neg_m) & pos_s & ((1 << D) - 1)).bit_count()
            v0_wm1 = (~(pos_m | neg_m) & neg_s & ((1 << D) - 1)).bit_count()
            
            dist_sq = v1_w0 + 4*v1_wm1 + vm1_w0 + 4*vm1_w1 + v0_w1 + v0_wm1
            
            if dist_sq < 3:  # need dist >= sqrt(3), so dist_sq >= 3
                compatible = False
                break
        
        if compatible:
            selected_masks.append((pos_m, neg_m))
    
    if len(selected_masks) > best_score:
        # Convert back to vectors
        selected_vectors = []
        for pos_m, neg_m in selected_masks:
            v = [0] * D
            for i in range(D):
                if (pos_m >> i) & 1:
                    v[i] = 1
                elif (neg_m >> i) & 1:
                    v[i] = -1
            selected_vectors.append(v)
        best_vectors = np.array(selected_vectors, dtype=np.int64)
        best_score = len(selected_vectors)
    
    return best_vectors
