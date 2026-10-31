D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def valid_solution(vecs):
        if len(vecs) == 0:
            return True
        if len(vecs) < 2:
            return all(np.dot(v, v) > 0 for v in vecs)
        norms_sq = np.array([np.dot(v, v) for v in vecs])
        max_norm = np.max(norms_sq)
        vecs_arr = np.array(vecs)
        dots = vecs_arr @ vecs_arr.T
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        np.fill_diagonal(dist_sq, 1e18)
        return np.min(dist_sq) >= max_norm - 1e-9

    def gen_weight_vectors(weight):
        vecs = []
        for comb in combinations(range(D), weight):
            for signs in product((1, -1), repeat=weight):
                v = [0]*D
                for idx, s in zip(comb, signs):
                    v[idx] = s
                vecs.append(v)
        return np.array(vecs, dtype=np.int64)

    # Generate seed (weight 2) vectors
    seed_vecs = gen_weight_vectors(2)
    
    best_vecs = seed_vecs
    best_score = len(seed_vecs)
    
    # Try weight 3
    if elapsed() < deadline - 20:
        wt3_vecs = gen_weight_vectors(3)
        n3 = len(wt3_vecs)
        dots3 = wt3_vecs @ wt3_vecs.T
        # Conflict if dot >= 2 (since dist^2 = 6 - 2*dot, need dist^2 >= 3 => 6-2d >= 3 => 2d <= 3 => d <= 1.5 => d <= 1)
        # So conflict if dot >= 2.
        degrees3 = np.sum(dots3 >= 2, axis=1) - 1
        order3 = np.argsort(degrees3)
        
        selected_idx3 = []
        removed3 = np.zeros(n3, dtype=bool)
        
        for i in order3:
            if removed3[i]:
                continue
            selected_idx3.append(i)
            neighbors = np.where(dots3[i] >= 2)[0]
            removed3[neighbors] = True
            if elapsed() > deadline - 10:
                break

        wt3_selected = wt3_vecs[selected_idx3]
        
        if valid_solution(wt3_selected) and len(wt3_selected) > best_score:
            best_vecs = wt3_selected
            best_score = len(wt3_selected)

    # Try weight 4
    if elapsed() < deadline - 10:
        wt4_vecs = gen_weight_vectors(4)
        n4 = len(wt4_vecs)
        dots4 = wt4_vecs @ wt4_vecs.T
        # Conflict if dot >= 3 (since dist^2 = 8 - 2*dot, need dist^2 >= 4 => 8-2d >= 4 => 2d <= 4 => d <= 2)
        # So conflict if dot >= 3.
        degrees4 = np.sum(dots4 >= 3, axis=1) - 1
        order4 = np.argsort(degrees4)
        
        selected_idx4 = []
        removed4 = np.zeros(n4, dtype=bool)
        
        for i in order4:
            if removed4[i]:
                continue
            selected_idx4.append(i)
            neighbors = np.where(dots4[i] >= 3)[0]
            removed4[neighbors] = True
            if elapsed() > deadline - 5:
                break

        wt4_selected = wt4_vecs[selected_idx4]
        
        if valid_solution(wt4_selected) and len(wt4_selected) > best_score:
            best_vecs = wt4_selected
            best_score = len(wt4_selected)

    return best_vecs

# EVOLVE-BLOCK-END
