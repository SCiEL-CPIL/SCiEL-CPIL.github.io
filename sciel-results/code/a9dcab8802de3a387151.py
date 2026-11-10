import random
import numpy as np
import time

N = 64

def construct():
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    coords = np.array(points)
    
    # Distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sum(diff**2, axis=2).astype(np.int32)
    
    def can_add(p_idx, S_indices, dist_from):
        # 1. p_idx is apex
        seen_dists = set()
        for a_idx in S_indices:
            d = D[p_idx, a_idx]
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
        # 2. Any a in S is apex
        for a_idx in S_indices:
            d = D[p_idx, a_idx]
            if d in dist_from[a_idx]:
                return False
                
        return True

    def add_point(p_idx, S_indices, dist_from):
        S_indices.append(p_idx)
        dist_from[p_idx] = set()
        for a_idx in S_indices[:-1]:
            d = D[p_idx, a_idx]
            dist_from[a_idx].add(d)
            dist_from[p_idx].add(d)

    def remove_point(r_idx, S_indices, dist_from):
        idx_pos = S_indices.index(r_idx)
        S_indices.pop(idx_pos)
        for a_idx in S_indices:
            d = D[r_idx, a_idx]
            dist_from[a_idx].discard(d)
        del dist_from[r_idx]

    def greedy(order_indices):
        S_indices = []
        dist_from = {}
        for p_idx in order_indices:
            if can_add(p_idx, S_indices, dist_from):
                add_point(p_idx, S_indices, dist_from)
        return S_indices, dist_from

    def hill_climb(S_indices, dist_from, max_steps=100):
        best_S = S_indices[:]
        best_dist = {k: v.copy() for k, v in dist_from.items()}
        best_size = len(best_S)
        
        for _ in range(max_steps):
            if not best_S:
                break
            
            r_idx = random.choice(best_S)
            
            exp_S = best_S[:]
            exp_dist = {k: v.copy() for k, v in best_dist.items()}
            
            remove_point(r_idx, exp_S, exp_dist)
            
            # Try to add points
            # We iterate through all points to find one that fits.
            # To be faster, we can pre-filter candidates?
            # For now, just iterate.
            # Optimization: If we find one, do we stop?
            # The original code added ALL that fit.
            # Adding all that fit is a "greedy fill".
            # This might jump to a different local optimum.
            
            for p_idx in range(num_points):
                if p_idx not in exp_S:
                    if can_add(p_idx, exp_S, exp_dist):
                        add_point(p_idx, exp_S, exp_dist)
                        
            if len(exp_S) > best_size:
                best_S = exp_S
                best_dist = exp_dist
                best_size = len(exp_S)
                
        return best_S, best_dist

    def perturb(S_indices, dist_from, k=2):
        exp_S = S_indices[:]
        exp_dist = {k: v.copy() for k, v in dist_from.items()}
        
        removed = []
        for _ in range(k):
            if not exp_S:
                break
            r_idx = random.choice(exp_S)
            remove_point(r_idx, exp_S, exp_dist)
            removed.append(r_idx)
            
        # Try to add back or new
        candidates = removed + [i for i in range(num_points) if i not in exp_S][:20]
        random.shuffle(candidates)
        
        for p_idx in candidates:
            if p_idx not in exp_S:
                if can_add(p_idx, exp_S, exp_dist):
                    add_point(p_idx, exp_S, exp_dist)
                    
        return exp_S, exp_dist

    best_S = []
    best_dist = {}
    best_size = 0
    start_time = time.time()
    seed_counter = 0
    
    while time.time() - start_time < 280:
        seed_counter += 1
        random.seed(seed_counter)
        
        order = list(range(num_points))
        random.shuffle(order)
        
        S, dist_from = greedy(order)
        S, dist_from = hill_climb(S, dist_from, max_steps=50)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
            best_dist = {k: v.copy() for k, v in dist_from.items()}
            
        # ILS Perturbations
        for _ in range(10):
            if time.time() - start_time > 280:
                break
            S_p, dist_p = perturb(best_S, best_dist, k=2)
            S_p, dist_p = hill_climb(S_p, dist_p, max_steps=50)
            
            if len(S_p) > best_size:
                best_size = len(S_p)
                best_S = S_p[:]
                best_dist = {k: v.copy() for k, v in dist_p.items()}

    return [points[idx] for idx in best_S]
