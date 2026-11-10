import time
import random
import numpy as np

N = 64
NUM_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290.0

    # Precompute squared distances
    coords = np.array([(i // N, i % N) for i in range(NUM_POINTS)], dtype=np.int32)
    dist2 = ((coords[:, None, :] - coords[None, :, :])**2).sum(axis=2).astype(np.int32)
    
    # Helper functions
    def can_add(p, S, D):
        if p in S:
            return False
        if not S:
            return True
        
        # Check if distances from p to all q in S are distinct
        s_arr = np.array(list(S), dtype=np.int32)
        d_p_to_s = dist2[p, s_arr]
        if len(np.unique(d_p_to_s)) != len(s_arr):
            return False
        
        # Check if any distance d(p, q) is already in D[q]
        for q in s_arr:
            d_val = dist2[p, q]
            if d_val in D[q]:
                return False
        return True

    def add_point(p, S, D):
        S.add(p)
        D[p] = set()
        for q in S:
            if q == p:
                continue
            d_val = dist2[p, q]
            D[p].add(d_val)
            D[q].add(d_val)

    def remove_point(p, S, D):
        S.remove(p)
        for q in S:
            D[q].discard(dist2[p, q])
        del D[p]

    def can_swap(u, v, S, D):
        if u not in S or v in S:
            return False
        
        s_arr = np.array([q for q in S if q != u], dtype=np.int32)
        if len(s_arr) == 0:
            return True
        
        # Check if distances from v to s_arr are distinct
        d_v_to_s = dist2[v, s_arr]
        if len(np.unique(d_v_to_s)) != len(s_arr):
            return False
        
        # Check conflicts
        for q in s_arr:
            d_val = dist2[v, q]
            if d_val in D[q] and d_val != dist2[u, q]:
                return False
        return True

    def swap_points(u, v, S, D):
        remove_point(u, S, D)
        add_point(v, S, D)

    def can_swap2(u1, u2, v1, v2, S, D):
        if u1 not in S or u2 not in S or v1 in S or v2 in S:
            return False
        if u1 == u2 or v1 == v2:
            return False
        
        s_arr = np.array([q for q in S if q != u1 and q != u2], dtype=np.int32)
        
        if len(s_arr) > 0:
            d_v1_to_s = dist2[v1, s_arr]
            if len(np.unique(d_v1_to_s)) != len(s_arr):
                return False
            d_v2_to_s = dist2[v2, s_arr]
            if len(np.unique(d_v2_to_s)) != len(s_arr):
                return False
        
        d_v1_v2 = dist2[v1, v2]
        
        for q in s_arr:
            d_val = dist2[v1, q]
            if d_val in D[q] and d_val != dist2[u1, q] and d_val != dist2[u2, q]:
                return False
            d_val = dist2[v2, q]
            if d_val in D[q] and d_val != dist2[u1, q] and d_val != dist2[u2, q]:
                return False
        
        if len(s_arr) > 0:
            if d_v1_v2 in set(dist2[v1, s_arr]):
                return False
            if d_v1_v2 in set(dist2[v2, s_arr]):
                return False
        
        return True

    def swap_points2(u1, u2, v1, v2, S, D):
        remove_point(u1, S, D)
        remove_point(u2, S, D)
        add_point(v1, S, D)
        add_point(v2, S, D)

    best_set = set()
    best_D = {}
    rng = random.Random(42)

    # Main search loop
    while time.time() < deadline - 5:
        S = set()
        D = {}
        
        # Greedy Init
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        for p in order:
            if can_add(p, S, D):
                add_point(p, S, D)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try to add
            unselected = [p for p in range(NUM_POINTS) if p not in S]
            rng.shuffle(unselected)
            for p in unselected:
                if can_add(p, S, D):
                    add_point(p, S, D)
                    improved = True
                    break
            
            if improved:
                continue

            # 2. Try 1-swaps
            s_list = list(S)
            rng.shuffle(s_list)
            swap_found = False
            for u in s_list:
                for v in unselected:
                    if can_swap(u, v, S, D):
                        swap_points(u, v, S, D)
                        improved = True
                        swap_found = True
                        break
                if swap_found:
                    break
            
            if improved:
                continue

            # 3. Try 2-swaps (Random)
            if len(S) >= 2:
                u1, u2 = random.sample(list(S), 2)
                if len(unselected) >= 2:
                    v1, v2 = random.sample(unselected, 2)
                    if can_swap2(u1, u2, v1, v2, S, D):
                        swap_points2(u1, u2, v1, v2, S, D)
                        improved = True
            
        if len(S) > len(best_set):
            best_set = S.copy()
            best_D = {k: v.copy() for k, v in D.items()}
            
    return [(i // N, i % N) for i in best_set]
