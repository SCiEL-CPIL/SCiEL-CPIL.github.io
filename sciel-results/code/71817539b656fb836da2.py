import time
import random
import numpy as np

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 285

    # Precompute all grid points
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    # Precompute squared distances between all pairs
    # dist[i, j] = squared Euclidean distance between points[i] and points[j]
    dx = points[:, 0].astype(np.int64)
    dy = points[:, 1].astype(np.int64)
    dx_diff = dx[:, None] - dx[None, :]
    dy_diff = dy[:, None] - dy[None, :]
    dist_sq = dx_diff * dx_diff + dy_diff * dy_diff  # shape (M, M)

    best_size = 0
    best_set = set()

    def is_isosceles_with_set(S, p_idx):
        """
        Check if adding p_idx to set S creates an isosceles triangle.
        S is a set of indices.
        """
        if not S:
            return False
        S_list = list(S)
        # Distances from p_idx to all points in S
        d_p = dist_sq[p_idx, S_list]  # shape (|S|,)
        
        # Case 1: p_idx is apex - two distances from p_idx equal
        if len(set(d_p.tolist())) != len(d_p):
            return True
        
        # Case 2: one of the existing points is apex
        # For each existing point s, check if dist(s, p_idx) equals dist(s, s') for some s' in S\{s}
        for s_idx in S_list:
            d_sp = dist_sq[s_idx, p_idx]
            # Distances from s_idx to all other points in S (excluding s_idx itself, but dist_sq[s,s]=0)
            d_s_others = dist_sq[s_idx, S_list]
            # Check if d_sp appears in d_s_others (it will appear at position of p_idx, but p_idx not in S yet)
            # Actually p_idx is NOT in S, so d_s_others contains distances from s_idx to all points in S
            # We need to check if any of those equals d_sp
            # But we must be careful: d_sp is dist from s_idx to p_idx, which is NOT in S
            # So we check if d_sp is in the set of distances from s_idx to S
            if d_sp in set(d_s_others.tolist()):
                return True
        return False

    def greedy_build(order, time_limit):
        """Build a set using greedy approach with given order."""
        S = set()
        for p_idx in order:
            if time.time() > time_limit:
                break
            if is_isosceles_with_set(S, p_idx):
                continue
            S.add(p_idx)
        return S

    # Phase 1: Multiple greedy restarts with different random orders
    t_phase1_end = start_time + 80
    num_restarts = 0
    
    while time.time() < t_phase1_end:
        order = list(range(M))
        random.shuffle(order)
        S = greedy_build(order, t_phase1_end)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
        num_restarts += 1

    # Phase 2: Local search - try to improve by swapping points
    S = set(best_set)
    
    def try_add(S, p_idx, time_limit):
        """Try to add p_idx to S. Returns True if successful."""
        if p_idx in S:
            return False
        if time.time() > time_limit:
            return False
        if is_isosceles_with_set(S, p_idx):
            return False
        S.add(p_idx)
        return True

    def remove_point(S, p_idx):
        """Remove p_idx from S."""
        S.discard(p_idx)

    def try_swap(S, remove_idx, add_idx, time_limit):
        """Try to swap: remove remove_idx, add add_idx. Returns True if successful."""
        if remove_idx not in S or add_idx in S:
            return False
        if time.time() > time_limit:
            return False
        
        # Temporarily remove
        S.discard(remove_idx)
        
        # Try to add
        if not is_isosceles_with_set(S, add_idx):
            S.add(add_idx)
            return True
        else:
            # Roll back
            S.add(remove_idx)
            return False

    # Phase 2a: Try adding points that weren't added in greedy
    t_phase2a_end = min(start_time + 150, deadline)
    
    while time.time() < t_phase2a_end:
        improved = False
        candidates = [i for i in range(M) if i not in S]
        random.shuffle(candidates)
        
        for p_idx in candidates:
            if time.time() > t_phase2a_end:
                break
            if try_add(S, p_idx, t_phase2a_end):
                improved = True
        
        if not improved:
            break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Phase 2b: Swap-based local search
    t_phase2b_end = deadline
    
    no_improve = 0
    max_no_improve = 15
    
    while time.time() < t_phase2b_end:
        improved = False
        S_list = list(S)
        random.shuffle(S_list)
        
        for remove_idx in S_list:
            if time.time() > t_phase2b_end:
                break
            
            # Generate candidate points to add
            r_x, r_y = points[remove_idx]
            window = 12
            x_min = max(0, r_x - window)
            x_max = min(N, r_x + window + 1)
            y_min = max(0, r_y - window)
            y_max = min(N, r_y + window + 1)
            
            # Find indices of points in the window
            nearby = []
            for i in range(M):
                if i in S or i == remove_idx:
                    continue
                px, py = points[i]
                if x_min <= px < x_max and y_min <= py < y_max:
                    nearby.append(i)
            
            random.shuffle(nearby)
            
            # Also try some random global candidates
            global_cands = [i for i in range(M) if i not in S and i != remove_idx]
            random.shuffle(global_cands)
            nearby.extend(global_cands[:30])
            
            for add_idx in nearby:
                if time.time() > t_phase2b_end:
                    break
                if add_idx in S:
                    continue
                if try_swap(S, remove_idx, add_idx, t_phase2b_end):
                    improved = True
                    break
            
            if improved:
                break
        
        if improved:
            no_improve = 0
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
        else:
            no_improve += 1
            
            # Perturbation: remove a few random points
            if no_improve > max_no_improve and len(S) > 5:
                k = random.randint(2, 4)
                to_remove = random.sample(list(S), min(k, len(S)))
                for r in to_remove:
                    remove_point(S, r)
                no_improve = 0
            
            if no_improve > max_no_improve * 2:
                # Full restart from current best
                S = set(best_set)
                no_improve = 0

    return [tuple(points[i]) for i in best_set]
