import random
import time
import numpy as np

N = 64

def construct():
    start_time = time.time()
    time_limit = 290  # Safety margin before 330s kill
    
    # Precompute all points
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    # Precompute squared distances between all pairs
    # dist_matrix[i][j] = squared distance between points[i] and points[j]
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    # Compute all pairwise squared distances
    # Shape: (num_points, num_points)
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    dist_matrix = (dx * dx + dy * dy).astype(np.int64)
    
    # For each point, we need to track which squared distances are "used" by other points in the set
    # A point P can be added to set S if:
    # 1. For all Q in S, dist(P,Q) is distinct (no two Q give same distance)
    # 2. For all Q in S, dist(P,Q) is not already in dist_sets[Q]
    
    # Strategy: greedy construction with multiple random orderings, then local search
    
    best_size = 0
    best_set = []
    
    def try_add_greedy(P_idx, S_indices, used_dists):
        """
        Try to add point P_idx to the set S (represented by indices).
        used_dists: list of sets, used_dists[i] = set of squared distances from S[i] to other points in S
        Returns True if P_idx can be added.
        """
        S_list = list(S_indices)
        if not S_list:
            return True
        
        # Get distances from P to all points in S
        dists = dist_matrix[P_idx, S_list]
        
        # Check 1: all distances must be distinct
        if len(set(dists.tolist())) != len(dists):
            return False
        
        # Check 2: for each Q in S, dist(P,Q) must not be in used_dists[S_list.index(Q)]
        for i, q_idx in enumerate(S_list):
            d2 = dists[i]
            if d2 in used_dists[q_idx]:
                return False
        
        return True
    
    def greedy_construct(order):
        """
        Greedy construction given a random ordering of points.
        Returns (S_indices, used_dists).
        """
        S_indices = []
        used_dists = {}  # index -> set of squared distances
        
        for P_idx in order:
            if try_add_greedy(P_idx, S_indices, used_dists):
                S_indices.append(P_idx)
                used_dists[P_idx] = set()
                # Update used_dists for existing points
                for Q_idx in S_indices[:-1]:
                    d2 = dist_matrix[P_idx, Q_idx]
                    used_dists[Q_idx].add(d2)
                    used_dists[P_idx].add(d2)
        
        return S_indices, used_dists
    
    def local_search(S_indices, used_dists, time_budget):
        """
        Local search: try to add more points or swap points to increase set size.
        Returns (S_indices, used_dists, improved_size).
        """
        current_size = len(S_indices)
        S_set = set(S_indices)
        
        # Build a list of candidate points not in S
        all_indices = list(range(num_points))
        candidates = [i for i in all_indices if i not in S_set]
        
        # Try adding points
        added_any = True
        while added_any and time.time() - start_time < time_budget:
            added_any = False
            random.shuffle(candidates)
            for P_idx in candidates:
                if P_idx in S_set:
                    continue
                if try_add_greedy(P_idx, S_indices, used_dists):
                    S_indices.append(P_idx)
                    S_set.add(P_idx)
                    used_dists[P_idx] = set()
                    for Q_idx in S_indices[:-1]:
                        d2 = dist_matrix[P_idx, Q_idx]
                        used_dists[Q_idx].add(d2)
                        used_dists[P_idx].add(d2)
                    candidates.remove(P_idx)
                    current_size += 1
                    added_any = True
                    break
        
        # Try swap-based local search
        for _ in range(2000):
            if time.time() - start_time > time_budget:
                break
            
            # Randomly pick a point in S and a point not in S
            if not candidates:
                break
            
            i = random.randint(0, len(S_indices) - 1)
            Q_idx = S_indices[i]
            
            # Try removing Q_idx and adding a random candidate
            removed_dist_set = used_dists[Q_idx].copy()
            
            # Remove Q_idx from S
            S_indices.pop(i)
            S_set.discard(Q_idx)
            del used_dists[Q_idx]
            
            # Remove Q_idx's distances from other points' used_dists
            for j, R_idx in enumerate(S_indices):
                d2 = dist_matrix[Q_idx, R_idx]
                used_dists[R_idx].discard(d2)
            
            # Try to add a random candidate
            added = False
            for _ in range(min(20, len(candidates))):
                P_idx = random.choice(candidates)
                if P_idx in S_set:
                    continue
                if try_add_greedy(P_idx, S_indices, used_dists):
                    S_indices.append(P_idx)
                    S_set.add(P_idx)
                    used_dists[P_idx] = set()
                    for R_idx in S_indices[:-1]:
                        d2 = dist_matrix[P_idx, R_idx]
                        used_dists[R_idx].add(d2)
                        used_dists[P_idx].add(d2)
                    candidates.remove(P_idx)
                    added = True
                    break
            
            if not added:
                # Restore Q_idx
                S_indices.append(Q_idx)
                S_set.add(Q_idx)
                used_dists[Q_idx] = removed_dist_set
                for R_idx in S_indices[:-1]:
                    d2 = dist_matrix[Q_idx, R_idx]
                    used_dists[R_idx].add(d2)
                    used_dists[Q_idx].add(d2)
        
        return S_indices, used_dists, len(S_indices)
    
    # Main search loop
    for seed in range(500):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(seed)
        order = list(range(num_points))
        random.shuffle(order)
        
        S_indices, used_dists = greedy_construct(order)
        
        if len(S_indices) > best_size:
            best_size = len(S_indices)
            best_set = [points[i] for i in S_indices]
        
        # Do local search if we have time
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5:
            S_indices, used_dists, new_size = local_search(S_indices, used_dists, time_limit - 2)
            if new_size > best_size:
                best_size = new_size
                best_set = [points[i] for i in S_indices]
    
    return best_set
