"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import heapq
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 285  # Safety margin
    
    # Grid points indexed as y*N + x
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    # Precompute squared distances
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx*dx + dy*dy).astype(np.int32)
    
    # For each point, sort other points by distance
    # point_by_dist[i] = list of (dist, j) sorted by dist
    point_by_dist = []
    for i in range(num_points):
        d = dist2[i].copy()
        d[i] = -1  # exclude self
        # Get indices sorted by distance
        order = np.argsort(d)
        lst = [(int(d[j]), int(j)) for j in order if d[j] >= 0]
        point_by_dist.append(lst)
    
    def greedy_with_priority():
        """Greedy construction using a priority queue based on conflict count."""
        S = []
        S_set = set()
        # For each point, track which distances are already used
        used_dists = [set() for _ in range(num_points)]
        
        # Priority: number of available (unused) distances from this point to unselected points
        # Initially all points have full availability
        # We'll use a heap of (-available_count, point_idx)
        
        # Count how many distances each point has to other points
        # This is just num_points - 1 for all, so we need dynamic updates
        
        # Alternative: use a simpler greedy with randomization
        rng = random.Random(42)
        
        # Try multiple random orders with early termination
        best_local = []
        
        for trial in range(20):
            if time.time() > deadline - 30:
                break
            
            # Create a random order with some structure
            order = list(range(num_points))
            if trial % 3 == 0:
                rng.shuffle(order)
            elif trial % 3 == 1:
                # Diagonal-ish
                order = sorted(range(num_points), key=lambda i: (points[i][0] + points[i][1], rng.random()))
            else:
                # Radial from random center
                cx = rng.randint(0, N-1)
                cy = rng.randint(0, N-1)
                order = sorted(range(num_points), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2 + rng.random()*0.1)
            
            S_local = []
            S_set_local = set()
            used_local = [set() for _ in range(num_points)]
            
            for idx_P in order:
                if idx_P in S_set_local:
                    continue
                
                # Check if P can be added: all distances from P to S must be distinct
                # AND for each Q in S, d(P,Q) not in used_local[Q]
                valid = True
                new_dists = set()
                
                for idx_Q in S_local:
                    d = int(dist2[idx_P, idx_Q])
                    if d in new_dists:
                        valid = False
                        break
                    new_dists.add(d)
                    if d in used_local[idx_Q]:
                        valid = False
                        break
                
                if valid:
                    S_local.append(idx_P)
                    S_set_local.add(idx_P)
                    used_local[idx_P] = new_dists
                    for idx_Q in S_local:
                        if idx_Q != idx_P:
                            used_local[idx_Q].add(int(dist2[idx_Q, idx_P]))
            
            if len(S_local) > len(best_local):
                best_local = S_local[:]
        
        return best_local
    
    def local_search_improve(S, used_dists, time_limit):
        """Improve via 1-swap and 2-swap moves."""
        S_set = set(S)
        complement = set(range(num_points)) - S_set
        comp_list = list(complement)
        rng = random.Random(123)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try 1-swap: remove Q, add P
            s_list = S[:]
            rng.shuffle(s_list)
            
            for idx_Q in s_list:
                if time.time() > time_limit:
                    break
                if idx_Q not in S_set:
                    continue
                
                # Find a P that can replace Q
                # P must satisfy: for all R in S\{Q}, d(P,R) not in used[R] and all d(P,R) distinct
                # Also d(P,R) != d(Q,R) is not required, but d(P,R) must not conflict with R's other distances
                
                S_without_Q = [i for i in S if i != idx_Q]
                
                # Sample candidates
                rng.shuffle(comp_list)
                candidates = comp_list[:200]
                
                found = False
                for idx_P in candidates:
                    if idx_P in S_set:
                        continue
                    
                    valid = True
                    new_dists_P = set()
                    
                    for idx_R in S_without_Q:
                        d_PR = int(dist2[idx_P, idx_R])
                        
                        if d_PR in new_dists_P:
                            valid = False
                            break
                        new_dists_P.add(d_PR)
                        
                        # R currently has distance d(R,Q) to Q. After swap, R will have d(R,P) instead.
                        # R's other distances (excluding d(R,Q)) must not include d(R,P)
                        if d_PR in used_dists[idx_R] and d_PR != int(dist2[idx_R, idx_Q]):
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        S.remove(idx_Q)
                        S_set.discard(idx_Q)
                        comp_list.remove(idx_P)
                        comp_list.append(idx_Q)
                        
                        S.append(idx_P)
                        S_set.add(idx_P)
                        
                        # Update used_dists
                        for idx_R in S:
                            if idx_R != idx_P:
                                d_old = int(dist2[idx_R, idx_Q])
                                d_new = int(dist2[idx_R, idx_P])
                                used_dists[idx_R].discard(d_old)
                                used_dists[idx_R].add(d_new)
                        
                        used_dists[idx_P] = set()
                        for idx_R in S:
                            if idx_R != idx_P:
                                used_dists[idx_P].add(int(dist2[idx_P, idx_R]))
                        
                        del used_dists[idx_Q]
                        
                        improved = True
                        found = True
                        break
                
                if found:
                    continue
            
            # Try 2-swap: remove Q1, Q2, add P1, P2
            if not improved and time.time() < time_limit - 5:
                s_list2 = S[:]
                rng.shuffle(s_list2)
                for i in range(min(len(s_list2), 50)):
                    for j in range(i+1, min(len(s_list2), 50)):
                        if time.time() > time_limit:
                            break
                        idx_Q1 = s_list2[i]
                        idx_Q2 = s_list2[j]
                        if idx_Q1 not in S_set or idx_Q2 not in S_set:
                            continue
                        
                        S_without = [x for x in S if x != idx_Q1 and x != idx_Q2]
                        
                        # Try to find two replacement points
                        rng.shuffle(comp_list)
                        candidates = comp_list[:50]
                        
                        # Try all pairs of candidates
                        for a in range(len(candidates)):
                            for b in range(a+1, len(candidates)):
                                idx_P1 = candidates[a]
                                idx_P2 = candidates[b]
                                if idx_P1 in S_set or idx_P2 in S_set:
                                    continue
                                
                                # Check validity
                                valid = True
                                new_dists = {idx_P1: set(), idx_P2: set()}
                                
                                # Check P1 and P2 among themselves
                                d_12 = int(dist2[idx_P1, idx_P2])
                                new_dists[idx_P1].add(d_12)
                                new_dists[idx_P2].add(d_12)
                                
                                for idx_R in S_without:
                                    d_1R = int(dist2[idx_P1, idx_R])
                                    d_2R = int(dist2[idx_P2, idx_R])
                                    
                                    if d_1R in new_dists[idx_P1]:
                                        valid = False
                                        break
                                    new_dists[idx_P1].add(d_1R)
                                    
                                    if d_2R in new_dists[idx_P2]:
                                        valid = False
                                        break
                                    new_dists[idx_P2].add(d_2R)
                                    
                                    # Check R's perspective
                                    old_dists_R = set()
                                    for idx_X in [idx_Q1, idx_Q2]:
                                        old_dists_R.add(int(dist2[idx_R, idx_X]))
                                    
                                    if d_1R in used_dists[idx_R] and d_1R not in old_dists_R:
                                        valid = False
                                        break
                                    if d_2R in used_dists[idx_R] and d_2R not in old_dists_R:
                                        valid = False
                                        break
                                
                                if valid:
                                    # Perform 2-swap
                                    for idx_Q in [idx_Q1, idx_Q2]:
                                        S.remove(idx_Q)
                                        S_set.discard(idx_Q)
                                        comp_list.append(idx_Q)
                                    for idx_P in [idx_P1, idx_P2]:
                                        comp_list.remove(idx_P)
                                        S.append(idx_P)
                                        S_set.add(idx_P)
                                    
                                    # Update used_dists
                                    for idx_R in S:
                                        for idx_Q in [idx_Q1, idx_Q2]:
                                            used_dists[idx_R].discard(int(dist2[idx_R, idx_Q]))
                                        for idx_P in [idx_P1, idx_P2]:
                                            used_dists[idx_R].add(int(dist2[idx_R, idx_P]))
                                    
                                    for idx_P in [idx_P1, idx_P2]:
                                        used_dists[idx_P] = set()
                                        for idx_R in S:
                                            if idx_R != idx_P:
                                                used_dists[idx_P].add(int(dist2[idx_P, idx_R]))
                                    
                                    for idx_Q in [idx_Q1, idx_Q2]:
                                        del used_dists[idx_Q]
                                    
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return S, used_dists
    
    # Phase 1: Greedy with multiple strategies
    best_S = greedy_with_priority()
    best_used = [set() for _ in range(num_points)]
    best_S_set = set(best_S)
    for idx_P in best_S:
        for idx_Q in best_S:
            if idx_P != idx_Q:
                best_used[idx_P].add(int(dist2[idx_P, idx_Q]))
    
    # Phase 2: Local search
    S, used = local_search_improve(best_S, best_used, time_limit=deadline - 20)
    
    # Phase 3: More restarts if time remains
    if time.time() < deadline - 60:
        for _ in range(5):
            if time.time() > deadline - 30:
                break
            S_new = greedy_with_priority()
            if len(S_new) > len(S):
                used_new = [set() for _ in range(num_points)]
                S_new_set = set(S_new)
                for idx_P in S_new:
                    for idx_Q in S_new:
                        if idx_P != idx_Q:
                            used_new[idx_P].add(int(dist2[idx_P, idx_Q]))
                S, used = local_search_improve(S_new, used_new, time_limit=deadline - 20)
    
    return [points[i] for i in S]
# EVOLVE-BLOCK-END
