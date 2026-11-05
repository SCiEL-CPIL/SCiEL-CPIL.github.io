"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 280  # Safety margin
    
    # Grid points
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    # Create coordinate arrays
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    # Calculate squared distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx*dx + dy*dy).astype(np.int32)
    
    def greedy_construct(order, time_limit):
        """Greedy construction following a given order."""
        S = []
        S_set = set()
        used_dists = {}  # point_idx -> set of distances from that point to other points in S
        
        for idx_P in order:
            if time.time() > time_limit:
                break
            if idx_P in S_set:
                continue
            
            # Check if P can be added
            new_dists = set()
            valid = True
            
            for idx_Q in S:
                d = int(dist2[idx_P, idx_Q])
                if d in new_dists:
                    valid = False
                    break
                new_dists.add(d)
                if d in used_dists[idx_Q]:
                    valid = False
                    break
            
            if valid:
                S.append(idx_P)
                S_set.add(idx_P)
                used_dists[idx_P] = new_dists
                for idx_Q in S:
                    if idx_Q != idx_P:
                        used_dists[idx_Q].add(int(dist2[idx_Q, idx_P]))
        
        return S, used_dists
    
    def is_valid_swap(idx_P, idx_Q, S_indices, D_dict):
        """
        Check if swapping idx_Q for idx_P keeps the set valid.
        S_indices contains idx_Q. We remove Q and add P.
        """
        if idx_P in D_dict:
            return False
        
        dists_from_P = set()
        for idx_R in S_indices:
            if idx_R == idx_Q:
                continue
            
            d_PR = int(dist2[idx_P, idx_R])
            
            # Check P's own distances are unique
            if d_PR in dists_from_P:
                return False
            dists_from_P.add(d_PR)
            
            # Check R's distances after swap
            # R currently has d_RQ in D_dict[R]
            # After swap, R will have d_PR instead of d_RQ
            # Need: d_PR not in (D_dict[R] - {d_RQ})
            d_RQ = int(dist2[idx_R, idx_Q])
            
            if d_PR in D_dict[idx_R]:
                if d_PR != d_RQ:
                    return False
        
        return True
    
    def perform_swap(idx_P, idx_Q, S_indices, D_dict, S_set, complement):
        """Perform the swap of Q for P."""
        # Remove Q
        S_indices.remove(idx_Q)
        S_set.discard(idx_Q)
        complement.append(idx_Q)
        
        for idx_R in S_indices:
            d = int(dist2[idx_Q, idx_R])
            D_dict[idx_R].discard(d)
        del D_dict[idx_Q]
        
        # Add P
        S_indices.append(idx_P)
        S_set.add(idx_P)
        complement.remove(idx_P)
        
        D_dict[idx_P] = set()
        for idx_R in S_indices:
            if idx_R != idx_P:
                d = int(dist2[idx_P, idx_R])
                D_dict[idx_P].add(d)
                D_dict[idx_R].add(d)
    
    def local_search(S_indices, D_dict, time_limit, rng, max_passes=100):
        """Improve S via swaps with time budget."""
        S_set = set(S_indices)
        complement = list(set(range(num_points)) - S_set)
        
        improved = True
        passes = 0
        
        while improved and time.time() < time_limit and passes < max_passes:
            improved = False
            passes += 1
            
            s_list = S_indices[:]
            rng.shuffle(s_list)
            
            for idx_Q in s_list:
                if time.time() > time_limit:
                    break
                if idx_Q not in S_set:
                    continue
                
                # Try to find a valid P in complement to replace Q
                comp_list = complement[:]
                rng.shuffle(comp_list)
                
                for idx_P in comp_list:
                    if idx_P in S_set:
                        continue
                    
                    if is_valid_swap(idx_P, idx_Q, S_indices, D_dict):
                        perform_swap(idx_P, idx_Q, S_indices, D_dict, S_set, complement)
                        improved = True
                        break
        
        return S_indices, D_dict, S_set, complement
    
    # Phase 1: Multiple greedy initializations with different orderings
    rng = random.Random(42)
    
    best_S = []
    best_used = {}
    
    # Generate diverse orderings
    orderings = []
    
    # Random shuffles
    for _ in range(8):
        order = list(range(num_points))
        rng.shuffle(order)
        orderings.append(order)
    
    # Diagonal scan
    order_diag = sorted(range(num_points), key=lambda i: (points[i][0] + points[i][1], points[i][0]))
    orderings.append(order_diag)
    
    # Anti-diagonal
    order_anti = sorted(range(num_points), key=lambda i: (points[i][0] - points[i][1], points[i][0]))
    orderings.append(order_anti)
    
    # Radial from center
    cx, cy = N/2.0, N/2.0
    order_radial = sorted(range(num_points), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2)
    orderings.append(order_radial)
    
    # Row-major
    order_rows = list(range(num_points))
    orderings.append(order_rows)
    
    # Stride patterns
    for stride in [3, 5, 7, 11, 13]:
        order_stride = list(range(0, num_points, stride))
        orderings.append(order_stride)
    
    # Run greedy on each ordering
    greedy_time_limit = deadline - 60  # Leave time for local search
    for order in orderings:
        S, used = greedy_construct(order, time_limit=greedy_time_limit)
        if len(S) > len(best_S):
            best_S = S[:]
            best_used = {k: v.copy() for k, v in used.items()}
    
    # Phase 2: Local search on best solution
    S = best_S[:]
    D_dict = {k: v.copy() for k, v in best_used.items()}
    S_set = set(S)
    complement = list(set(range(num_points)) - S_set)
    
    # Do multiple local search passes with different random seeds
    ls_time_limit = deadline - 20
    for seed_offset in range(5):
        rng_ls = random.Random(42 + seed_offset * 1000)
        S, D_dict, S_set, complement = local_search(S, D_dict, ls_time_limit, rng_ls, max_passes=50)
        if time.time() > ls_time_limit:
            break
    
    # Phase 3: Try a few more restarts if time remains
    if time.time() < deadline - 50:
        for _ in range(3):
            order = list(range(num_points))
            rng.shuffle(order)
            S_new, used_new = greedy_construct(order, time_limit=deadline - 30)
            if len(S_new) > len(S):
                S = S_new
                D_dict = {k: v.copy() for k, v in used_new.items()}
                S_set = set(S)
                complement = list(set(range(num_points)) - S_set)
                if time.time() < deadline - 20:
                    S, D_dict, S_set, complement = local_search(S, D_dict, deadline - 15, rng, max_passes=30)
    
    return [points[i] for i in S]
# EVOLVE-BLOCK-END
