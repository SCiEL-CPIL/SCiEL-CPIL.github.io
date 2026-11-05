"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 280  # Safety margin (limit is 330s)
    
    # Grid points
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    # Precompute indices and coordinates
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    # Precompute squared distances
    # dist2[i, j] = (xs[i]-xs[j])^2 + (ys[i]-ys[j])^2
    # 4096x4096 matrix of int32 (max dist is 63^2+63^2 = 7938, fits in int16 but int32 is safer/faster for ops)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx*dx + dy*dy).astype(np.int32)
    
    # Helper to get distance
    def get_dist(i, j):
        return dist2[i, j]

    def is_valid_addition(idx_P, S_indices, D_dict):
        """
        Check if point idx_P can be added to set S_indices.
        S_indices: list of indices of points in S.
        D_dict: dict {idx: set of squared distances to other points in S}
        """
        if not S_indices:
            return True
        
        dists_from_P = set()
        # Optimization: iterate over S_indices
        for idx_Q in S_indices:
            d = dist2[idx_P, idx_Q]
            if d in dists_from_P:
                return False
            dists_from_P.add(d)
            if d in D_dict[idx_Q]:
                return False
        return True

    def is_valid_swap(idx_P, idx_Q, S_indices, D_dict):
        """
        Check if swapping idx_Q for idx_P keeps the set valid.
        S' = S \ {Q} U {P}
        """
        if idx_P in D_dict:
            return False
            
        dists_from_P = set()
        for idx_R in S_indices:
            if idx_R == idx_Q:
                continue
            d_PR = dist2[idx_P, idx_R]
            
            # Check uniqueness among P to others in S'
            if d_PR in dists_from_P:
                return False
            dists_from_P.add(d_PR)
            
            # Check if d_PR is in D'_R
            # D'_R contains d_PR if d_PR in D_R AND d_PR != dist(R, Q)
            if d_PR in D_dict[idx_R]:
                d_RQ = dist2[idx_R, idx_Q]
                if d_PR != d_RQ:
                    return False
            # If d_PR not in D_R, it is not in D'_R. Valid.
            # If d_PR == d_RQ, then d_PR was removed from D_R to form D'_R, so it is NOT in D'_R. Valid.
            
        return True

    def add_point(idx_P, S_indices, D_dict, S_set, complement):
        S_indices.append(idx_P)
        S_set.add(idx_P)
        complement.remove(idx_P)
        
        dists_from_P = set()
        for idx_Q in S_indices:
            if idx_Q == idx_P:
                continue
            d = dist2[idx_P, idx_Q]
            dists_from_P.add(d)
            D_dict[idx_Q].add(d)
        D_dict[idx_P] = dists_from_P

    def remove_point(idx_P, S_indices, D_dict, S_set, complement):
        S_indices.remove(idx_P)
        S_set.discard(idx_P)
        complement.append(idx_P)
        
        dists_from_P = D_dict[idx_P]
        for idx_Q in S_indices:
            d = dist2[idx_P, idx_Q]
            D_dict[idx_Q].discard(d)
        del D_dict[idx_P]

    def swap_point(idx_P, idx_Q, S_indices, D_dict, S_set, complement):
        remove_point(idx_Q, S_indices, D_dict, S_set, complement)
        add_point(idx_P, S_indices, D_dict, S_set, complement)

    # Main Search
    best_S_indices = []
    best_D = {}
    best_S_set = set()
    best_complement = list(range(num_points))
    
    rng = random.Random(42)
    
    # Phase 1: Multiple Greedy Initializations
    # Try different orderings to get a good starting set
    
    orderings = []
    # Random
    orderings.append(list(range(num_points)))
    rng.shuffle(orderings[0])
    # Sorted by X
    orderings.append(sorted(range(num_points), key=lambda i: points[i][0]))
    # Sorted by Y
    orderings.append(sorted(range(num_points), key=lambda i: points[i][1]))
    # Sorted by X+Y
    orderings.append(sorted(range(num_points), key=lambda i: points[i][0] + points[i][1]))
    # Sorted by distance from center
    cx, cy = N/2, N/2
    orderings.append(sorted(range(num_points), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2))
    
    for order in orderings:
        S_indices = []
        D_dict = {}
        S_set = set()
        complement = list(range(num_points))
        
        for idx_P in order:
            if idx_P in S_set:
                continue
            if is_valid_addition(idx_P, S_indices, D_dict):
                add_point(idx_P, S_indices, D_dict, S_set, complement)
                
        if len(S_indices) > len(best_S_indices):
            best_S_indices = S_indices[:]
            best_D = {k: v.copy() for k, v in D_dict.items()}
            best_S_set = S_set.copy()
            best_complement = complement[:]
            
    # Phase 2: Local Search
    # We operate on the current best state
    S_indices = best_S_indices[:]
    D_dict = {k: v.copy() for k, v in best_D.items()}
    S_set = best_S_set.copy()
    complement = best_complement[:]
    
    no_improve_count = 0
    max_no_improve = 100  # Lower threshold to trigger perturbations more often
    
    while time.time() < deadline:
        if not complement:
            break
            
        # Choose action: Add or Swap
        # Bias towards Add if possible, especially when set is small
        if len(S_indices) < 20:
            action = 'add' if rng.random() < 0.8 else 'swap'
        else:
            action = 'add' if rng.random() < 0.4 else 'swap'
        
        if action == 'add':
            # Try to find a valid add
            # Sample a few candidates
            found = False
            # Increase sampling attempts to find valid adds faster
            for _ in range(20):
                if not complement: break
                idx_P = rng.choice(complement)
                if is_valid_addition(idx_P, S_indices, D_dict):
                    add_point(idx_P, S_indices, D_dict, S_set, complement)
                    found = True
                    if len(S_indices) > len(best_S_indices):
                        best_S_indices = S_indices[:]
                        best_D = {k: v.copy() for k, v in D_dict.items()}
                        best_S_set = S_set.copy()
                        best_complement = complement[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    break
            if not found:
                no_improve_count += 1
        else:
            # Swap
            # Try to find a valid swap
            found = False
            for _ in range(20):
                if not complement or not S_indices: break
                idx_P = rng.choice(complement)
                idx_Q = rng.choice(S_indices)
                if is_valid_swap(idx_P, idx_Q, S_indices, D_dict):
                    swap_point(idx_P, idx_Q, S_indices, D_dict, S_set, complement)
                    found = True
                    # Swap doesn't increase size, but might enable future adds
                    # Check if we can add now
                    if rng.random() < 0.5 and complement:
                         idx_P_add = rng.choice(complement)
                         if is_valid_addition(idx_P_add, S_indices, D_dict):
                             add_point(idx_P_add, S_indices, D_dict, S_set, complement)
                             if len(S_indices) > len(best_S_indices):
                                 best_S_indices = S_indices[:]
                                 best_D = {k: v.copy() for k, v in D_dict.items()}
                                 best_S_set = S_set.copy()
                                 best_complement = complement[:]
                                 no_improve_count = 0
                    break
            if not found:
                no_improve_count += 1
                
        if no_improve_count > max_no_improve:
            # Perturb
            S_indices = best_S_indices[:]
            D_dict = {k: v.copy() for k, v in best_D.items()}
            S_set = best_S_set.copy()
            complement = best_complement[:]
            no_improve_count = 0
            # Remove 2-3 random points to escape local optimum
            if len(S_indices) > 3:
                idx1 = rng.randrange(len(S_indices))
                idx2 = rng.randrange(len(S_indices))
                idx3 = rng.randrange(len(S_indices))
                # Ensure distinct indices
                if idx1 == idx2: idx2 = (idx2 + 1) % len(S_indices)
                if idx1 == idx3: idx3 = (idx3 + 1) % len(S_indices)
                if idx2 == idx3: idx3 = (idx3 + 1) % len(S_indices)
                
                P1 = S_indices[idx1]
                P2 = S_indices[idx2]
                P3 = S_indices[idx3]
                
                remove_point(P1, S_indices, D_dict, S_set, complement)
                remove_point(P2, S_indices, D_dict, S_set, complement)
                remove_point(P3, S_indices, D_dict, S_set, complement)
            elif len(S_indices) > 1:
                idx1 = rng.randrange(len(S_indices))
                idx2 = rng.randrange(len(S_indices))
                if idx1 == idx2: idx2 = (idx2 + 1) % len(S_indices)
                P1 = S_indices[idx1]
                P2 = S_indices[idx2]
                remove_point(P1, S_indices, D_dict, S_set, complement)
                remove_point(P2, S_indices, D_dict, S_set, complement)

    # Convert indices back to points
    return [points[i] for i in best_S_indices]
# EVOLVE-BLOCK-END
