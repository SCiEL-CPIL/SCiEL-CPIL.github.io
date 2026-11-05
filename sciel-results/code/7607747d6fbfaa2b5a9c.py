"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 285  # Safety margin
    
    # Grid points
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    # Create coordinate arrays
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    # Calculate squared distances using int32 to save memory
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx*dx + dy*dy).astype(np.int32)
    
    # Precompute for faster access: use a 2D numpy array
    # dist2[i, j] gives squared distance between point i and point j
    
    def is_valid_addition(idx_P, S_indices, D_dict):
        """
        Check if point idx_P can be added to set S_indices.
        """
        if not S_indices:
            return True
        dists_from_P = set()
        # Get all distances from P to points in S at once using numpy
        S_arr = np.array(S_indices, dtype=np.int32)
        dists_to_S = dist2[idx_P, S_arr]
        
        for idx_Q, d in zip(S_indices, dists_to_S):
            d_val = int(d)
            if d_val in dists_from_P:
                return False
            dists_from_P.add(d_val)
            if d_val in D_dict[idx_Q]:
                return False
        return True

    def is_valid_swap(idx_P, idx_Q, S_indices, D_dict):
        """
        Check if swapping idx_Q for idx_P keeps the set valid.
        """
        if idx_P in D_dict:
            return False
            
        dists_from_P = set()
        for idx_R in S_indices:
            if idx_R == idx_Q:
                continue
            d_PR = int(dist2[idx_P, idx_R])
            
            if d_PR in dists_from_P:
                return False
            dists_from_P.add(d_PR)
            
            if d_PR in D_dict[idx_R]:
                d_RQ = int(dist2[idx_R, idx_Q])
                if d_PR != d_RQ:
                    return False
        return True

    def add_point(idx_P, S_indices, D_dict, S_set, complement):
        S_indices.append(idx_P)
        S_set.add(idx_P)
        complement.remove(idx_P)
        
        dists_from_P = set()
        for idx_Q in S_indices:
            if idx_Q == idx_P:
                continue
            d = int(dist2[idx_P, idx_Q])
            dists_from_P.add(d)
            D_dict[idx_Q].add(d)
        D_dict[idx_P] = dists_from_P

    def remove_point(idx_P, S_indices, D_dict, S_set, complement):
        S_indices.remove(idx_P)
        S_set.discard(idx_P)
        complement.append(idx_P)
        
        for idx_Q in S_indices:
            d = int(dist2[idx_P, idx_Q])
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
    orderings = []
    orderings.append(list(range(num_points)))
    rng.shuffle(orderings[0])
    orderings.append(sorted(range(num_points), key=lambda i: points[i][0]))
    orderings.append(sorted(range(num_points), key=lambda i: points[i][1]))
    orderings.append(sorted(range(num_points), key=lambda i: points[i][0] + points[i][1]))
    cx, cy = N/2, N/2
    orderings.append(sorted(range(num_points), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2))
    # Add a few more random orderings
    for _ in range(3):
        orderings.append(list(range(num_points)))
        rng.shuffle(orderings[-1])
    
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
    S_indices = best_S_indices[:]
    D_dict = {k: v.copy() for k, v in best_D.items()}
    S_set = best_S_set.copy()
    complement = best_complement[:]
    
    no_improve_count = 0
    max_no_improve = 300
    # Track how many times we've tried each type of action
    add_attempts = 0
    swap_attempts = 0
    
    while time.time() < deadline - 15:
        if not complement:
            break
            
        # Dynamically choose action based on recent history
        # If adds have been failing, try swaps; if swaps have been failing, try adds
        if add_attempts > 50 and swap_attempts < 50:
            action = 'swap'
        elif swap_attempts > 50 and add_attempts < 50:
            action = 'add'
        else:
            # Default: try add more often when set is small
            if len(S_indices) < 15:
                action = 'add' if rng.random() < 0.6 else 'swap'
            else:
                action = 'add' if rng.random() < 0.4 else 'swap'
        
        if action == 'add':
            add_attempts += 1
            swap_attempts = max(0, swap_attempts - 1)
            found = False
            # Try more candidates for add
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
            swap_attempts += 1
            add_attempts = max(0, add_attempts - 1)
            found = False
            # Try more candidates for swap
            for _ in range(20):
                if not complement or not S_indices: break
                idx_P = rng.choice(complement)
                idx_Q = rng.choice(S_indices)
                if is_valid_swap(idx_P, idx_Q, S_indices, D_dict):
                    swap_point(idx_P, idx_Q, S_indices, D_dict, S_set, complement)
                    found = True
                    # After a swap, try to add a point to grow the set
                    for _ in range(10):
                        if not complement: break
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
                    break
            if not found:
                no_improve_count += 1
                
        if no_improve_count > max_no_improve:
            # Perturb: restart from best and remove some points
            S_indices = best_S_indices[:]
            D_dict = {k: v.copy() for k, v in best_D.items()}
            S_set = best_S_set.copy()
            complement = best_complement[:]
            no_improve_count = 0
            add_attempts = 0
            swap_attempts = 0
            # Remove 1-3 random points to create space
            num_remove = rng.randint(1, 3)
            if len(S_indices) > num_remove:
                to_remove = rng.sample(S_indices, num_remove)
                for P in to_remove:
                    remove_point(P, S_indices, D_dict, S_set, complement)
                
    # Convert indices back to points
    return [points[i] for i in best_S_indices]
# EVOLVE-BLOCK-END
