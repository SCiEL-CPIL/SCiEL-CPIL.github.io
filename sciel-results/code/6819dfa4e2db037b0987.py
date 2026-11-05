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
    
    # Calculate squared distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx*dx + dy*dy).astype(np.int32)
    
    # State:
    # S_indices: list of indices in the set
    # S_masks: dict mapping index to bitmask of distances to other points in S
    # in_S: boolean array
    
    def can_add(P, S_indices, S_masks):
        """Check if P can be added to the set."""
        if not S_indices:
            return True
            
        # Get distances from P to all points in S
        d_vals = dist2[P, S_indices]
        
        # Check 1: Distances from P must be unique
        # Using np.unique is fast for small arrays
        if len(np.unique(d_vals)) != len(d_vals):
            return False
            
        # Check 2: Distances must not conflict with existing distances in S
        # For each Q in S, d(P,Q) must not be in S_masks[Q]
        for i, Q in enumerate(S_indices):
            d = int(d_vals[i])
            if S_masks[Q] & (1 << d):
                return False
                
        return True

    def add_point(P, S_indices, S_masks, in_S):
        """Add P to the set."""
        S_indices.append(P)
        in_S[P] = True
        
        # Create mask for P: bits set for distances to all other points in S
        mask_P = 0
        for Q in S_indices[:-1]: # All except P which is last
            d = int(dist2[P, Q])
            mask_P |= (1 << d)
            # Update mask for Q
            S_masks[Q] |= (1 << d)
            
        S_masks[P] = mask_P

    def remove_point(P, S_indices, S_masks, in_S):
        """Remove P from the set."""
        in_S[P] = False
        
        # Remove P from S_indices
        # Find index
        for i, idx in enumerate(S_indices):
            if idx == P:
                S_indices.pop(i)
                break
                
        # Update masks for remaining points
        for Q in S_indices:
            d = int(dist2[P, Q])
            S_masks[Q] &= ~(1 << d)
            
        del S_masks[P]

    def swap_point(P, Q, S_indices, S_masks, in_S):
        """Swap Q for P. Assumes P is not in S and Q is in S."""
        # First check if P can be added if Q was not there?
        # Simpler: Remove Q, then try to add P.
        # But we need to ensure the swap is valid.
        # If we remove Q, the set is valid. Then we add P.
        # So we just need to check if P is valid against S \ {Q}.
        
        # Temporarily remove Q
        remove_point(Q, S_indices, S_masks, in_S)
        
        # Check if P is valid
        if can_add(P, S_indices, S_masks):
            add_point(P, S_indices, S_masks, in_S)
            return True
        else:
            # Revert: Add Q back
            add_point(Q, S_indices, S_masks, in_S)
            return False

    rng = random.Random(42)
    
    # Phase 1: Multiple Greedy Initializations
    best_S_indices = []
    best_S_masks = {}
    best_in_S = np.zeros(num_points, dtype=bool)
    
    orderings = []
    orderings.append(list(range(num_points)))
    rng.shuffle(orderings[0])
    orderings.append(sorted(range(num_points), key=lambda i: points[i][0]))
    orderings.append(sorted(range(num_points), key=lambda i: points[i][1]))
    orderings.append(sorted(range(num_points), key=lambda i: points[i][0] + points[i][1]))
    cx, cy = N/2, N/2
    orderings.append(sorted(range(num_points), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2))
    
    # Add a few more random orderings
    for _ in range(5):
        orderings.append(list(range(num_points)))
        rng.shuffle(orderings[-1])

    for order in orderings:
        S_indices = []
        S_masks = {}
        in_S = np.zeros(num_points, dtype=bool)
        
        for idx_P in order:
            if in_S[idx_P]:
                continue
            if can_add(idx_P, S_indices, S_masks):
                add_point(idx_P, S_indices, S_masks, in_S)
                
        if len(S_indices) > len(best_S_indices):
            best_S_indices = S_indices[:]
            best_S_masks = {k: v for k, v in S_masks.items()}
            best_in_S = in_S.copy()
            
    # Phase 2: Local Search
    S_indices = best_S_indices[:]
    S_masks = {k: v for k, v in best_S_masks.items()}
    in_S = best_in_S.copy()
    
    complement = [i for i in range(num_points) if not in_S[i]]
    best_size = len(S_indices)
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 10:
        if not complement:
            break
            
        # Try to add a random point from complement
        added = False
        # Try a few random candidates
        for _ in range(20):
            if not complement: break
            idx_P = rng.choice(complement)
            if can_add(idx_P, S_indices, S_masks):
                add_point(idx_P, S_indices, S_masks, in_S)
                complement.remove(idx_P)
                added = True
                if len(S_indices) > best_size:
                    best_size = len(S_indices)
                    best_S_indices = S_indices[:]
                    best_S_masks = {k: v for k, v in S_masks.items()}
                    best_in_S = in_S.copy()
                    no_improve_count = 0
                break
                
        if not added:
            # Try swaps
            swapped = False
            if S_indices:
                for _ in range(20):
                    if not complement: break
                    idx_P = rng.choice(complement)
                    idx_Q = rng.choice(S_indices)
                    if swap_point(idx_P, idx_Q, S_indices, S_masks, in_S):
                        # Update complement
                        complement.remove(idx_P)
                        complement.append(idx_Q)
                        swapped = True
                        # Try to add a point after swap to grow
                        for _ in range(10):
                            if not complement: break
                            idx_P_add = rng.choice(complement)
                            if can_add(idx_P_add, S_indices, S_masks):
                                add_point(idx_P_add, S_indices, S_masks, in_S)
                                complement.remove(idx_P_add)
                                if len(S_indices) > best_size:
                                    best_size = len(S_indices)
                                    best_S_indices = S_indices[:]
                                    best_S_masks = {k: v for k, v in S_masks.items()}
                                    best_in_S = in_S.copy()
                                    no_improve_count = 0
                                break
                        break
                        
            if not swapped:
                no_improve_count += 1
            else:
                no_improve_count += 1 # Swapping doesn't increase size, so it's not an improvement in size, but it might unlock future adds. Let's count it as activity.
            
        if no_improve_count > max_no_improve:
            # Restart from best
            S_indices = best_S_indices[:]
            S_masks = {k: v for k, v in best_S_masks.items()}
            in_S = best_in_S.copy()
            complement = [i for i in range(num_points) if not in_S[i]]
            no_improve_count = 0
            # Perturb: remove 1-2 points
            num_remove = rng.randint(1, 2)
            if len(S_indices) > num_remove:
                to_remove = rng.sample(S_indices, num_remove)
                for P in to_remove:
                    remove_point(P, S_indices, S_masks, in_S)
                    if P in complement:
                        pass # It's already not in in_S, but complement list needs updating?
                    # We built complement from in_S, so if we remove from in_S, we must add to complement
                    # But our remove_point doesn't update complement list.
                    # Let's rebuild complement to be safe, or maintain it carefully.
                    # Rebuilding is O(N^2) which is 4096, cheap.
                    pass
            
            # Rebuild complement after perturbation
            complement = [i for i in range(num_points) if not in_S[i]]

    return [points[i] for i in best_S_indices]
# EVOLVE-BLOCK-END
