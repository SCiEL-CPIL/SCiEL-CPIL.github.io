import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Safety margin before 300s kill

    N = 64
    M = N * N

    # Precompute all points as (x, y)
    points = [(x, y) for x in range(N) for y in range(N)]
    coords = np.array(points, dtype=np.int32)

    # Precompute squared distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist_mat = np.sum(diff**2, axis=-1).astype(np.int32)
    del diff

    # Max distance is (N-1)^2 + (N-1)^2 = 2 * 63^2 = 7938
    # We use Python integers as bitsets. Bit d is set if distance d is present.

    def try_add(S, masks, p):
        """
        Check if p can be added to S.
        S: set of indices
        masks: dict mapping index to int bitset of distances to S
        """
        if not S:
            return True
            
        S_list = list(S)
        
        # Step 1: Check for duplicate distances from p to points in S
        # Use numpy for speed
        d_vals = dist_mat[p, S_list]
        if len(np.unique(d_vals)) < len(d_vals):
            return False

        # Step 2: For each q in S, check if d(p,q) is already in masks[q]
        # Iterate in Python
        for i, q in enumerate(S_list):
            d_pq = d_vals[i]
            # Check if bit d_pq is set in masks[q]
            if (masks[q] >> d_pq) & 1:
                return False

        return True

    def add_point(S, masks, p):
        """Add point p to S and update masks."""
        S.add(p)
        masks[p] = 0
        for q in S:
            if q == p:
                continue
            d = dist_mat[p, q]
            # Set bit d in masks[p] and masks[q]
            bit = 1 << d
            masks[p] |= bit
            masks[q] |= bit

    def remove_point(S, masks, p):
        """Remove point p from S and update masks."""
        for q in S:
            if q == p:
                continue
            d = dist_mat[p, q]
            # Clear bit d in masks[q]
            masks[q] &= ~(1 << d)
        # masks[p] is discarded
        S.discard(p)
        # Note: We don't delete masks[p] immediately to avoid key error if called again?
        # Better to delete to save memory, but we must handle it in try_add?
        # try_add only accesses masks[q] for q in S. So deleting masks[p] is safe.
        del masks[p]

    def greedy_construct():
        S = set()
        masks = {}
        candidates = list(range(M))
        random.shuffle(candidates)

        for p in candidates:
            if time.time() > deadline:
                break
            if try_add(S, masks, p):
                add_point(S, masks, p)

        return S, masks

    def local_search(S, masks):
        S = set(S)
        masks = {k: v for k, v in masks.items()} # Copy ints
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            S_list = list(S)
            random.shuffle(S_list)
            
            # Try adding points not in S
            candidates = [i for i in range(M) if i not in S]
            random.shuffle(candidates)
            
            for p in candidates:
                if time.time() > deadline - 10:
                    break
                if try_add(S, masks, p):
                    add_point(S, masks, p)
                    improved = True
                    break
            
            if improved:
                continue
            
            # If no addition, try swaps
            if S_list:
                swaps_tried = 0
                max_swaps = 50
                while swaps_tried < max_swaps and time.time() < deadline - 10:
                    q_to_remove = random.choice(S_list)
                    remove_point(S, masks, q_to_remove)
                    
                    candidates = [i for i in range(M) if i not in S]
                    random.shuffle(candidates)
                    
                    added = False
                    for p in candidates:
                        if time.time() > deadline - 10:
                            break
                        if p == q_to_remove:
                            continue
                        if try_add(S, masks, p):
                            add_point(S, masks, p)
                            added = True
                            break
                    
                    if not added:
                        # Revert
                        add_point(S, masks, q_to_remove)
                        swaps_tried += 1
                    else:
                        improved = True
                        break
                if not improved:
                    break
            else:
                break
                
        return S, masks

    best_S = set()
    best_masks = {}

    # Multiple restarts
    num_restarts = 40
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break

        S, masks = greedy_construct()
        S, masks = local_search(S, masks)

        if len(S) > len(best_S):
            best_S = S.copy()
            best_masks = {k: v for k, v in masks.items()}

    # Final polish on the best set found
    if time.time() < deadline - 5 and len(best_S) > 0:
        best_S, best_masks = local_search(best_S, best_masks)

    return [points[i] for i in best_S]

# For testing
if __name__ == "__main__":
    res = construct()
    print(len(res))
