# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import sys

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Local Search strategy for "no 5 on a sphere" on 8x8x8 grid.
    1. Greedy initialization.
    2. Local search with add and swap moves.
    """
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    sq_norms = (grid ** 2).sum(axis=1)
    
    def can_add_point(p, current_indices):
        """
        Check if point p can be added to the set of points at current_indices.
        """
        m = len(current_indices)
        if m < 4:
            return True
        
        pts = grid[current_indices]
        
        # Lifted vectors relative to p
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = pts - p
        w[:, 3] = sq_norms[current_indices] - (p ** 2).sum()
        
        # Check all combinations of 4
        # To speed up, we can sample if m is very large?
        # No, we need exact validity.
        
        combos = np.array(list(itertools.combinations(range(m), 4)))
        if len(combos) == 0:
            return True
        sub = w[combos]
        dets = np.linalg.det(sub)
        return not np.any(np.abs(dets) < 0.5)

    # 1. Greedy Initialization
    np.random.seed(42)
    order = np.random.permutation(N)
    S = []
    S_set = set()
    for idx in order:
        if can_add_point(grid[idx], S):
            S.append(idx)
            S_set.add(idx)
    
    best_S = S[:]
    best_size = len(S)
    
    start_time = time.time()
    timeout = 500 # Leave margin
    
    # 2. Local Search
    while time.time() - start_time < timeout:
        # Try to add a point
        added = False
        candidates = [i for i in range(N) if i not in S_set]
        if not candidates:
            break
            
        # Shuffle to avoid bias
        np.random.shuffle(candidates)
        
        for idx in candidates:
            if can_add_point(grid[idx], S):
                S.append(idx)
                S_set.add(idx)
                best_S = S[:]
                best_size = len(S)
                added = True
                break
        
        if not added:
            # Try swaps
            # To avoid infinite loops on useless swaps, limit attempts
            swapped = False
            for _ in range(5):
                if not S:
                    break
                rem_idx = np.random.choice(len(S))
                rem_point_idx = S[rem_idx]
                
                # Pick a random candidate to add
                # We can pick from all non-active points
                add_candidates = [i for i in range(N) if i not in S_set]
                if not add_candidates:
                    break
                add_idx = np.random.choice(add_candidates)
                
                # Check if add_idx can be added to S without rem_point_idx
                S_temp = S[:rem_idx] + S[rem_idx+1:]
                if can_add_point(grid[add_idx], S_temp):
                    # Perform swap
                    S[rem_idx] = add_idx
                    S_set.remove(rem_point_idx)
                    S_set.add(add_idx)
                    swapped = True
                    break
            
            if not swapped:
                # If we are stuck, we might want to restart or accept.
                # For now, continue.
                pass

    return np.array(best_S, dtype=np.int64)

# EVOLVE-BLOCK-END
