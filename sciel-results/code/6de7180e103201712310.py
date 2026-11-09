"""
Combined approach: Vectorized minimum-degree greedy + local search refinement.
Constructs a maximal no-isosceles set on [0, N)^2.
"""

import time
import random
import itertools
import numpy as np

N = 16
DEADLINE_SECONDS = 290  # Safety margin before 300s kill

# Precompute all points and indices
ALL_POINTS = np.array(list(itertools.product(range(N), repeat=2)), dtype=np.int32)
NUM_POINTS = len(ALL_POINTS)

# Precompute squared distances between all pairs
# D[i, j] is squared distance between ALL_POINTS[i] and ALL_POINTS[j]
diff = ALL_POINTS[:, np.newaxis, :] - ALL_POINTS[np.newaxis, :, :]
DIST_MATRIX = np.sum(diff * diff, axis=2)

def _can_add_indices(p_idx, S_indices):
    """
    Check if point p_idx can be added to set S_indices without forming an isosceles triangle.
    Uses vectorized operations for speed.
    """
    if len(S_indices) < 2:
        return True
    
    S_indices = np.array(S_indices)
    # Distances from p to all points in S
    d_p_s = DIST_MATRIX[p_idx, S_indices]
    
    # Check 1: Is p equidistant to any two points in S?
    # If any value in d_p_s appears more than once, it's an isosceles triangle with vertex p.
    if len(np.unique(d_p_s)) < len(d_p_s):
        return False

    # Check 2 & 3: Is d(p, S[i]) == d(S[i], S[j]) or d(p, S[j]) == d(S[i], S[j])?
    # Get the submatrix of distances within S
    D_S = DIST_MATRIX[np.ix_(S_indices, S_indices)]
    
    # We need to check if any d_p_s[i] matches any D_S[i, j] (i != j) 
    # OR any d_p_s[j] matches any D_S[i, j] (i != j).
    
    # Construct a boolean matrix of matches
    # Term 1: d_p_s[i] == D_S[i, j]
    # Term 2: d_p_s[j] == D_S[i, j]
    # Note: D_S[i, i] = 0. Since p is not in S, d_p_s > 0. So diagonal matches are impossible.
    
    # d_p_s shape (k,), D_S shape (k, k)
    match_matrix = (d_p_s[:, np.newaxis] == D_S) | (d_p_s[np.newaxis, :] == D_S)
    
    if np.any(match_matrix):
        return False
        
    return True

def _build_minimal_greedy(order_seed=0):
    """
    Greedy construction using minimum-degree heuristic.
    At each step, among all addable points, pick the one that blocks the fewest others.
    """
    rng = random.Random(order_seed)
    S_indices = []
    addable = set(range(NUM_POINTS))
    
    while addable:
        best_p = None
        best_score = float('inf')
        
        candidates = list(addable)
        rng.shuffle(candidates)
        
        for p_idx in candidates:
            if not _can_add_indices(p_idx, S_indices):
                continue
            
            # Calculate blocking count: how many other addable points become non-addable if p is added?
            # This is expensive: O(M * K^2) where M is addable size, K is S size.
            # Optimization: We can approximate or just use a simpler heuristic if too slow.
            # For N=16, M<=256, K~10-20. 256 * 400 = 100k ops. Fast enough in NumPy?
            # _can_add_indices is vectorized, so it's fast.
            
            new_S = S_indices + [p_idx]
            block_count = 0
            # We only need to check points in `addable` that are not p_idx
            # To save time, we can sample or just check all.
            # Let's check all for accuracy.
            for q_idx in addable:
                if q_idx == p_idx:
                    continue
                if not _can_add_indices(q_idx, new_S):
                    block_count += 1
            
            # Add small random noise to break symmetries
            score = block_count + rng.random() * 0.01
            
            if score < best_score:
                best_score = score
                best_p = p_idx
        
        if best_p is None:
            break
            
        S_indices.append(best_p)
        addable.discard(best_p)
        
        # Update addable set
        new_addable = set()
        for q_idx in addable:
            if _can_add_indices(q_idx, S_indices):
                new_addable.add(q_idx)
        addable = new_addable
        
    return S_indices

def _local_search(S_indices, rng, max_iters=5):
    """
    Local search: try to remove a point and re-maximize.
    If the resulting maximal set is smaller, keep it.
    """
    current_S = list(S_indices)
    current_size = len(current_S)
    
    for _ in range(max_iters):
        improved = False
        # Try removing each point in the current set
        for i in range(len(current_S)):
            p_idx = current_S[i]
            
            # Remove p_idx
            temp_S = [x for j, x in enumerate(current_S) if j != i]
            
            # Re-maximize temp_S
            S_max = list(temp_S)
            addable = set(range(NUM_POINTS)) - set(S_max)
            
            # Greedy addition to maximize
            while True:
                added_any = False
                # Shuffle to be non-deterministic in re-maximization
                candidates = list(addable)
                rng.shuffle(candidates)
                
                for q_idx in candidates:
                    if _can_add_indices(q_idx, S_max):
                        S_max.append(q_idx)
                        addable.discard(q_idx)
                        added_any = True
                        # Update addable? 
                        # For efficiency, we can just restart the inner loop with the new set.
                        # But we need to filter `addable` properly.
                        # Let's just break and restart the while loop to re-filter `addable` from scratch?
                        # No, that's O(N^2). 
                        # Let's filter `addable` in place.
                        # Actually, since we break after adding one, the `addable` set is stale for the next iteration of the while loop?
                        # No, we discard q_idx. But other points might have become non-addable.
                        # So we must re-filter `addable` after adding.
                        # This makes the re-maximization O(K * N * K^2) which is slow.
                        
                        # Alternative: Just break out of the for loop, and in the next while iteration,
                        # re-compute the valid candidates from the full set? No, that's also slow.
                        
                        # Let's stick to a simpler re-maximization:
                        # Just try to add ANY valid point.
                        # We'll do a pass over all remaining addable points.
                        # If we find one, add it, remove from addable, and restart the pass?
                        # Or just continue? If we continue, we might miss points that became invalid.
                        # But if we just want to reach *a* maximal set, we can be lazy.
                        # However, we want a *small* maximal set.
                        
                        # Let's refine:
                        # After adding q_idx, we should remove any points from `addable` that are now blocked.
                        new_addable = set()
                        for r_idx in addable:
                            if _can_add_indices(r_idx, S_max):
                                new_addable.add(r_idx)
                        addable = new_addable
                        break # Restart while loop to check if more can be added
                
                if not added_any:
                    break
            
            # Check if the new maximal set is smaller
            if len(S_max) < current_size:
                current_S = S_max
                current_size = len(S_max)
                improved = True
                break # Restart local search with new set
        
        if not improved:
            break
            
    return current_S

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = float('inf')
    
    # Run multiple greedy starts
    attempts = 50
    
    for seed in range(attempts):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
            
        S_indices = _build_minimal_greedy(order_seed=seed)
        
        # Local search
        rng = random.Random(seed + 1000)
        S_indices = _local_search(S_indices, rng, max_iters=3)
        
        if len(S_indices) < best_size:
            best_size = len(S_indices)
            best_set = S_indices
            
    # Convert indices back to points
    if best_set is None:
        # Fallback: simple greedy
        best_set = _build_minimal_greedy(order_seed=0)
        
    return [tuple(ALL_POINTS[i]) for i in best_set]

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
