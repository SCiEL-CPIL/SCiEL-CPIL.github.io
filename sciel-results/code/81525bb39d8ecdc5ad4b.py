import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin

    def time_left():
        return deadline - time.time()

    # 1. Generate Base Points: +/- 2 * e_i
    base_points = []
    for i in range(d):
        p1 = np.zeros(d, dtype=np.int64)
        p1[i] = 2
        base_points.append(p1)
        
        p2 = np.zeros(d, dtype=np.int64)
        p2[i] = -2
        base_points.append(p2)
    
    base_points = np.array(base_points)
    
    # 2. Generate Candidate Points: 4 non-zero entries of +/- 1
    # There are C(11, 4) = 330 combinations of indices.
    # Each has 2^4 = 16 sign variations.
    # Total candidates = 5280.
    
    combos = list(combinations(range(d), 4))
    candidates = []
    for comb in combos:
        # Generate all 16 sign patterns for 4 positions
        for mask in range(16):
            v = np.zeros(d, dtype=np.int64)
            for k, idx in enumerate(comb):
                if (mask >> k) & 1:
                    v[idx] = -1
                else:
                    v[idx] = 1
            candidates.append(v)
            
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # 3. Precompute Conflict Matrix
    # Two candidates u, v are incompatible if u . v > 2
    # Since u, v have norm 4, we need u . v <= 2 for distance >= 4.
    dots = candidates @ candidates.T
    conflict = (dots > 2)
    
    # 4. Greedy Search for Maximum Independent Set
    # We want to select the largest subset of candidates such that no two conflict.
    
    best_selected = []
    best_count = 0
    
    # Strategy: Multiple random restarts + Greedy + Fill
    rng = np.random.RandomState(42)
    
    # Precompute degrees for initial ordering heuristic
    degrees = conflict.sum(axis=1)
    
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time_left() < 10:
            break
            
        # Create an ordering of candidates
        if restart == 0:
            # Sort by degree (ascending) - pick least constrained first
            order = np.argsort(degrees)
        elif restart % 2 == 0:
            # Random permutation
            order = rng.permutation(n_cand)
        else:
            # Mix: Bias towards low degree
            # Simple approach: sort by degree + noise
            noise = rng.random(n_cand) * 0.1 * degrees.max()
            order = np.argsort(degrees + noise)
            
        selected = []
        selected_set = set()
        # Use a boolean array for fast lookup of selected items? 
        # Or just check against the conflict matrix row?
        # Efficient greedy:
        # Maintain a bitmask or boolean array of 'available' nodes.
        available = np.ones(n_cand, dtype=bool)
        
        for idx in order:
            if not available[idx]:
                continue
            # Select idx
            selected.append(idx)
            selected_set.add(idx)
            # Remove idx and all its neighbors from available
            available[idx] = False
            available[conflict[idx]] = False
            
        # After greedy, try to fill in any remaining compatible nodes
        # This is a "relaxation" step. Sometimes greedy misses nodes that become available?
        # No, in a standard greedy independent set, once a node is marked unavailable, it stays unavailable.
        # However, we can iterate: check all nodes not in selected_set, if compatible with all selected, add it.
        # Since we used a greedy that marks neighbors as unavailable, any node marked unavailable 
        # was a neighbor of some selected node. 
        # Wait, if A is selected, B is neighbor, B is marked unavailable.
        # If C is selected later, and C is not neighbor of B, B is still unavailable because of A.
        # So the greedy set is maximal. No single node can be added.
        
        # So we just record the count.
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected[:]
            
        # Optional: Local Search (1-swap) to escape local optima?
        # 1-swap doesn't increase size. It might allow a different structure.
        # But for now, 200 restarts should be enough for N=5280.
        
    # 5. Construct Final Array
    # Combine base points and selected candidates
    if best_selected:
        cand_points = candidates[best_selected]
        all_points = np.vstack((base_points, cand_points))
    else:
        all_points = base_points
        
    # 6. Final Verification and Cleanup
    # The problem requires:
    # 1. Shape (k, 11)
    # 2. k >= 2
    # 3. No zero vector
    # 4. Min squared distance >= Max squared norm
    
    # Round and cast
    all_points = np.around(all_points).astype(np.int64)
    
    # Check validity
    k = all_points.shape[0]
    if k < 2:
        # Fallback to a known valid small set if something went wrong
        # e.g., 2 points: [1,0..], [-1,0..]
        fallback = np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        return fallback

    # Calculate norms and distances
    norms_sq = np.sum(all_points**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Calculate min squared distance
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i+1, k):
            diff = all_points[i] - all_points[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
    
    if min_dist_sq < max_norm_sq:
        # Invalid set found? This should not happen with our construction logic.
        # If it does, fall back to base points only.
        # Base points: min dist is 4 (between e_i and e_j is 8, between e_i and -e_i is 16, between e_i and 2e_i? No, base points are +/- 2e_i).
        # Dist between 2e_i and -2e_i is 16.
        # Dist between 2e_i and 2e_j (i!=j) is 8.
        # Max norm is 4.
        # Min dist is 8. 8 >= 4. Valid.
        all_points = base_points
        
    return all_points
