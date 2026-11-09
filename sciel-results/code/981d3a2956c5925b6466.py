# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates
    using greedy set packing over weight-4 integer vectors to maximize the number of points.
    
    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Leave 10s margin
    
    # Generate pool of weight-4 integer vectors: 4 non-zero entries, each ±1 or ±2
    # This gives a much larger candidate pool than weight-2
    candidates = []
    
    # Generate all combinations of 4 positions out of 11
    pos_combos = list(combinations(range(d), 4))
    
    # For each position combo, generate all sign/magnitude combinations
    # Each of the 4 positions can be: -2, -1, 1, 2 (4 choices)
    # Total: C(11,4) * 4^4 = 330 * 256 = 84480 candidates
    # But we'll use a subset for speed; let's try all first
    
    for pos in pos_combos:
        for a in [-2, -1, 1, 2]:
            for b in [-2, -1, 1, 2]:
                for c in [-2, -1, 1, 2]:
                    for e in [-2, -1, 1, 2]:
                        p = np.zeros(d, dtype=np.int64)
                        p[pos[0]] = a
                        p[pos[1]] = b
                        p[pos[2]] = c
                        p[pos[3]] = e
                        candidates.append(p)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)
    
    # Compute norms squared for all candidates
    norms_sq = np.sum(candidates**2, axis=1)
    # All weight-4 vectors have norm^2 = a^2 + b^2 + c^2 + e^2
    # Minimum norm^2 = 1+1+1+1 = 4, maximum = 4+4+4+4 = 16
    
    # We need min_dist_sq >= max_norm_sq for the selected set
    # For weight-4 vectors, max_norm_sq in our set will be at most 16
    # So we need all pairwise distances >= 16 (if max norm is 16)
    
    # Greedy approach: sort candidates by some heuristic, then greedily add
    # if compatible with all previously added points
    
    # Heuristic: prefer vectors with smaller norm (they're "tighter")
    # Actually, let's just do a simple greedy with randomization
    
    np.random.seed(42)
    
    # Shuffle candidates for diversity
    order = np.random.permutation(n_cands)
    
    selected = []
    selected_norms_sq = []
    
    # For efficiency, we'll check compatibility incrementally
    # A new point p is compatible if dist(p, q) >= max_norm_sq for all q in selected
    # where max_norm_sq = max(norms_sq of all selected points)
    
    # We'll track the current max norm squared
    current_max_norm_sq = 0
    
    # To speed up, we can batch-check using numpy
    # But for large sets, we need to be careful about memory
    
    # Let's use a two-phase approach:
    # Phase 1: Quick greedy with vectorized checks for small batches
    # Phase 2: If time allows, try to improve
    
    batch_size = 1000
    for start_idx in range(0, n_cands, batch_size):
        if time.time() > deadline - 30:  # Leave 30s for finalization
            break
        
        end_idx = min(start_idx + batch_size, n_cands)
        batch_indices = order[start_idx:end_idx]
        batch = candidates[batch_indices]
        
        # For each candidate in batch, check if it's compatible
        for idx in batch_indices:
            if time.time() > deadline - 30:
                break
            
            p = candidates[idx]
            p_norm_sq = norms_sq[idx]
            
            # The new max norm would be max(current_max_norm_sq, p_norm_sq)
            new_max_norm_sq = max(current_max_norm_sq, p_norm_sq)
            
            # Check if p is compatible with all selected points
            # dist(p, q)^2 >= new_max_norm_sq for all q in selected
            compatible = True
            
            if len(selected) > 0:
                sel = np.array(selected)
                diffs = p[None, :] - sel
                dists_sq = np.sum(diffs**2, axis=1)
                if np.any(dists_sq < new_max_norm_sq):
                    compatible = False
            
            if compatible:
                selected.append(p)
                selected_norms_sq.append(p_norm_sq)
                current_max_norm_sq = new_max_norm_sq
    
    if len(selected) < 2:
        # Fallback: 2-point solution
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    points = np.array(selected, dtype=np.int64)
    k = points.shape[0]
    
    # Final verification
    norms_sq_final = np.sum(points**2, axis=1)
    max_norm_sq_final = int(np.max(norms_sq_final))
    
    valid = True
    min_dist_sq = float('inf')
    for i in range(k):
        if time.time() > deadline:
            break
        for j in range(i+1, k):
            diff = points[i] - points[j]
            dist_sq = int(np.sum(diff**2))
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            if dist_sq < max_norm_sq_final:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Try to trim: remove points that cause conflicts
        # Simple approach: keep only points that are mutually compatible
        # Use a greedy removal
        keep = list(range(k))
        for i in range(k):
            if i not in keep:
                continue
            for j in range(i+1, k):
                if j not in keep:
                    continue
                diff = points[i] - points[j]
                dist_sq = int(np.sum(diff**2))
                if dist_sq < max_norm_sq_final:
                    # Remove one of them (remove j)
                    if j in keep:
                        keep.remove(j)
                    break
        
        points = points[keep]
        k = points.shape[0]
        if k < 2:
            return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
        # Re-verify
        norms_sq_final = np.sum(points**2, axis=1)
        max_norm_sq_final = int(np.max(norms_sq_final))
        valid = True
        for i in range(k):
            for j in range(i+1, k):
                diff = points[i] - points[j]
                dist_sq = int(np.sum(diff**2))
                if dist_sq < max_norm_sq_final:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # Final fallback: 2-point solution
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    result = np.around(points).astype(np.int64)
    return result


# EVOLVE-BLOCK-END
