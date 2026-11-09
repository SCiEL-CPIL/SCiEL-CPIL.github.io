import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    d = 11
    
    # Fallback: D_11 roots (220 points) - weight-2 vectors with all sign combos
    def get_d11_points():
        pts = []
        for i, j in combinations(range(d), 2):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    pts.append(p)
        return np.array(pts)
    
    fallback_pts = get_d11_points()
    best_pts = fallback_pts.copy()
    best_k = fallback_pts.shape[0]
    
    # Strategy: Build a large set of weight-3 vectors (norm_sq = 3)
    # Constraint: No two vectors can share 2 or more indices (otherwise dist_sq < 3)
    # This is equivalent to finding a large set of 3-subsets of {0..10} with pairwise intersection <= 1
    # This is a packing design P(11, 3, 2) or a 2-(11,3,1) packing
    # The maximum size of a 2-(11,3,1) packing is known to be 12 (a Steiner triple system doesn't exist for v=11, but a near Steiner system does)
    # Actually, let's check: A Steiner triple system STS(v) exists if v ≡ 1 or 3 (mod 6). 11 ≡ 5 (mod 6), so no STS(11).
    # The maximum number of triples in a 2-(11,3,1) packing is floor(11/3 * floor(10/2)) = floor(11/3 * 5) = floor(18.33) = 18? No.
    # The bound is floor(v/3 * floor((v-1)/2)) = floor(11/3 * 5) = 18.
    # But actually, for v=11, the maximum size of a 2-(11,3,1) packing is 12.
    # Let's just use a greedy algorithm to find a large set.
    
    # Generate all 3-subsets
    all_triples = list(combinations(range(d), 3))
    
    # Greedy selection: pick triples that don't share 2 elements with any already picked
    selected_triples = []
    used_pairs = set()  # Track all pairs used
    
    for triple in all_triples:
        if time.time() > deadline:
            break
        
        # Check if any pair in this triple is already used
        valid = True
        for i in range(3):
            for j in range(i+1, 3):
                pair = (triple[i], triple[j])
                if pair in used_pairs:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            selected_triples.append(triple)
            # Add all pairs from this triple to used_pairs
            for i in range(3):
                for j in range(i+1, 3):
                    used_pairs.add((triple[i], triple[j]))
    
    # Now assign signs to each triple to maximize the set size
    # For each triple, we can choose 8 sign combinations
    # But we need to ensure that the resulting vectors are "far enough" from each other
    # Actually, since the supports are disjoint in pairs (intersection <= 1), the distance between any two weight-3 vectors with supports intersecting in <= 1 element is:
    # - If intersection = 0: dist_sq = 6
    # - If intersection = 1: dist_sq = 4 (if signs agree on the common element) or dist_sq = 8 (if signs disagree)
    # So any sign assignment works! We just need to pick one sign per triple.
    
    # Let's pick all +1 signs for simplicity
    weight3_pts = []
    for triple in selected_triples:
        p = np.zeros(d, dtype=np.int64)
        for idx in triple:
            p[idx] = 1
        weight3_pts.append(p)
    
    weight3_pts = np.array(weight3_pts) if weight3_pts else np.array([]).reshape(0, d)
    
    # Verify the weight-3 set
    if weight3_pts.shape[0] >= 2:
        k3 = weight3_pts.shape[0]
        max_norm_sq = np.max(np.sum(weight3_pts**2, axis=1))
        min_dist_sq = float('inf')
        for i in range(k3):
            for j in range(i+1, k3):
                diff = weight3_pts[i] - weight3_pts[j]
                dist_sq = np.sum(diff**2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
        if min_dist_sq >= max_norm_sq:
            if k3 > best_k:
                best_pts = weight3_pts
                best_k = k3
    
    # Now try to add weight-2 vectors to the weight-3 set
    # A weight-2 vector (norm_sq = 2) can be added if its distance to all weight-3 vectors is >= 3
    # (since max_norm_sq would become 3)
    # Let's try to greedily add weight-2 vectors
    
    if best_k >= 2 and best_pts.shape[0] >= 2:
        current_pts = best_pts.tolist()
        current_k = best_k
        max_norm_sq = 3  # Assuming we have weight-3 vectors
        
        # Generate all weight-2 vectors
        all_weight2 = []
        for i, j in combinations(range(d), 2):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    all_weight2.append(p)
        
        added_count = 0
        for p in all_weight2:
            if time.time() > deadline:
                break
            
            # Check if p is far enough from all current points
            valid = True
            for q in current_pts:
                diff = p - np.array(q)
                dist_sq = np.sum(diff**2)
                if dist_sq < max_norm_sq:
                    valid = False
                    break
            
            if valid:
                current_pts.append(p.tolist())
                added_count += 1
        
        if added_count > 0:
            new_pts = np.array(current_pts)
            k_new = new_pts.shape[0]
            
            # Verify the combined set
            max_norm_sq_new = np.max(np.sum(new_pts**2, axis=1))
            min_dist_sq_new = float('inf')
            for i in range(k_new):
                for j in range(i+1, k_new):
                    diff = new_pts[i] - new_pts[j]
                    dist_sq = np.sum(diff**2)
                    if dist_sq < min_dist_sq_new:
                        min_dist_sq_new = dist_sq
            
            if min_dist_sq_new >= max_norm_sq_new:
                if k_new > best_k:
                    best_pts = new_pts
                    best_k = k_new
    
    # Final verification
    if best_pts.shape[1] != d:
        return fallback_pts
        
    k = best_pts.shape[0]
    if k < 2:
        return fallback_pts
    
    # Check no zero vector
    if np.any(np.all(best_pts == 0, axis=1)):
        mask = np.any(best_pts != 0, axis=1)
        best_pts = best_pts[mask]
        k = best_pts.shape[0]
        if k < 2:
            return fallback_pts
    
    # Check min dist sq >= max norm sq
    max_norm_sq = np.max(np.sum(best_pts**2, axis=1))
    
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i+1, k):
            diff = best_pts[i] - best_pts[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            if min_dist_sq < max_norm_sq:
                # Invalid, return fallback
                return fallback_pts
    
    return best_pts
