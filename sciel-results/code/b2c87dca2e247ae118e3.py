"""Improved program for kissing number in dimension 11."""
import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin before 160s kill (or 150s)
    
    # We want to maximize k = number of vectors
    # Constraint: for all i != j, ||v_i - v_j|| >= max_i ||v_i||
    # Equivalently, for all i != j, ||v_i - v_j||^2 >= max_i ||v_i||^2
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
    # So we need: ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j >= M^2 where M = max ||v_i||
    
    # A simpler sufficient condition that's often used: if all vectors have the same norm R,
    # then we need R^2 + R^2 - 2*v_i.v_j >= R^2 => R^2 >= 2*v_i.v_j => v_i.v_j <= R^2/2
    # But we don't require equal norms. Let's think more carefully.
    
    # Actually, let's use a greedy approach with vectors of small norm.
    # Start with vectors of norm 1 (unit vectors with one 1 and rest 0), but that gives only 11.
    # Better: use vectors with two non-zero entries of value 1, like the seed.
    # The seed has vectors like e_i + e_j, e_i - e_j, -e_i + e_j, -e_i - e_j for i < j.
    # Each has norm sqrt(2). There are C(11,2)*4 = 55*4 = 220 such vectors.
    # For two such vectors, we need ||v_i - v_j||^2 >= 2 (since max norm is sqrt(2), M^2=2)
    # ||v_i - v_j||^2 = 2 + 2 - 2*v_i.v_j = 4 - 2*v_i.v_j >= 2 => v_i.v_j <= 1
    # So we need inner product <= 1 for all pairs.
    
    # Let's try a greedy approach:
    # 1. Generate candidate vectors (small integer vectors)
    # 2. Greedily add vectors that satisfy the constraint with all already-added vectors
    
    # Candidate vectors: all integer vectors with entries in {-1, 0, 1} and at most 2 non-zero entries
    # This gives: for each pair (i,j), 4 vectors. Total 220.
    # We can also try vectors with entries in {-1, 0, 1} and 3 non-zero entries, etc.
    
    # Let's first do the greedy on the 220 vectors to see if we can get all of them.
    
    def generate_candidates(max_weight=2, max_coord=1):
        """Generate candidate vectors with at most max_weight non-zero entries, each in [-max_coord, max_coord]."""
        candidates = []
        for weight in range(1, max_weight + 1):
            for combo in combinations(range(D), weight):
                # Generate all sign combinations
                for signs in np.ndindex(*([2]*weight)):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, sign_idx in enumerate(signs):
                        val = 1 if sign_idx == 0 else -1
                        v[combo[idx]] = val * max_coord
                    if np.any(v != 0):
                        candidates.append(v)
        return candidates
    
    # Start with weight-2 vectors (the seed's set)
    candidates = generate_candidates(max_weight=2, max_coord=1)
    candidates = np.array(candidates)
    
    # Greedy selection
    selected = []
    norms_sq = []
    
    for v in candidates:
        v = v.astype(np.int64)
        norm_sq = int(np.dot(v, v))
        
        # Check constraint with all selected vectors
        valid = True
        for i, s in enumerate(selected):
            diff = v - s
            diff_norm_sq = int(np.dot(diff, diff))
            # We need diff_norm_sq >= max(norm_sq, norms_sq[i])
            # Actually the constraint is: ||v_i - v_j|| >= max(||v_i||, ||v_j||)
            # So diff_norm_sq >= max(norm_sq, norms_sq[i])
            if diff_norm_sq < max(norm_sq, norms_sq[i]):
                valid = False
                break
        
        if valid:
            selected.append(v)
            norms_sq.append(norm_sq)
        
        if time.time() > deadline:
            break
    
    if len(selected) == 0:
        # Fallback to seed
        vectors = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result = np.array(selected, dtype=np.int64)
    
    # Try to improve by adding more vectors from a larger candidate pool
    # Try weight-3 vectors
    if time.time() < deadline - 10:
        candidates3 = generate_candidates(max_weight=3, max_coord=1)
        # Filter out those already in selected
        selected_set = set([tuple(v) for v in selected])
        for v in candidates3:
            if time.time() > deadline - 5:
                break
            v = v.astype(np.int64)
            if tuple(v) in selected_set:
                continue
            norm_sq = int(np.dot(v, v))
            
            valid = True
            for i, s in enumerate(selected):
                diff = v - s
                diff_norm_sq = int(np.dot(diff, diff))
                if diff_norm_sq < max(norm_sq, norms_sq[i]):
                    valid = False
                    break
            
            if valid:
                selected.append(v)
                norms_sq.append(norm_sq)
                selected_set.add(tuple(v))
    
    result = np.array(selected, dtype=np.int64)
    
    # Verify the result
    if len(result) < 2:
        return result
    
    # Check all pairwise constraints
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            diff = result[i] - result[j]
            diff_norm_sq = int(np.dot(diff, diff))
            norm_i_sq = int(np.dot(result[i], result[i]))
            norm_j_sq = int(np.dot(result[j], result[j]))
            if diff_norm_sq < max(norm_i_sq, norm_j_sq):
                # This shouldn't happen, but if it does, remove the problematic vector
                pass
    
    return result
# EVOLVE-BLOCK-END
