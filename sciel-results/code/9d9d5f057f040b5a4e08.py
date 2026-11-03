"""Rewritten program for kissing number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Start with the seed: all pairs of coordinates with ±1
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    
    def is_valid(new_v, existing):
        """Check if new_v can be added to existing set."""
        new_norm_sq = np.dot(new_v, new_v)
        if new_norm_sq == 0:
            return False
        for v in existing:
            diff = new_v - v
            dist_sq = np.dot(diff, diff)
            if dist_sq < new_norm_sq:
                return False
        # Also check against max norm of existing
        for v in existing:
            v_norm_sq = np.dot(v, v)
            diff = new_v - v
            dist_sq = np.dot(diff, diff)
            if dist_sq < v_norm_sq:
                return False
        return True
    
    def check_all_valid(vecs):
        """Verify all pairwise constraints hold."""
        n = len(vecs)
        norms_sq = np.array([np.dot(v, v) for v in vecs])
        max_norm_sq = norms_sq.max()
        for i in range(n):
            for j in range(i+1, n):
                diff = vecs[i] - vecs[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < max_norm_sq:
                    return False
        return True
    
    # Try to add vectors with 3 nonzero coordinates
    # Pattern: three coordinates with ±1, sum of signs = 0 mod something? 
    # Actually, let's try vectors like (1,1,1,0,...), (1,1,-1,0,...), etc.
    
    candidate_pool = []
    
    # Vectors with exactly 3 nonzero entries, each ±1
    for idxs in itertools.combinations(range(D), 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[idxs[0]] = s1
                    v[idxs[1]] = s2
                    v[idxs[2]] = s3
                    candidate_pool.append(v)
    
    # Also try vectors with 4 nonzero entries
    for idxs in itertools.combinations(range(D), 4):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    for s4 in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[idxs[0]] = s1
                        v[idxs[1]] = s2
                        v[idxs[2]] = s3
                        v[idxs[3]] = s4
                        candidate_pool.append(v)
    
    candidate_pool = np.array(candidate_pool, dtype=np.int64)
    
    # Greedy: try to add candidates that are valid
    current = vectors.copy()
    current_list = list(vectors)
    
    # Shuffle candidates for diversity
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(candidate_pool))
    candidate_pool = candidate_pool[perm]
    
    added_count = 0
    for idx in range(len(candidate_pool)):
        if time.time() > deadline - 10.0:
            break
        new_v = candidate_pool[idx]
        # Quick check: norm should not exceed current max norm too much
        new_norm_sq = np.dot(new_v, new_v)
        current_max_norm_sq = max(np.dot(v, v) for v in current_list) if current_list else 0
        
        # The constraint requires dist >= max(norm_new, norm_existing)
        # So if new norm is larger, all existing must be at distance >= new_norm
        # This is expensive to check, so do a quick filter first
        
        # Quick filter: check against a subset
        if len(current_list) > 0:
            # Sample some vectors for quick rejection
            sample_size = min(50, len(current_list))
            sample_idx = rng.choice(len(current_list), size=sample_size, replace=False)
            valid_so_far = True
            for si in sample_idx:
                v = current_list[si]
                v_norm_sq = np.dot(v, v)
                diff = new_v - v
                dist_sq = np.dot(diff, diff)
                required = max(new_norm_sq, v_norm_sq)
                if dist_sq < required:
                    valid_so_far = False
                    break
            
            if not valid_so_far:
                continue
        
        # Full check
        valid = True
        for v in current_list:
            v_norm_sq = np.dot(v, v)
            diff = new_v - v
            dist_sq = np.dot(diff, diff)
            required = max(new_norm_sq, v_norm_sq)
            if dist_sq < required:
                valid = False
                break
        
        if valid:
            current_list.append(new_v)
            added_count += 1
    
    current = np.array(current_list, dtype=np.int64)
    
    # Final verification
    if not check_all_valid(current):
        # Fall back to seed
        return vectors
    
    return current

# EVOLVE-BLOCK-END
