"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    rng = np.random.default_rng(42)
    
    # Generate a large pool of random integer vectors with entries in {-2,-1,0,1,2}
    # Focus on vectors with small L2 norm to maximize packing density
    # Weight: number of non-zero entries
    # Try weights 2, 3, 4 with entries from {-2,-1,1,2}
    
    candidate_pool = []
    
    # Generate candidates by weight (number of non-zero entries)
    for weight in [2, 3, 4]:
        # Number of ways to choose positions
        # For weight w in D=11, C(11,w) positions
        # For each position set, 4^w sign/magnitude combinations
        # Limit total candidates to keep memory and time manageable
        max_candidates_per_weight = 50000
        count = 0
        while count < max_candidates_per_weight:
            # Randomly pick weight positions
            positions = rng.choice(D, size=weight, replace=False)
            # Randomly assign values from {-2,-1,1,2}
            values = rng.choice([-2, -1, 1, 2], size=weight)
            v = np.zeros(D, dtype=np.int64)
            v[positions] = values
            # Skip zero vectors (shouldn't happen but just in case)
            if np.any(v != 0):
                candidate_pool.append(v)
                count += 1
    
    candidate_pool = np.array(candidate_pool, dtype=np.int64)
    n_cands = len(candidate_pool)
    
    # Precompute norms squared for candidates
    cand_norms_sq = np.sum(candidate_pool * candidate_pool, axis=1)
    
    # Greedy independent set: process candidates in a fixed order
    # A vector is "accepted" if it has valid distance to all previously accepted vectors
    # Condition: dist(u,v)^2 >= max(norm(u)^2, norm(v)^2)
    # i.e., norm(u)^2 + norm(v)^2 - 2*dot(u,v) >= max(norm(u)^2, norm(v)^2)
    # Simplifies to: min(norm(u)^2, norm(v)^2) >= 2*dot(u,v)
    # (assuming dot can be negative, this is automatically satisfied)
    # More carefully: norm(u)^2 + norm(v)^2 - 2*dot >= max(nu2, nv2)
    # => min(nu2, nv2) >= 2*dot(u,v)
    
    accepted = []
    accepted_norms_sq = []
    
    # Sort candidates by norm to process smaller ones first (they're easier to pack)
    order = np.argsort(cand_norms_sq)
    
    for idx in order:
        if time.time() > deadline:
            break
        
        cv = candidate_pool[idx]
        cv_norm_sq = cand_norms_sq[idx]
        
        if len(accepted) == 0:
            accepted.append(cv)
            accepted_norms_sq.append(cv_norm_sq)
            continue
        
        # Check against all accepted vectors
        # dist^2 = cv_norm_sq + acc_norm_sq - 2 * dot
        # Need: dist^2 >= max(cv_norm_sq, acc_norm_sq)
        # i.e., min(cv_norm_sq, acc_norm_sq) >= 2 * dot(cv, acc)
        
        acc_array = np.array(accepted)
        dots = acc_array @ cv
        acc_norms = np.array(accepted_norms_sq)
        
        # Condition: min(cv_norm_sq, acc_norms) >= 2 * dots
        min_norms = np.minimum(cv_norm_sq, acc_norms)
        
        if np.all(min_norms >= 2 * dots):
            accepted.append(cv)
            accepted_norms_sq.append(cv_norm_sq)
    
    if len(accepted) == 0:
        # Fallback: return at least the baseline
        vecs = []
        for i in range(D):
            for j in range(i+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        vecs.append(v)
        return np.array(vecs, dtype=np.int64)
    
    return np.array(accepted, dtype=np.int64)
# EVOLVE-BLOCK-END
