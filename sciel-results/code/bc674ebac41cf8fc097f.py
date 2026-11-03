"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from itertools import combinations

def _generate_shell(m):
    """Generate all integer vectors in R^D with exactly m non-zero entries of value 1 or -1."""
    if m > D:
        return np.empty((0, D), dtype=np.int64)
    
    # Number of such vectors: C(D, m) * 2^m
    # For D=11, m=4: 330 * 16 = 5280. Very manageable.
    vecs = []
    indices = range(D)
    for comb in combinations(indices, m):
        # Generate all 2^m sign combinations
        # Use bit operations for speed
        for mask in range(1 << m):
            v = np.zeros(D, dtype=np.int64)
            for j, idx in enumerate(comb):
                if mask & (1 << j):
                    v[idx] = 1
                else:
                    v[idx] = -1
            vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def construct():
    deadline = time.time() + 110  # Safety margin before 150s limit
    
    # The constraint is: min pairwise distance >= max norm.
    # For vectors u, v with norms ||u||, ||v||:
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u.v
    # We require ||u - v|| >= max(||u||, ||v||, ... all norms)
    # Let M be the maximum norm in the set.
    # Then we need ||u - v||^2 >= M^2.
    # ||u||^2 + ||v||^2 - 2 u.v >= M^2
    
    # Strategy:
    # 1. Start with a high-norm shell to establish a large M?
    #    Actually, if we mix norms, M is determined by the largest norm vector.
    #    If we use only vectors from a single shell m, then M = sqrt(m).
    #    Then constraint becomes: m + m - 2 u.v >= m  =>  m >= 2 u.v  => u.v <= m/2.
    #    Since u, v are integer vectors with entries in {-1, 0, 1}, u.v is integer.
    #    So we need u.v <= floor(m/2).
    
    # Let's try to pack vectors from shell m=4 (norm sqrt(4)=2).
    # Constraint: u.v <= 2.
    # Since max possible u.v is 4 (if u=v), and min is -4.
    # If we pick a set S such that for all u,v in S, u.v <= 2, then the condition holds.
    # Note: u.v = 2 means they share 3 coordinates with same sign and 1 with opposite?
    # No, u.v = sum u_i v_i.
    # If u, v have support size 4.
    # Intersection size k.
    # If signs match on k coords and mismatch on 4-k? No.
    # Let's just use the inner product bound.
    
    # Can we do better with m=3?
    # Norm sqrt(3). Constraint u.v <= 1.5 => u.v <= 1.
    # Max possible u.v is 3.
    
    # Can we do better with m=5?
    # Norm sqrt(5). Constraint u.v <= 2.5 => u.v <= 2.
    # Max possible u.v is 5.
    
    # Let's try m=4 first as it has a good density.
    # We will use a greedy algorithm to select a maximal subset of shell m=4
    # such that pairwise inner products <= floor(m/2).
    
    m = 4
    max_ip = m // 2  # floor(4/2) = 2
    
    # Generate candidates
    candidates = _generate_shell(m)
    n_cand = len(candidates)
    
    # Shuffle to allow different random restarts if needed, but greedy order matters.
    # A good heuristic: random shuffle.
    idx = np.arange(n_cand)
    np.random.shuffle(idx)
    candidates = candidates[idx]
    
    selected = []
    # max_ip_with_selected[i] = max inner product between candidates[i] and any selected vector
    # We want to add candidates where max_ip_with_selected[i] <= max_ip
    # Actually, we need to check against ALL selected vectors.
    # We can maintain an array `max_ip` of length n_cand, initialized to -inf.
    # When we select a vector v, we update max_ip for all candidates c:
    # max_ip[c] = max(max_ip[c], c.v)
    # Then we can quickly check which candidates are valid: max_ip[c] <= max_ip_bound.
    
    max_ip_arr = np.full(n_cand, -np.inf)
    
    # Precompute Gram matrix? n_cand is ~5000. 5000x5000 is 25M entries.
    # 25M * 8 bytes = 200MB. Might be too much or slow to compute.
    # Let's compute dot products on the fly or in chunks.
    # However, the vectorized update `max_ip_arr = np.maximum(max_ip_arr, candidates @ v)` is O(N).
    # If we select K vectors, total work is O(K * N * D).
    # K can be up to N. So O(N^2 * D).
    # N=5000, D=11. 25M * 11 = 275M ops. This is fast in NumPy.
    
    # Let's try to select as many as possible.
    # We can iterate through candidates in random order.
    # If a candidate is valid (max_ip_arr[i] <= max_ip), we select it.
    # But wait, if we select it, we must update max_ip_arr for ALL remaining candidates.
    
    # To optimize, we can keep a list of "active" candidates (those not yet selected and not yet rejected? No, rejection is dynamic).
    # Actually, once a candidate is rejected because max_ip > max_ip, it might become valid? No, max_ip only increases.
    # So we can maintain a boolean mask of "alive" candidates.
    
    alive = np.ones(n_cand, dtype=bool)
    selected_indices = []
    
    for i in range(n_cand):
        if not alive[i]:
            continue
        
        # Check if candidate i is valid
        if max_ip_arr[i] > max_ip:
            alive[i] = False
            continue
            
        # Select candidate i
        selected_indices.append(i)
        alive[i] = False
        
        # Update max_ip_arr for all alive candidates
        # dot_products = candidates[alive] @ candidates[i]
        # max_ip_arr[alive] = np.maximum(max_ip_arr[alive], dot_products)
        
        # To avoid creating a large temporary array, we can update in place if we handle indices carefully.
        # But boolean indexing creates a copy.
        # Let's just do the full update on the whole array, it's fast enough.
        # candidates @ candidates[i] gives shape (n_cand,)
        dots = candidates @ candidates[i]
        max_ip_arr = np.maximum(max_ip_arr, dots)
        
        # Update alive mask: any candidate with max_ip_arr > max_ip is dead
        # We can do this periodically or every step. Every step is O(N).
        # Since we are in a loop of N steps, total O(N^2).
        # Let's update alive mask.
        alive = max_ip_arr <= max_ip
        alive[i] = False # Just in case, though it's already handled
        
        # If no alive candidates remain, break
        if not np.any(alive):
            break
            
    # Construct the selected vectors
    if len(selected_indices) > 0:
        best_vectors = candidates[selected_indices]
    else:
        # Fallback to seed if something goes wrong
        best_vectors = _generate_seed_vectors()

    return best_vectors

def _generate_seed_vectors():
    """Generate the initial set of vectors (same as seed)."""
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
