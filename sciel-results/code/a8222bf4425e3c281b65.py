import numpy as np
import time
import itertools
from scipy.optimize import milp, LinearConstraint, Bounds

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Generate Candidate Pool
    # We focus on the D11 lattice structure.
    # Weight 2 vectors (roots) are highly compatible.
    # Weight 4 vectors provide additional directions.
    # We include Weight 2 and Weight 4 vectors.
    
    candidates = []
    
    # Weight 2 vectors: (±1, ±1, 0, ...)
    # Count: C(11,2) * 4 = 55 * 4 = 220
    for comb in itertools.combinations(range(D), 2):
        i, j = comb
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(D, dtype=np.int64)
                v[i] = s1
                v[j] = s2
                candidates.append(v)
                
    # Weight 4 vectors: (±1, ±1, ±1, ±1, 0, ...)
    # Count: C(11,4) * 16 = 330 * 16 = 5280
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Precompute Gram matrix
    # n_cand is ~5500. 5500x5500 matrix is manageable.
    G = candidates @ candidates.T
    
    # Conflict Condition:
    # ||u - v||^2 >= max(||u||^2, ||v||^2)
    # ||u||^2 + ||v||^2 - 2<u,v> >= max(||u||^2, ||v||^2)
    # Let M = max, m = min. M + m - 2<u,v> >= M  =>  m >= 2<u,v>  =>  <u,v> <= m/2
    
    min_norms_sq = np.minimum(norms_sq[:, None], norms_sq[None, :])
    limits = min_norms_sq / 2.0
    np.fill_diagonal(limits, np.inf)
    
    conflict_mask = G > limits
    np.fill_diagonal(conflict_mask, False)
    
    # Convert conflict mask to adjacency lists for MIS solver
    # We want to find the Maximum Independent Set in the graph defined by conflict_mask.
    # An independent set is a set of vertices no two of which are adjacent (i.e., no conflicts).
    
    # Since n_cand is ~5500, a general MIS solver might be slow if not careful.
    # However, the graph is sparse.
    # We will use a branch-and-bound approach with bitsets for efficiency.
    
    # Represent the graph using bitsets for fast neighbor queries.
    # Python integers can be used as bitsets.
    # n_cand = 5500 bits. This fits comfortably in Python's arbitrary precision integers.
    
    adj_bitsets = []
    for i in range(n_cand):
        neighbors = np.nonzero(conflict_mask[i])[0]
        # Create a bitmask
        mask = 0
        for n in neighbors:
            mask |= (1 << n)
        adj_bitsets.append(mask)
        
    # Initial solution: All weight-2 vectors are mutually compatible (as derived in previous thought).
    # Let's verify this again to ensure a strong starting point.
    # Weight 2 vectors have norm^2 = 2.
    # Distinct weight 2 vectors have dot product <= 1.
    # Limit = 2/2 = 1.
    # So dot <= 1 is allowed.
    # Thus, all distinct weight 2 vectors form an independent set.
    
    w2_count = 220
    initial_set_mask = 0
    for i in range(w2_count):
        initial_set_mask |= (1 << i)
        
    best_size = w2_count
    best_mask = initial_set_mask
    
    # Branch and Bound for MIS
    # We will try to add more vertices to the initial set, or replace vertices to allow more.
    # A standard recursive MIS solver on 5500 nodes is too slow if naive.
    # We use a greedy + local search hybrid or a specific B&B heuristic.
    
    # Given the time constraint, a pure B&B on 5500 nodes is risky.
    # Let's use a "Greedy with Bitset Pruning" approach which is very fast.
    # 1. Start with empty set.
    # 2. Repeatedly pick a vertex with minimum degree in the remaining graph? 
    #    Or maximum degree? For MIS, picking min degree is a common heuristic (Tomita).
    #    However, we know W2 is a large independent set.
    #    Let's try to extend the W2 set with W4 vectors using a fast bitset-based greedy.
    
    # State: 
    # available_mask: vertices not yet decided
    # current_set_mask: vertices in current independent set
    # 
    # We will perform a search that prioritizes adding W4 vectors to the W2 core.
    
    # To be safe and fast, we will use a deterministic greedy algorithm with bitset operations
    # which is O(N^2 / word_size) effectively.
    
    # Let's try a different structural approach: 
    # Since the problem is symmetric, we can fix the first few vectors to break symmetry?
    # No, let's stick to the efficient bitset greedy.
    
    # Reset state for a fresh greedy attempt that might beat the "W2 + Greedy W4" approach.
    # Heuristic: Pick vertex with highest "potential" (low conflict with existing set).
    
    # We'll run multiple passes with different ordering heuristics.
    
    def greedy_mis(order):
        """
        Greedy MIS construction given a fixed order of vertices to consider.
        """
        in_set = np.zeros(n_cand, dtype=bool)
        current_mask = 0
        count = 0
        
        # Precompute conflict masks for fast checking
        # We maintain a bitmask of 'blocked' vertices.
        # If a vertex is in the set, all its neighbors are blocked.
        
        blocked_mask = 0
        
        for v in order:
            if in_set[v]:
                continue
            # Check if v is blocked
            if (blocked_mask >> v) & 1:
                continue
            
            # Add v to set
            in_set[v] = True
            count += 1
            current_mask |= (1 << v)
            
            # Block neighbors
            blocked_mask |= adj_bitsets[v]
            # Also block v itself (it's in set, so it's "used")
            blocked_mask |= (1 << v)
            
        return count, current_mask

    # Heuristic 1: Standard Lexicographic (W2 then W4)
    order1 = list(range(n_cand))
    c1, m1 = greedy_mis(order1)
    
    # Heuristic 2: Sort by degree (ascending) - Tomita's heuristic
    # Degree of a vertex is the number of its neighbors.
    degrees = np.array([bin(adj_bitsets[i]).count('1') for i in range(n_cand)])
    order2 = list(np.argsort(degrees, kind='mergesort'))
    c2, m2 = greedy_mis(order2)
    
    # Heuristic 3: Sort by degree (descending)
    order3 = list(np.argsort(degades, kind='mergesort'))
    c3, m3 = greedy_mis(order3)
    
    # Heuristic 4: Randomized restarts with fixed seed for determinism
    # We can't do many random restarts, but a few might help.
    # Let's try 10 random orders.
    rng = np.random.default_rng(42)
    best_c = max(c1, c2, c3)
    best_m = m1 if best_c == c1 else (m2 if best_c == c2 else m3)
    
    for _ in range(20):
        order_rand = list(rng.permutation(n_cand))
        cr, mr = greedy_mis(order_rand)
        if cr > best_c:
            best_c = cr
            best_m = mr
            
    # Check time
    if time.time() > deadline - 10:
        # Return best found so far
        final_indices = np.nonzero((best_m & ((1 << n_cand) - 1)) != 0)[0] # This is wrong, need to extract indices from bitmask
        # Correct extraction:
        indices = []
        temp_m = best_m
        i = 0
        while temp_m:
            if temp_m & 1:
                indices.append(i)
            temp_m >>= 1
            i += 1
            if i >= n_cand:
                break
        return candidates[indices]
        
    # Local Search Improvement
    # Try to swap: remove 1 vertex, add 2?
    # This is complex with bitsets. 
    # Let's try a simple "Add if possible" pass on the best solution.
    
    # Convert best_m to boolean array for easier manipulation
    in_set = np.zeros(n_cand, dtype=bool)
    for i in range(n_cand):
        if (best_m >> i) & 1:
            in_set[i] = True
            
    # Recalculate blocked mask from in_set
    blocked_mask = 0
    for i in range(n_cand):
        if in_set[i]:
            blocked_mask |= adj_bitsets[i]
            blocked_mask |= (1 << i)
            
    # Try to add any vertex not in set and not blocked
    improved = True
    while improved:
        improved = False
        for i in range(n_cand):
            if not in_set[i] and not ((blocked_mask >> i) & 1):
                in_set[i] = True
                blocked_mask |= adj_bitsets[i]
                blocked_mask |= (1 << i)
                best_c += 1
                improved = True
                if time.time() > deadline - 5:
                    break
        if time.time() > deadline - 5:
            break

    # Final extraction
    final_indices = np.nonzero(in_set)[0]
    return candidates[final_indices]

# EVOLVE-BLOCK-END
