"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # We are looking for vectors in Z^11.
    # The condition is: min_{i != j} ||v_i - v_j|| >= max_i ||v_i||
    # Let R = max_i ||v_i||. Then for all i != j, ||v_i - v_j|| >= R.
    # Squaring: ||v_i - v_j||^2 >= R^2.
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i . v_j.
    
    # Strategy:
    # 1. Generate a pool of candidate vectors.
    #    - Weight 2 vectors (norm^2 = 2): C(11,2)*4 = 220 vectors.
    #    - Weight 3 vectors (norm^2 = 3): C(11,3)*8 = 660 vectors.
    #    - Weight 4 vectors (norm^2 = 4): C(11,4)*16 = 3360 vectors.
    #    - Weight 1 vectors (norm^2 = 1): 22 vectors.
    
    # If we include weight 1 vectors, R^2 >= 1.
    # If we include weight 4 vectors, R^2 >= 4.
    # If R^2 = 4, we need ||v_i - v_j||^2 >= 4 for all pairs.
    
    # Let's focus on a homogeneous set of weight 4 vectors first, as they allow for larger kissing numbers in higher dimensions due to more "room" in the integer lattice relative to the radius.
    # Actually, the kissing number in dimension 11 is known to be at least 552 (Levenshtein & Rudakov, 1993? No, that's for 12? Let's check known bounds).
    # Known kissing numbers:
    # d=1: 2
    # d=2: 6
    # d=3: 12
    # d=4: 24
    # d=5: 40
    # d=6: 72
    # d=7: 126
    # d=8: 240
    # d=9: 460 (or 464?)
    # d=10: 830 (or 840?)
    # d=11: ?
    # d=12: 2304 (Leech lattice)
    
    # The Leech lattice construction often involves vectors of weight 4 in the E8 lattice or similar.
    # In R^11, we can use a subset of the Leech lattice or construct directly.
    
    # Let's try to find a large independent set in the graph where edges connect vectors with distance < R.
    # If we restrict to vectors of norm^2 = 4 (weight 4, entries +/-1), then R^2 = 4.
    # Condition: ||v_i - v_j||^2 >= 4.
    # ||v_i - v_j||^2 = 4 + 4 - 2 v_i . v_j = 8 - 2 v_i . v_j.
    # So 8 - 2 v_i . v_j >= 4  =>  4 >= 2 v_i . v_j  =>  v_i . v_j <= 2.
    
    # So we need to find a set of weight-4 vectors (entries +/-1) such that their pairwise dot products are <= 2.
    
    # Let's generate all weight-4 vectors in R^11.
    # Number of such vectors: C(11,4) * 2^4 = 330 * 16 = 5280.
    
    # We can use a greedy algorithm or a local search to find a large subset.
    
    # Generate all weight-4 vectors
    candidates = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int8)
    N = len(candidates)
    
    # Precompute dot products? N=5280, N^2/2 ~ 14 million. This is feasible in NumPy.
    # Let's compute the full dot product matrix M where M[i,j] = candidates[i] . candidates[j].
    # M = candidates @ candidates.T
    
    # We want to select a subset S such that for all i != j in S, M[i,j] <= 2.
    # This is equivalent to finding a large clique in the graph where edges exist if M[i,j] <= 2.
    # Or an independent set in the graph where edges exist if M[i,j] > 2.
    
    # Let's build the adjacency matrix for the "conflict" graph (where M[i,j] > 2).
    # We want to find a Maximum Independent Set (MIS) in this conflict graph.
    
    # Since N=5280, building the full NxN matrix takes ~22MB. This is fine.
    # However, finding MIS is NP-hard. We will use a greedy heuristic with random restarts and local improvement.
    
    # Conflict matrix: True if dot > 2
    dots = candidates @ candidates.T
    conflict = dots > 2
    np.fill_diagonal(conflict, False) # No self-conflict
    
    # Greedy MIS:
    # 1. Shuffle vertices.
    # 2. Iterate through shuffled list.
    # 3. If vertex is not connected to any already selected vertex, select it.
    
    best_set = []
    best_size = 0
    
    # To speed up, we can precompute the neighbors of each vertex in the conflict graph.
    # neighbors[i] = list of j such that conflict[i,j] is True.
    # This might be large, but let's try.
    
    # Alternative: Vectorized check.
    # Keep a boolean array `selected` of size N.
    # When we select a vertex v, we mark all its neighbors as "blocked".
    
    rng = np.random.default_rng(42)
    
    for restart in range(200): # Max 200 restarts
        if time.time() > deadline - 5:
            break
            
        order = rng.permutation(N)
        selected = np.zeros(N, dtype=bool)
        blocked = np.zeros(N, dtype=bool)
        current_set = []
        
        for v_idx in order:
            if blocked[v_idx]:
                continue
                
            # Select v_idx
            selected[v_idx] = True
            current_set.append(v_idx)
            
            # Block all neighbors of v_idx
            # neighbors are indices where conflict[v_idx, :] is True
            # We can update blocked using bitwise OR
            blocked |= conflict[v_idx]
            
            # Optimization: If we have selected enough, we can stop? No, we want max.
            
        size = len(current_set)
        if size > best_size:
            best_size = size
            best_set = current_set
            
            # Try local improvement (swap)
            # If we have time, try to improve the current set.
            # For each v in current_set, try to swap with some u not in set.
            # This is complex. Let's stick to multiple random restarts first.
            # If best_size > 500, we might be close to optimal.
            
            # Early termination if we hit a known high bound?
            # Kissing number in d=11 is at least 552?
            # If we reach 550, we can stop?
            if best_size >= 550:
                break

    # Convert indices back to vectors
    if best_set:
        result = candidates[best_set]
    else:
        # Fallback: just return a single vector or empty?
        # Must return non-empty if possible.
        result = candidates[:1]
        
    return np.array(result, dtype=np.int64)
# EVOLVE-BLOCK-END
