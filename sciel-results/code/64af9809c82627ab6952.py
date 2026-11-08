"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    # Build candidate pool: all integer vectors with L2 norm <= some bound
    # We'll use weight-4 vectors (4 non-zero entries of ±1)
    # Total: C(11,4) * 2^4 = 330 * 16 = 5280
    
    candidates = []
    for idx in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = [0] * D
            for i, s in zip(idx, signs):
                v[i] = s
            candidates.append(tuple(v))
    
    N = len(candidates)  # 5280
    
    # Compute norms
    norms = np.array([sum(x*x for x in c) for c in candidates], dtype=np.float64)
    
    # For the constraint: min pairwise distance >= max norm
    # If we normalize to norm 2, then we need min pairwise distance >= 2
    # Original: min ||v_i - v_j|| >= max ||v_i||
    # After scaling: if max_norm = sqrt(max_norm_sq), we need ||v_i - v_j|| >= max_norm
    # Equivalently: ||v_i - v_j||^2 >= max_norm_sq
    
    max_norm_sq = float(np.max(norms))
    
    # We want to find a large subset S such that for all i,j in S, i≠j:
    # ||v_i - v_j||^2 >= max_norm_sq
    # Note: ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i·v_j
    # Since all have the same norm (weight-4 vectors all have norm^2 = 4):
    # ||v_i - v_j||^2 = 4 + 4 - 2*v_i·v_j = 8 - 2*v_i·v_j
    # We need: 8 - 2*v_i·v_j >= 4  =>  v_i·v_j <= 2
    
    # So we need an independent set in the graph where edges connect vectors with dot product > 2
    
    # Build conflict graph: edge between i,j if dot(i,j) > 2
    # For weight-4 vectors, dot product can be: -4,-3,-2,-1,0,1,2,3,4
    # Conflict if dot >= 3
    
    # Use greedy: sort by degree, pick lowest degree first
    # But first, let's build the conflict list efficiently
    
    # Dot product computation for all pairs would be O(N^2) = ~27M, might be slow
    # Let's use a smarter approach: group by which coordinates are non-zero
    
    # Actually, let's just build the conflict graph using numpy for speed
    # But N=5280, N^2/2 = ~14M pairs, each dot is 11-dim... that's feasible
    
    # Convert to numpy array
    V = np.array(candidates, dtype=np.float64)
    
    # Compute all pairwise dot products (memory: 5280x5280 floats = ~220MB, too much)
    # Instead, use chunked computation
    
    # Let's use a greedy approach with a different strategy:
    # Randomized greedy with local search
    
    # First, build adjacency using efficient method
    # For each pair, check if dot > 2
    
    # Use numpy matrix multiply in chunks
    chunk_size = 500
    conflicts = [set() for _ in range(N)]
    
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        if time.time() > deadline - 30:
            break
        V_chunk = V[start:end]
        # Compute dot products of chunk with all
        dots = V_chunk @ V.T  # shape (chunk_size, N)
        for local_i in range(end - start):
            i = start + local_i
            # Find j where dot > 2 and j != i
            mask = dots[local_i] > 2
            mask[i] = False  # exclude self
            for j in np.where(mask)[0]:
                conflicts[i].add(int(j))
                conflicts[int(j)].add(i)
    
    # Now do greedy independent set: sort by degree, pick lowest degree unprocessed node
    if time.time() > deadline - 10:
        # Fallback: just return seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    degrees = [len(c) for c in conflicts]
    order = sorted(range(N), key=lambda x: degrees[x])
    
    in_set = [False] * N
    result_indices = []
    
    for idx in order:
        if in_set[idx]:
            continue
        # Check if all neighbors are already excluded or in set
        # Actually, for greedy independent set: pick idx, remove idx and all its neighbors
        # Better: standard greedy - pick idx if no neighbor is in set
        # Let's do: pick idx if none of its neighbors are already in result
        can_add = True
        for nb in conflicts[idx]:
            if in_set[nb]:
                can_add = False
                break
        if can_add:
            in_set[idx] = True
            result_indices.append(idx)
    
    if not result_indices:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result_vectors = np.array([candidates[i] for i in result_indices], dtype=np.int64)
    
    return result_vectors
# EVOLVE-BLOCK-END
