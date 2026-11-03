"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Generate candidates
    # wt-1: 11 * 2 = 22
    # wt-2: C(11,2) * 4 = 55 * 4 = 220
    # wt-3: C(11,3) * 8 = 165 * 8 = 1320
    # Total candidates for MIS: 220 + 1320 = 1540
    
    candidates = []
    indices = []
    
    # Generate wt-2
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
                    indices.append((i, j, si, sj))
    
    # Generate wt-3
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
                            indices.append((i, j, k, si, sj, sk))
                            
    N = len(candidates)
    
    # Build conflict matrix
    # Conflict if dot product >= 2
    # We can compute dot products efficiently
    # But N=1540, N^2 = 2.3M. 
    # Using numpy:
    V = np.array(candidates, dtype=np.int8)
    # Dot product matrix
    # We only need upper triangle or full.
    # dot = V @ V.T
    # This might be slow in pure Python loop, but NumPy is fast.
    # 1540x11 @ 11x1540 -> 1540x1540.
    # This is very fast.
    dot_prod = V @ V.T
    
    # Conflicts are where dot_prod >= 2
    # Note: Diagonal is norm^2 (4 or 9). We ignore self-conflict in MIS logic usually, 
    # but we must not pick the same vector twice.
    # We want to pick a set S such that for all i, j in S, i!=j, dot_prod[i,j] <= 1.
    
    # Greedy initialization
    # Random order
    order = np.random.permutation(N)
    selected = np.zeros(N, dtype=bool)
    count = 0
    
    for idx in order:
        if time.time() > deadline:
            break
        # Check if idx conflicts with any already selected
        # selected_indices = np.where(selected)[0]
        # If any dot_prod[idx, selected_indices] >= 2, skip
        # To be faster:
        # conflicts = dot_prod[idx, selected]
        # if np.any(conflicts >= 2):
        #    continue
        # else:
        #    selected[idx] = True
        #    count += 1
        
        # Vectorized check:
        # We can maintain a "conflict count" array or just check.
        # Since N is small, checking against selected is O(N) per step, total O(N^2).
        # 1540^2 ~ 2.3M ops. Very fast.
        
        # Actually, we can just check:
        # if np.any(dot_prod[idx, selected] >= 2):
        #     continue
        # else:
        #     selected[idx] = True
        #     count += 1
        
        # Let's do it.
        if not np.any(dot_prod[idx, selected] >= 2):
            selected[idx] = True
            count += 1
            
    # Local Search
    # Try to add a random unselected vector by removing a conflicting one.
    # Or swap.
    # Move: Pick random u not in S. If u conflicts with S, pick random v in S that conflicts with u.
    # Remove v, add u.
    # Accept if size increases (always 1-1 swap so size same).
    # To improve, we need to accept swaps that allow future additions?
    # Standard hill climbing on independent set size is tricky because size doesn't change in 1-1 swap.
    # Instead, we can do:
    # 1. Try to add a vector. If blocked, remove one blocker.
    # This is what I described. It keeps size same.
    # To increase size, we need to find a vector that can be added WITHOUT removing anyone.
    # So:
    # Loop:
    #   Find a random u not in S.
    #   If u is compatible with S, add it. (Size increases)
    #   Else, pick a random v in S that conflicts with u.
    #   Remove v. Add u. (Size same, but structure changes)
    #   Repeat.
    
    # We can also try "2-opt" style swaps?
    # For now, 1-1 swaps + greedy addition is good.
    
    # To speed up, we can precompute for each u, the list of conflicts.
    # But checking against `selected` is fast enough.
    
    # Let's run local search for a few seconds.
    while time.time() < deadline - 5.0:
        # Try to add
        # Pick random unselected
        unselected = np.where(~selected)[0]
        if len(unselected) == 0:
            break
        u = unselected[np.random.randint(len(unselected))]
        
        # Check if u can be added
        if not np.any(dot_prod[u, selected] >= 2):
            selected[u] = True
            count += 1
        else:
            # Find conflicts
            conflicts = np.where(dot_prod[u, selected] >= 2)[0]
            if len(conflicts) > 0:
                # Pick one to remove
                v = conflicts[np.random.randint(len(conflicts))]
                # Swap
                selected[v] = False
                selected[u] = True
                # count stays same
    
    # Add wt-1 vectors
    # They are compatible with all wt-2 and wt-3 (dot product <= 1)
    # And compatible with each other (dot product 0 or 1)
    # So we just append them.
    
    final_vectors = []
    for i in range(N):
        if selected[i]:
            final_vectors.append(candidates[i])
            
    # Add wt-1
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            final_vectors.append(v)
            
    return np.array(final_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
