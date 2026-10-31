"""Constructing a kissing configuration in 11 dimensions using a graph-based greedy selection on a mixed-norm candidate set."""

import itertools
import numpy as np

D = 11

def construct():
    # Candidate generation
    # Norm squared 4 vectors
    # Type A: 1 entry +/-2
    # Type B: 4 entries +/-1
    
    # Norm squared 8 vectors
    # Type C: 2 entries +/-2
    # Type D: 8 entries +/-1 (Note: 11-8=3 zeros, so choose 8 indices)
    
    candidates = []
    
    # Type A: Norm^2 = 4
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            candidates.append(v)
            
    # Type B: Norm^2 = 4
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for k, idx in enumerate(idxs):
                v[idx] = signs[k]
            candidates.append(v)
            
    # Type C: Norm^2 = 8
    for idxs in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            for k, idx in enumerate(idxs):
                v[idx] = 2 * signs[k]
            candidates.append(v)
            
    # Type D: Norm^2 = 8
    # Choose 8 indices out of 11
    for idxs in itertools.combinations(range(D), 8):
        for signs in itertools.product([1, -1], repeat=8):
            v = np.zeros(D, dtype=np.int64)
            for k, idx in enumerate(idxs):
                v[idx] = signs[k]
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Compatibility Check
    # We want to find a subset S such that for all u, v in S:
    # ||u - v||^2 >= max(||u||^2, ||v||^2)
    # ||u||^2 + ||v||^2 - 2 u.v >= max(||u||^2, ||v||^2)
    # If ||u||^2 == ||v||^2 = R^2: R^2 - 2 u.v >= 0 => u.v <= R^2/2
    # If ||u||^2 < ||v||^2: ||u||^2 + ||v||^2 - 2 u.v >= ||v||^2 => ||u||^2 >= 2 u.v => u.v <= ||u||^2 / 2
    
    # General condition: 2 * u.v <= min(||u||^2, ||v||^2)
    
    # We use a greedy algorithm to build a clique in the compatibility graph.
    # To improve the greedy choice, we can order candidates or use a heuristic.
    # A simple improvement: Process candidates in an order that might lead to larger cliques.
    # Often, vectors with "smaller" norms are more compatible with many things? 
    # Or we can just iterate.
    
    selected_indices = []
    selected_vectors = []
    
    # To make the greedy step more effective, we can sort candidates.
    # However, the order in the list above is fixed. 
    # Let's try to be smarter: 
    # The condition is u.v <= min(||u||^2, ||v||^2)/2.
    
    # We will iterate through candidates. 
    # To potentially get a better result than simple sequential greedy, 
    # we can try multiple random orderings or a degree-based heuristic.
    # Given time limits, let's stick to a deterministic but slightly improved greedy:
    # We'll just do the standard greedy, but the candidate set is much larger and mixed.
    
    # Optimization: Pre-compute norms squared
    # Vectorized check for each new candidate against all selected
    
    for i, v in enumerate(candidates):
        if not selected_vectors:
            selected_vectors.append(v)
            continue
            
        sel_arr = np.array(selected_vectors)
        sel_norms_sq = np.sum(sel_arr**2, axis=1)
        v_norm_sq = norms_sq[i]
        
        dots = sel_arr @ v
        
        # Condition: 2 * dots <= min(sel_norms_sq, v_norm_sq)
        # min_sel_v = np.minimum(sel_norms_sq, v_norm_sq)
        # if np.all(2 * dots <= min_sel_v):
        
        # Since v_norm_sq is constant for this iteration:
        # We need: 2*dots <= sel_norms_sq AND 2*dots <= v_norm_sq
        
        # Check 1: 2 * dots <= v_norm_sq
        # Check 2: 2 * dots <= sel_norms_sq
        
        if np.all(2 * dots <= v_norm_sq) and np.all(2 * dots <= sel_norms_sq):
            selected_vectors.append(v)
            
    return np.array(selected_vectors, dtype=np.int64)
