import numpy as np
import itertools
import time
import sys
import heapq

D = 11

def generate_vectors(weight):
    """Generate all vectors of given weight in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), weight):
        for signs in itertools.product([1, -1], repeat=weight):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def build_adjacency(vectors, threshold):
    """
    Build adjacency list for vectors where dot product >= threshold.
    Returns a list of lists of indices.
    """
    N = len(vectors)
    # Compute dot products in chunks to save memory?
    # N=5280, N^2 = 27M ints. 27M * 4 bytes = 100MB. OK.
    dots = vectors @ vectors.T
    # We want indices j where dots[i][j] >= threshold and i != j
    adj = [[] for _ in range(N)]
    for i in range(N):
        # Find j where dots[i][j] >= threshold
        # Use np.where for speed
        js = np.where(dots[i] >= threshold)[0]
        for j in js:
            if j != i:
                adj[i].append(j)
    return adj

def greedy_mis_fast(adj, N, deadline, rng):
    """
    Fast greedy MIS using sets.
    """
    alive = set(range(N))
    # Calculate degrees
    degrees = {i: len(adj[i]) for i in range(N)}
    
    selected = []
    
    while alive:
        if time.time() > deadline:
            break
            
        # Find min degree
        min_deg = min(degrees[i] for i in alive)
        candidates = [i for i in alive if degrees[i] == min_deg]
        
        # Random tie break
        u = candidates[rng.integers(len(candidates))]
        selected.append(u)
        
        # Remove u and its neighbors from alive
        to_remove = {u}
        for v in adj[u]:
            if v in alive:
                to_remove.add(v)
                
        for v in to_remove:
            alive.discard(v)
            
        # Update degrees for remaining alive nodes
        # For each v in alive, if u was a neighbor, degree decreases
        # adj[u] contains neighbors of u.
        # So for each v in adj[u], if v is still in alive, degrees[v] -= 1
        for v in adj[u]:
            if v in alive:
                degrees[v] -= 1
                
    return selected

def construct():
    t0 = time.time()
    deadline = t0 + 115
    
    rng = np.random.default_rng(42)
    
    # 1. Generate W4 vectors
    w4_vectors = generate_vectors(4)
    N4 = len(w4_vectors)
    
    # 2. Build adjacency for W4
    # Conflict if dot >= 3
    adj4 = build_adjacency(w4_vectors, 3)
    
    # 3. Solve MIS on W4
    best_set4 = []
    # Run multiple greedy restarts
    # Since the graph is sparse, greedy is fast.
    for i in range(100): # Try many restarts
        if time.time() > deadline - 20: break
        cur_set = greedy_mis_fast(adj4, N4, deadline - 20, rng)
        if len(cur_set) > len(best_set4):
            best_set4 = cur_set
            
    selected_w4 = w4_vectors[best_set4]
    
    # 4. Add W3 vectors
    w3_vectors = generate_vectors(3)
    N3 = len(w3_vectors)
    
    if len(selected_w4) > 0:
        dots34 = w3_vectors @ selected_w4.T
        # Condition: u (W3) . v (W4) <= 1
        # If dot >= 2, conflict
        compatible3 = (dots34 < 2).all(axis=1)
    else:
        compatible3 = np.ones(N3, dtype=bool)
        
    w3_candidates = w3_vectors[compatible3]
    N3c = len(w3_candidates)
    
    if N3c > 0:
        # Build conflict graph among W3 candidates
        # Condition: u (W3) . v (W3) <= 1
        # Conflict if dot >= 2
        adj3 = build_adjacency(w3_candidates, 2)
        
        best_set3 = []
        for i in range(20):
            if time.time() > deadline - 5: break
            cur_set = greedy_mis_fast(adj3, N3c, deadline - 5, rng)
            if len(cur_set) > len(best_set3):
                best_set3 = cur_set
                
        selected_w3 = w3_candidates[best_set3]
    else:
        selected_w3 = np.empty((0, D), dtype=np.int8)
        
    # 5. Add W2 vectors
    w2_vectors = generate_vectors(2)
    N2 = len(w2_vectors)
    
    # Condition with W4: u (W2) . v (W4) <= 1
    # Condition with W3: u (W2) . v (W3) <= 1
    # Conflict if dot >= 2
    if len(selected_w4) > 0:
        dots24 = w2_vectors @ selected_w4.T
        comp24 = (dots24 < 2).all(axis=1)
    else:
        comp24 = np.ones(N2, dtype=bool)
        
    if len(selected_w3) > 0:
        dots23 = w2_vectors @ selected_w3.T
        comp23 = (dots23 < 2).all(axis=1)
    else:
        comp23 = np.ones(N2, dtype=bool)
        
    compatible2 = comp24 & comp23
    w2_candidates = w2_vectors[compatible2]
    N2c = len(w2_candidates)
    
    if N2c > 0:
        # Build conflict graph among W2 candidates
        # Condition: u (W2) . v (W2) <= 0
        # Conflict if dot >= 1
        adj2 = build_adjacency(w2_candidates, 1)
        
        best_set2 = []
        for i in range(10):
            if time.time() > deadline - 1: break
            cur_set = greedy_mis_fast(adj2, N2c, deadline - 1, rng)
            if len(cur_set) > len(best_set2):
                best_set2 = cur_set
                
        selected_w2 = w2_candidates[best_set2]
    else:
        selected_w2 = np.empty((0, D), dtype=np.int8)
        
    # Combine results
    all_vectors = np.vstack([selected_w4, selected_w3, selected_w2])
    
    if len(all_vectors) == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    return all_vectors.astype(np.int64)
