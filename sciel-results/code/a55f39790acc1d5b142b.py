import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)
    
    # Deadline guard: 55 seconds to be safe
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute edge mapping for fast access
    # We need to map (u, v) to index
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[frozenset((u, v))] = i
        
    # Helper to get sign from array given u, v
    def get_sign(signs_arr, u, v):
        idx = edge_index[frozenset((u, v))]
        return signs_arr[idx]

    # Helper to build the sign array from a dictionary or pattern
    def build_signs_from_dict(sign_dict):
        signs = np.ones(E, dtype=np.int8)
        for i, (u, v) in enumerate(edges):
            signs[i] = sign_dict[frozenset((u, v))]
        return signs

    # 1. Structured Enumeration
    # Define the spanning tree: matching (i, i+m) and bottom path (i, i+1) for i in 0..m-2
    tree_edges = set()
    for i in range(m):
        tree_edges.add(frozenset((i, m+i)))
    for i in range(m-1):
        tree_edges.add(frozenset((i, i+1)))
    
    chord_edges = []
    chord_edges.append(frozenset((m-1, 0)))
    for i in range(m):
        chord_edges.append(frozenset((m+i, m+(i+1)%m)))
        
    num_chords = len(chord_edges)
    best_signs = None
    best_radius = float('inf')
    
    # Enumerate 2^(m+1) patterns
    # For m=4,5,7,9, this is 2^5, 2^6, 2^8, 2^10 = 32, 64, 256, 1024. Very fast.
    for pattern in range(2 ** num_chords):
        if time.time() > deadline:
            break
            
        sign_dict = {}
        for e in tree_edges:
            sign_dict[e] = 1
        for j, e in enumerate(chord_edges):
            sign_dict[e] = 1 if (pattern >> j) & 1 else -1
            
        signs_arr = build_signs_from_dict(sign_dict)
        
        # Check validity and radius
        try:
            from autoresearch_suite.science_tcs_expansion import check_lift
            radius, _ = check_lift(graph, signs_arr)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs_arr.copy()
        except Exception:
            continue

    if best_signs is None:
        # Fallback if structured enumeration failed (should not happen)
        best_signs = np.ones(E, dtype=np.int8)
        try:
            from autoresearch_suite.science_tcs_expansion import check_lift
            radius, _ = check_lift(graph, best_signs)
            best_radius = radius
        except Exception:
            # If even all +1 is invalid (unlikely for prism), try all -1
            best_signs = -np.ones(E, dtype=np.int8)
            try:
                from autoresearch_suite.science_tcs_expansion import check_lift
                radius, _ = check_lift(graph, best_signs)
                best_radius = radius
            except Exception:
                # Last resort, return all +1 even if invalid? No, must be valid.
                # But the task implies a valid solution exists.
                return np.ones(E, dtype=np.int8)

    # 2. Local Search
    # Start from the best structured solution
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    # Precompute adjacency for potential faster checks if needed, 
    # but check_lift is the oracle.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(E):
            if time.time() > deadline:
                break
                
            # Flip edge i
            old_sign = current_signs[i]
            current_signs[i] = -old_sign
            
            try:
                from autoresearch_suite.science_tcs_expansion import check_lift
                new_radius, _ = check_lift(graph, current_signs)
                if new_radius < current_radius:
                    current_radius = new_radius
                    best_radius = new_radius
                    best_signs = current_signs.copy()
                    improved = True
                else:
                    # Revert
                    current_signs[i] = old_sign
            except Exception:
                # Invalid lift, revert
                current_signs[i] = old_sign

    # 3. Multi-start Local Search if time permits
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(3)
        # Try a few random starts
        for _ in range(20):
            if time.time() > deadline:
                break
            
            # Generate random signs
            rand_signs = rng.choice([-1, 1], size=E).astype(np.int8)
            
            # Quick validity check
            try:
                from autoresearch_suite.science_tcs_expansion import check_lift
                r, _ = check_lift(graph, rand_signs)
            except Exception:
                continue
                
            if r >= best_radius:
                continue
                
            # If better, do local search from here
            current_signs = rand_signs.copy()
            current_radius = r
            best_radius = r
            best_signs = rand_signs.copy()
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(E):
                    if time.time() > deadline:
                        break
                    old_sign = current_signs[i]
                    current_signs[i] = -old_sign
                    try:
                        from autoresearch_suite.science_tcs_expansion import check_lift
                        new_radius, _ = check_lift(graph, current_signs)
                        if new_radius < current_radius:
                            current_radius = new_radius
                            best_radius = new_radius
                            best_signs = current_signs.copy()
                            improved = True
                        else:
                            current_signs[i] = old_sign
                    except Exception:
                        current_signs[i] = old_sign

    return best_signs
