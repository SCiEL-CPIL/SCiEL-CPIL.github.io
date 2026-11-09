import numpy as np
import time
from itertools import product

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit (5s buffer)
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    e_count = len(edges)
    
    # Identify edge types:
    # For prism graph with m-cycles:
    # - Top cycle: (i, i+1 mod m) for i in 0..m-1
    # - Bottom cycle: (m+i, m+(i+1 mod m)) for i in 0..m-1  
    # - Matching: (i, i+m) for i in 0..m-1
    
    # Build signed adjacency matrix for quick spectral radius computation
    def compute_radius(signs):
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    # Strategy: Use the structure of prism graphs
    # The prism graph has a natural decomposition:
    # - 2m cycle vertices (two m-cycles)
    # - m matching edges
    
    # Key insight: For 3-regular bipartite lifts to be connected,
    # we need careful sign patterns.
    
    # Approach: Start with all +1 (gives 2 disconnected copies), 
    # then flip signs to make it connected while minimizing spectral radius.
    
    # Better approach: Use the known structure
    # For prism C_m x K_2, the edges can be categorized as:
    # - Edges in the two cycles (top and bottom)
    # - Matching edges
    
    # Let's identify which edges are which
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.append(i)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(i)
        else:
            matching_edges.append(i)
    
    # For the lift to be connected and 3-regular, we need a specific pattern.
    # The signed graph A_s where A_s[u,v] = sign of edge (u,v)
    # The lift is connected iff the associated covering is connected.
    
    # Key structural approach:
    # The prism graph has a 2-fold symmetry. We can try sign patterns
    # that respect or break this symmetry.
    
    # For small m, we can do exhaustive search over the cycle edges
    # and optimize matching edges.
    
    # Since m is at most 9, the number of edges is at most 27.
    # Let's use a smarter search:
    
    # 1. First, find a valid connected sign assignment
    # 2. Then use local search to minimize spectral radius
    
    # Find initial valid assignment:
    # All -1 on matching edges, +1 on cycles gives one pattern
    # Let's try several structural patterns
    
    candidate_signs_list = []
    
    # Pattern 1: All +1
    s = np.ones(e_count, dtype=int)
    candidate_signs_list.append(s)
    
    # Pattern 2: Alternating on matching
    s = np.ones(e_count, dtype=int)
    for i, edge_idx in enumerate(matching_edges):
        if i % 2 == 0:
            s[edge_idx] = -1
    candidate_signs_list.append(s)
    
    # Pattern 3: All matching -1
    s = np.ones(e_count, dtype=int)
    for edge_idx in matching_edges:
        s[edge_idx] = -1
    candidate_signs_list.append(s)
    
    # Pattern 4: All -1
    s = np.ones(e_count, dtype=int)
    s[:] = -1
    candidate_signs_list.append(s)
    
    # Pattern 5: Top cycle alternating, rest +1
    s = np.ones(e_count, dtype=int)
    for i, edge_idx in enumerate(top_cycle_edges):
        if i % 2 == 0:
            s[edge_idx] = -1
    candidate_signs_list.append(s)
    
    # Evaluate initial candidates
    best_radius = float('inf')
    best_signs = np.ones(e_count, dtype=int)
    
    for s in candidate_signs_list:
        if time.time() > deadline:
            break
        try:
            r = compute_radius(s)
            if r < best_radius:
                best_radius = r
                best_signs = s.copy()
        except Exception:
            pass
    
    # Local search: flip one edge at a time
    current_signs = best_signs.copy()
    current_radius = best_radius
    no_improve_count = 0
    
    while time.time() < deadline and no_improve_count < 50:
        improved = False
        
        for edge_idx in range(e_count):
            if time.time() > deadline:
                break
            
            # Try flipping this edge
            new_signs = current_signs.copy()
            new_signs[edge_idx] = -new_signs[edge_idx]
            
            try:
                r = compute_radius(new_signs)
                if r < current_radius - 1e-10:
                    current_signs = new_signs
                    current_radius = r
                    best_signs = new_signs.copy()
                    best_radius = r
                    improved = True
                    no_improve_count = 0
                    break
            except Exception:
                pass
        
        if not improved:
            no_improve_count += 1
            # Try 2-flip moves
            if no_improve_count % 5 == 0:
                for i in range(e_count):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, e_count):
                        if time.time() > deadline:
                            break
                        new_signs = current_signs.copy()
                        new_signs[i] = -new_signs[i]
                        new_signs[j] = -new_signs[j]
                        
                        try:
                            r = compute_radius(new_signs)
                            if r < current_radius - 1e-10:
                                current_signs = new_signs
                                current_radius = r
                                best_signs = new_signs.copy()
                                best_radius = r
                                improved = True
                                no_improve_count = 0
                                break
                        except Exception:
                            pass
                    if improved:
                        break
    
    return best_signs
