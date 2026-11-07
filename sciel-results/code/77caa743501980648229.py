"""Spectral graph lifts: exact search via gauge invariance with local search polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.science_tcs_expansion import check_lift

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s
    
    m = graph['n'] // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Identify edge types and build spanning tree for gauge fixing
    # Spanning tree: all m matching edges + m-1 bottom cycle edges (0,1),...,(m-2,m-1)
    # Chords: 1 bottom edge (m-1,0) + m top cycle edges
    
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = idx
        edge_to_idx[(v, u)] = idx
    
    # Tree edges (fixed to +1)
    tree_edges = set()
    for i in range(m):
        tree_edges.add((i, i + m))
        tree_edges.add((i + m, i))
    for i in range(m - 1):
        tree_edges.add((i, i + 1))
        tree_edges.add((i + 1, i))
    
    # Chord edges
    chord_indices = []
    for idx, (u, v) in enumerate(edges):
        if (u, v) not in tree_edges:
            chord_indices.append(idx)
    
    num_chords = len(chord_indices)
    
    best_signs = None
    best_radius = float('inf')
    
    # Pre-build base adjacency (all +1) for speed
    base_adj = np.ones((2 * m, 2 * m))
    for u, v in edges:
        base_adj[u, v] = 1.0
        base_adj[v, u] = 1.0
    
    # Enumerate all 2^num_chords assignments
    total_assignments = 1 << num_chords
    
    for assignment in range(total_assignments):
        if time.time() > deadline:
            break
        
        # Build signs vector
        signs = np.ones(num_edges, dtype=np.int8)
        for j, eidx in enumerate(chord_indices):
            if assignment & (1 << j):
                signs[eidx] = -1
        
        # Check connectivity and compute spectral radius
        try:
            radius, details = check_lift(graph, signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        except Exception:
            pass
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(num_edges, dtype=np.int8)
    
    # Local search refinement: try flipping individual chord edges
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for j in range(num_chords):
                trial_signs = best_signs.copy()
                trial_signs[chord_indices[j]] = -trial_signs[chord_indices[j]]
                try:
                    radius, details = check_lift(graph, trial_signs)
                    if radius < best_radius - 1e-12:
                        best_radius = radius
                        best_signs = trial_signs
                        improved = True
                except Exception:
                    pass
    
    return best_signs
# EVOLVE-BLOCK-END
