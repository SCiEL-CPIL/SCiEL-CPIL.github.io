"""Spectral graph lifts: exact search via gauge invariance."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s (evaluator kills at 90s, but be safe)
    
    m = graph['n'] // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Identify edge types:
    # - cycle edges on bottom cycle: (i, i+1) mod m for i in 0..m-1
    # - cycle edges on top cycle: (m+i, m+i+1) mod m for i in 0..m-1
    # - matching edges: (i, i+m) for i in 0..m-1
    #
    # Gauge invariance: we can fix a spanning tree to +1.
    # A spanning tree of the prism graph (2m vertices, 3m edges) has 2m-1 edges.
    # The remaining m+1 edges are "chords" whose signs we can vary.
    #
    # Let's construct a specific spanning tree:
    # - All m matching edges (i, i+m)
    # - m-1 bottom cycle edges (0,1), (1,2), ..., (m-2, m-1)
    # - m-1 top cycle edges (m, m+1), (m+1, m+2), ..., (2m-2, 2m-1)
    # Total: m + (m-1) + (m-1) = 3m - 2 edges. But spanning tree needs 2m-1 edges.
    # That's too many. Let me reconsider.
    #
    # Prism graph has 2m vertices. Spanning tree has 2m-1 edges.
    # Total edges: 3m. So chords = 3m - (2m-1) = m+1 edges.
    #
    # Let's pick a spanning tree:
    # - All m matching edges (i, i+m): connects each bottom vertex to its top counterpart
    # - m-1 bottom cycle edges: (0,1), (1,2), ..., (m-2, m-1)
    # - m-1 top cycle edges: (m, m+1), (m+1, m+2), ..., (2m-2, 2m-1)
    # Total: m + (m-1) + (m-1) = 3m-2. Still too many.
    #
    # Wait, I need exactly 2m-1 edges. Let me be more careful.
    # 
    # Actually, let's just pick:
    # - All m matching edges
    # - m-1 bottom cycle edges (all except one, say (m-1, 0))
    # - 0 top cycle edges? No, that doesn't connect the top cycle.
    #
    # Better approach: 
    # - All m matching edges (i, i+m)
    # - m-1 bottom cycle edges: (0,1), (1,2), ..., (m-2, m-1)
    # - m-1 top cycle edges: (m, m+1), (m+1, m+2), ..., (2m-2, 2m-1)
    # That's m + (m-1) + (m-1) = 3m-2 edges for 2m vertices. Way too many.
    #
    # I confused myself. A spanning tree on 2m vertices has exactly 2m-1 edges.
    # Let me construct one properly:
    # - Start with all m matching edges. This gives m components, each being an isolated pair.
    # - Add bottom cycle edges to connect bottom vertices: need m-1 edges to make bottom a tree.
    # - Add top cycle edges to connect top vertices: need m-1 edges to make top a tree.
    # Total: m + (m-1) + (m-1) = 3m-2. But we only need 2m-1.
    #
    # The issue: matching edges already connect bottom i to top i. So if I add bottom cycle edges
    # to make bottom connected, and top cycle edges to make top connected, the whole graph is connected.
    # But that's 3m-2 edges, which is more than 2m-1.
    #
    # Actually wait - if I have all matching edges, and I add m-1 bottom cycle edges to connect all
    # bottom vertices, then all top vertices are already connected through the matching edges to the
    # connected bottom vertices. So I don't need ANY top cycle edges!
    # 
    # Spanning tree:
    # - All m matching edges (i, i+m)
    # - m-1 bottom cycle edges: (0,1), (1,2), ..., (m-2, m-1)
    # Total: m + (m-1) = 2m-1 edges. Perfect!
    #
    # The remaining m+1 edges (chords) are:
    # - 1 bottom cycle edge: (m-1, 0)
    # - All m top cycle edges: (m, m+1), (m+1, m+2), ..., (2m-2, 2m-1), and (2m-1, m)
    
    # Map each edge to its index
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = idx
        edge_to_idx[(v, u)] = idx
    
    # Identify which edges are in the spanning tree (fixed to +1)
    tree_edges = set()
    # Matching edges
    for i in range(m):
        tree_edges.add((i, i + m))
        tree_edges.add((i + m, i))
    # Bottom cycle edges except (m-1, 0)
    for i in range(m - 1):
        tree_edges.add((i, i + 1))
        tree_edges.add((i + 1, i))
    
    # Chord edges: all edges not in the spanning tree
    chord_indices = []
    for idx, (u, v) in enumerate(edges):
        if (u, v) not in tree_edges:
            chord_indices.append(idx)
    
    num_chords = len(chord_indices)
    # Should be m+1
    
    # Initialize all signs to +1
    best_signs = None
    best_radius = float('inf')
    
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
            from autoresearch_suite.science_tcs_expansion import check_lift
            radius, details = check_lift(graph, signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        except Exception:
            pass
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(num_edges, dtype=np.int8)
    
    return best_signs
# EVOLVE-BLOCK-END
