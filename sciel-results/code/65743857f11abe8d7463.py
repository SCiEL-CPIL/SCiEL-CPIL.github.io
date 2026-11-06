"""Optimized C4-free edge set construction for Q7."""
import time
import random

# Precompute all edges and their face relationships
# Each edge is identified by (min(u,v), max(u,v))
# For each edge, precompute the 3 edges that would complete a C4 with it

def _precompute():
    d = 7
    n = 1 << d
    edges = []
    edge_index = {}  # (u,v) -> index
    
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_index[(x, y)] = idx
    
    # For each edge, find the 3 edges that form a C4 with it
    # Edge (x,y) where y = x ^ (1<<i). The C4 is formed with another bit j.
    # The other 3 edges are: (x, x^(1<<j)), (y, y^(1<<j)), (x^(1<<j), y^(1<<j))
    edge_faces = [[] for _ in edges]
    
    for idx, (x, y) in enumerate(edges):
        # Find which bit differs
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            # Three other edges in the face
            e1 = (min(x, c), max(x, c))
            e2 = (min(y, e), max(y, e))
            e3 = (min(c, e), max(c, e))
            edge_faces[idx].extend([edge_index[e1], edge_index[e2], edge_index[e3]])
    
    return edges, edge_faces

# Cache precomputation
_edges = None
_edge_faces = None

def _get_precomputed():
    global _edges, _edge_faces
    if _edges is None:
        _edges, _edge_faces = _precompute()
    return _edges, _edge_faces

def _greedy_randomized(order, deadline):
    """Run greedy with given edge order, return list of selected edges."""
    edges, edge_faces = _get_precomputed()
    n_edges = len(edges)
    
    # Map from edge index to position in order
    # We'll iterate through the order and check each edge
    selected = [False] * n_edges
    result = []
    
    for idx in order:
        if time.time() > deadline:
            break
        # Check if adding this edge creates a C4
        ok = True
        for neighbor in edge_faces[idx]:
            if selected[neighbor]:
                # Check if all 3 neighbors are selected
                # Actually we need to check if the specific 3 edges that form the C4 are all present
                # But edge_faces has 3 entries per face, so we need to check all 3
                # Let me re-think: for edge idx, edge_faces[idx] contains the 3 edges from each of the 6 faces
                # We need to check: for each face, are all 3 other edges selected?
                ok = False
                break
        # Wait, that's not right. Let me reconsider.
        # For edge idx, it belongs to 6 faces (since d-1=6 other dimensions).
        # For each face, there are 3 other edges. If ALL 3 are selected, we can't add idx.
        # So edge_faces[idx] should be grouped by face.
        
        # Let me redo: check each face's 3 edges
        ok = True
        for k in range(0, len(edge_faces[idx]), 3):
            f1, f2, f3 = edge_faces[idx][k], edge_faces[idx][k+1], edge_faces[idx][k+2]
            if selected[f1] and selected[f2] and selected[f3]:
                ok = False
                break
        
        if ok:
            selected[idx] = True
            result.append([edges[idx][0], edges[idx][1]])
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    edges, edge_faces = _get_precomputed()
    n_edges = len(edges)
    
    best_result = None
    best_size = 0
    
    # First, do a deterministic greedy to get a baseline
    order = list(range(n_edges))
    result = _greedy_randomized(order, deadline)
    best_result = result
    best_size = len(result)
    
    # Now do randomized restarts
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Create a random order
        order = list(range(n_edges))
        rng.shuffle(order)
        
        result = _greedy_randomized(order, deadline)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
        
        # Check time
        if time.time() > deadline:
            break
    
    return best_result if best_result else []
