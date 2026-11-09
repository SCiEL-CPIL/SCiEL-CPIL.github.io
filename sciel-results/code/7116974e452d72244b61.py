"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = None
_DEADLINE = 135.0  # seconds, safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

# Q3 structure constants
_Q3_ADJ = {
    0: {1, 2, 4},
    1: {0, 3, 5},
    2: {0, 3, 6},
    3: {1, 2, 7},
    4: {0, 5, 6},
    5: {1, 4, 7},
    6: {2, 4, 7},
    7: {3, 5, 6}
}
_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

def _has_q3_in_8(adj_matrix, vlist):
    """Check if the subgraph induced by vlist (8 vertices) contains Q3.
    adj_matrix is a 16x16 boolean numpy array (or list of lists).
    vlist is a list of 8 vertex indices."""
    import numpy as np
    # Build local adjacency
    # vlist[i] -> local index i
    # Check if induced subgraph has at least 12 edges
    local_adj = np.zeros((8, 8), dtype=bool)
    for i in range(8):
        for j in range(i+1, 8):
            if adj_matrix[vlist[i], vlist[j]]:
                local_adj[i, j] = True
                local_adj[j, i] = True
    
    edge_count = np.sum(local_adj) // 2
    if edge_count < 12:
        return False
    
    # Backtracking to find Q3 embedding
    # Order Q3 vertices for better pruning: start with vertex 0 (degree 3)
    # We map Q3 vertex q to local vertex l
    # Use a simple backtracking with the constraint that neighbors must be adjacent
    
    mapping = [-1] * 8  # mapping[q] = local index
    used = [False] * 8
    
    # Precompute for each Q3 vertex, the list of Q3 neighbors
    q3_neighbors = [list(_Q3_ADJ[q]) for q in range(8)]
    
    def backtrack(q):
        if q == 8:
            return True
        # Try each unused local vertex
        for l in range(8):
            if used[l]:
                continue
            # Check adjacency constraints with already mapped neighbors
            ok = True
            for nu in q3_neighbors[q]:
                if nu < q:  # already mapped
                    if not local_adj[l, mapping[nu]]:
                        ok = False
                        break
            if not ok:
                continue
            mapping[q] = l
            used[l] = True
            if backtrack(q + 1):
                return True
            mapping[q] = -1
            used[l] = False
        return False
    
    return backtrack(0)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    import numpy as np
    
    # Start with K3,13: vertices 0,1,2 connect to 3..15
    adj = np.zeros((16, 16), dtype=bool)
    for a in range(3):
        for b in range(3, 16):
            adj[a, b] = True
            adj[b, a] = True
    
    best_adj = adj.copy()
    best_count = np.sum(adj) // 2  # 39
    
    # Candidate edges: all pairs not already present
    all_pairs = [(i, j) for i in range(16) for j in range(i+1, 16)]
    remaining = [p for p in all_pairs if not adj[p[0], p[1]]]
    
    random.seed(42)
    random.shuffle(remaining)
    
    # Precompute: for each pair (u,v), the 6-subsets of other vertices to check
    # Actually, we'll just iterate and check
    
    for (u, v) in remaining:
        if _elapsed() > _DEADLINE:
            break
        
        # Check if adding (u,v) creates a Q3
        # A new Q3 must use edge (u,v)
        # So we need 6 other vertices from the remaining 14 such that the induced subgraph on {u,v}+6 contains Q3
        # with (u,v) being an edge in the Q3.
        
        # In Q3, every edge is in exactly 4 distinct 4-cycles, but for subgraph:
        # Fix edge (0,1) in Q3. The other 6 vertices are {2,3,4,5,6,7}.
        # We need to find an injection from {2,3,4,5,6,7} to the 14 other vertices
        # such that all Q3 edges are present.
        
        # This is expensive. Let's use a smarter check:
        # Temporarily add edge (u,v)
        adj[u, v] = True
        adj[v, u] = True
        
        # Now check if there's any 8-vertex subset containing u and v that has Q3
        # But checking all C(14,6) = 3003 subsets is too slow for each edge.
        
        # Alternative: check if the graph now contains Q3 at all? No, that's even slower.
        
        # Better: only check subsets that could potentially contain Q3.
        # A Q3 containing edge (u,v) requires:
        # - 2 vertices a,b adjacent to u (in the full graph) that will map to Q3's {2,4} (neighbors of 0 except 1)
        # - 2 vertices c,d adjacent to v (in the full graph) that will map to Q3's {3,5} (neighbors of 1 except 0)
        # - The remaining 2 vertices e,f must complete the structure
        
        # Actually, let's just do the 6-subset check but with early termination and the numpy-based check.
        # C(14,6) = 3003, and each check is fast with numpy.
        
        other_vertices = [x for x in range(16) if x != u and x != v]
        
        found_q3 = False
        for subset in combinations(other_vertices, 6):
            vlist = [u, v] + list(subset)
            if _has_q3_in_8(adj, vlist):
                found_q3 = True
                break
        
        if found_q3:
            # Remove the edge
            adj[u, v] = False
            adj[v, u] = False
        else:
            # Keep the edge
            new_count = np.sum(adj) // 2
            if new_count > best_count:
                best_count = new_count
                best_adj = adj.copy()
    
    # Return the best graph found
    edges = []
    for i in range(16):
        for j in range(i+1, 16):
            if best_adj[i, j]:
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END
