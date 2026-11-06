"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _build_adjacency(n, edges):
    """Build adjacency as list of bitmasks and degrees."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    deg = [bin(adj[u]).count('1') for u in range(n)]
    return adj, deg

def _find_all_triangles(adj, n):
    """Find all triangles using bitmask intersection. Returns list of (u,v,w) with u<v<w."""
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                continue
            # Common neighbors of u and v that are > v
            common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
            w = common
            while w:
                low = w & (-w)
                w &= w - 1
                wi = low.bit_length() - 1
                triangles.append((u, v, wi))
    return triangles

def _max_triangle_degree_sum(adj, deg, triangles):
    """Compute the maximum degree sum over all triangles."""
    if not triangles:
        return 0
    max_sum = 0
    for u, v, w in triangles:
        s = deg[u] + deg[v] + deg[w]
        if s > max_sum:
            max_sum = s
    return max_sum

def _edge_set_from_adj(adj, n):
    """Extract edge set from bitmask adjacency."""
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                edges.add((u, v))
    return edges

def _add_edge(adj, deg, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    deg[u] += 1
    deg[v] += 1

def _remove_edge(adj, deg, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    deg[u] -= 1
    deg[v] -= 1

def _try_improve(adj, deg, n, max_iter=100000, deadline=None):
    """
    Iterative local search using bitmask adjacency for fast triangle enumeration.
    """
    triangles = _find_all_triangles(adj, n)
    if not triangles:
        return adj, deg
    
    current_max = _max_triangle_degree_sum(adj, deg, triangles)
    best_adj = list(adj)
    best_deg = list(deg)
    best_max = current_max
    
    start_time = time.time()
    
    for iteration in range(max_iter):
        if deadline and time.time() > deadline - 5:
            break
        if time.time() - start_time > 110:
            break
        
        # Find the worst triangle
        worst_tri = None
        worst_sum = -1
        for u, v, w in triangles:
            s = deg[u] + deg[v] + deg[w]
            if s > worst_sum:
                worst_sum = s
                worst_tri = (u, v, w)
        
        if worst_tri is None:
            break
        
        u, v, w = worst_tri
        
        # Try to remove one edge from the worst triangle and add a non-edge
        candidates = [(u, v), (u, w), (v, w)]
        random.shuffle(candidates)
        
        improved = False
        for rem_u, rem_v in candidates:
            # Try removing edge (rem_u, rem_v)
            _remove_edge(adj, deg, rem_u, rem_v)
            
            # Find non-edges to add back (need to maintain exactly 65 edges)
            possible_adds = []
            for a in range(n):
                for b in range(a + 1, n):
                    if not (adj[a] & (1 << b)):
                        # Check if adding this edge creates triangles
                        common = adj[a] & adj[b]
                        if common:
                            max_new = 0
                            c = common
                            while c:
                                low = c & (-c)
                                c &= c - 1
                                ci = low.bit_length() - 1
                                s = deg[a] + deg[b] + deg[ci]
                                if s > max_new:
                                    max_new = s
                        else:
                            max_new = 0
                        possible_adds.append((max_new, a, b))
            
            if not possible_adds:
                # Restore
                _add_edge(adj, deg, rem_u, rem_v)
                continue
            
            # Sort by estimated max new triangle degree sum
            possible_adds.sort()
            
            # Try the best few candidates
            for est_max, add_u, add_v in possible_adds[:8]:
                # Add the edge
                _add_edge(adj, deg, add_u, add_v)
                
                # Recompute triangles and max
                new_triangles = _find_all_triangles(adj, n)
                new_max = _max_triangle_degree_sum(adj, deg, new_triangles)
                
                if new_max < best_max:
                    best_max = new_max
                    best_adj = list(adj)
                    best_deg = list(deg)
                    triangles = new_triangles
                    improved = True
                    break
                else:
                    # Undo
                    _remove_edge(adj, deg, add_u, add_v)
            
            if improved:
                break
            
            # Restore removed edge
            _add_edge(adj, deg, rem_u, rem_v)
        
        if not improved:
            # Try a random perturbation
            if random.random() < 0.3:
                # Randomly remove an edge from a triangle and add a random non-edge
                if triangles:
                    u, v, w = random.choice(triangles)
                    rem_u, rem_v = random.choice([(u, v), (u, w), (v, w)])
                    _remove_edge(adj, deg, rem_u, rem_v)
                    
                    # Find all non-edges
                    non_edges = []
                    for a in range(n):
                        for b in range(a + 1, n):
                            if not (adj[a] & (1 << b)):
                                non_edges.append((a, b))
                    
                    if non_edges:
                        add_u, add_v = random.choice(non_edges)
                        _add_edge(adj, deg, add_u, add_v)
                        
                        triangles = _find_all_triangles(adj, n)
                        new_max = _max_triangle_degree_sum(adj, deg, triangles)
                        if new_max < best_max:
                            best_max = new_max
                            best_adj = list(adj)
                            best_deg = list(deg)
    
    return best_adj, best_deg

def _initial_graph(n=16, target_edges=65):
    """
    Start with K_{8,8} + 1 edge, then try to optimize.
    Returns bitmask adjacency and degrees.
    """
    # K_{8,8}: vertices 0-7 and 8-15, all cross edges
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    # Add one intra-part edge to get 65 edges
    edges.append([0, 1])
    
    adj, deg = _build_adjacency(n, edges)
    
    return adj, deg

def construct():
    n = 16
    target_edges = 65
    
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin
    
    # Start with the base construction
    adj, deg = _initial_graph(n, target_edges)
    
    # Run local search
    adj, deg = _try_improve(adj, deg, n, max_iter=200000, deadline=deadline)
    
    # Extract edges from bitmask adjacency
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                edges.append([u, v])
    
    # Verify
    assert len(edges) == target_edges, f"Expected {target_edges} edges, got {len(edges)}"
    
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
