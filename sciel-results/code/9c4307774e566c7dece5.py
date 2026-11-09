"""Construct a C4-free graph with high minimum degree using finite geometry / combinatorial designs."""
import time
import random
from itertools import combinations

def _is_c4_free(n, edges):
    """Check if graph is C4-free using bitsets for speed."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            # Count bits in common - if >= 2, C4 exists
            c = common
            if c & (c - 1):  # More than one bit set
                return False
    return True

def _creates_c4(adj, u, v):
    """Check if adding edge (u,v) would create a C4."""
    # Need w in N(v), x in N(u), w~x, w!=x, w!=u, x!=v
    nv = adj[v]
    nu = adj[u]
    
    w_mask = nv
    while w_mask:
        w_bit = w_mask & (-w_mask)
        w = w_bit.bit_length() - 1
        w_mask ^= w_bit
        
        if w == u:
            continue
        
        # Check if any neighbor of u (that isn't v or w) is adjacent to w
        x_mask = nu & ~w_bit & ~(1 << v)
        if x_mask & adj[w]:
            return True
    return False

def _greedy_build(target_n, seed, deadline):
    """Build a C4-free graph using random greedy edge addition."""
    random.seed(seed)
    edges = []
    adj = [0] * target_n
    deg = [0] * target_n
    
    all_edges = list(combinations(range(target_n), 2))
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline:
            break
        
        if not _creates_c4(adj, u, v):
            edges.append([u, v])
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            deg[u] += 1
            deg[v] += 1
    
    return edges, adj, deg

def _local_search(edges, adj, deg, target_n, min_deg_needed, deadline):
    """Improve the graph via edge swaps to increase minimum degree."""
    edges_set = set()
    for u, v in edges:
        edges_set.add((min(u,v), max(u,v)))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Find vertices with minimum degree
        min_deg = min(deg)
        if min_deg >= min_deg_needed:
            break
        
        # Try to add edges to low-degree vertices
        low_deg_vertices = [i for i in range(target_n) if deg[i] == min_deg]
        random.shuffle(low_deg_vertices)
        
        for vertex in low_deg_vertices:
            if time.time() > deadline:
                break
            
            # Find non-neighbors to potentially add
            non_neighbors = [v for v in range(target_n) 
                           if v != vertex and not (adj[vertex] >> v) & 1]
            random.shuffle(non_neighbors)
            
            for candidate in non_neighbors:
                if time.time() > deadline:
                    break
                
                # Try adding the edge
                if not _creates_c4(adj, vertex, candidate):
                    edge_key = (min(vertex, candidate), max(vertex, candidate))
                    if edge_key not in edges_set:
                        edges.append([vertex, candidate])
                        edges_set.add(edge_key)
                        adj[vertex] |= 1 << candidate
                        adj[candidate] |= 1 << vertex
                        deg[vertex] += 1
                        deg[candidate] += 1
                        improved = True
                        break
    
    return edges, adj, deg

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_n = 39
    best_edges = []
    
    # Try to build graphs for n from 47 down to 40
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
        min_deg_needed = target_n - 39
        
        # Multiple attempts with different seeds
        found_valid = False
        for seed in range(200):
            if time.time() > deadline:
                break
            
            # Greedy construction
            edges, adj, deg = _greedy_build(target_n, seed, deadline)
            
            # Check if we already meet the minimum degree requirement
            if min(deg) >= min_deg_needed:
                # Verify C4-free
                if _is_c4_free(target_n, edges):
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = edges
                    found_valid = True
                    break
            
            # If not, try local search to improve
            if min(deg) < min_deg_needed:
                edges, adj, deg = _local_search(edges, adj, deg, target_n, min_deg_needed, deadline)
                
                if min(deg) >= min_deg_needed:
                    # Verify C4-free
                    if _is_c4_free(target_n, edges):
                        if target_n > best_n:
                            best_n = target_n
                            best_edges = edges
                        found_valid = True
                        break
        
        # If we found a valid graph for this target_n, we can stop trying smaller n
        if found_valid:
            break
    
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
