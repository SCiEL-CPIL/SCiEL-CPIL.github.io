import random
import time
import sys

def _has_c6(u, v, adj):
    """
    Check if adding edge (u, v) to the graph defined by adjacency list 'adj'
    creates a cycle of length 6.
    This is true if and only if there exists a simple path of length 5
    from u to v in the current graph.
    """
    # We need a path u -> n1 -> n2 -> n3 -> n4 -> v
    # Depth 5 DFS from u
    visited = {u}
    
    def dfs(curr, depth):
        if depth == 5:
            return curr == v
        for next_node in adj[curr]:
            if next_node not in visited:
                visited.add(next_node)
                if dfs(next_node, depth + 1):
                    return True
                visited.remove(next_node)
        return False
    
    return dfs(u, 0)

def _greedy_add(order, initial_edges, adj, edge_set):
    """
    Greedily add edges from 'order' to the graph, starting from 'initial_edges'.
    Returns the list of added edges and the final edge set.
    """
    added = []
    for u, v in order:
        if (u, v) in edge_set or (v, u) in edge_set:
            continue
        # Check if adding (u, v) creates a C6
        if not _has_c6(u, v, adj):
            # Add edge
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((u, v))
            edge_set.add((v, u)) # For quick lookup if we use set of tuples
            added.append((u, v))
    return added, edge_set

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initial seed: 8 disjoint K5s
    seed_edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                seed_edges.append((5 * k + a, 5 * k + b))
    
    best_edges = seed_edges[:]
    best_count = len(seed_edges)
    
    # Helper to build adjacency structures
    def build_adj(edges):
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((u, v))
            edge_set.add((v, u))
        return adj, edge_set

    # Run multiple random restarts
    while time.time() < deadline:
        # Random order of edges
        order = all_pairs[:]
        random.shuffle(order)
        
        # Start from empty graph
        adj, edge_set = build_adj([])
        added, _ = _greedy_add(order, [], adj, edge_set)
        current_edges = added[:]
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
        
        # Start from seed graph
        adj_seed, edge_set_seed = build_adj(seed_edges)
        added_seed, _ = _greedy_add(order, seed_edges, adj_seed, edge_set_seed)
        current_edges_seed = seed_edges[:] + added_seed
        if len(current_edges_seed) > best_count:
            best_count = len(current_edges_seed)
            best_edges = current_edges_seed[:]
            
        # If we have time, try a few more passes or local search on the best found
        # For now, just do more restarts as they are cheap and effective.
        
    # Return in the required format: list of [u, v]
    return [[u, v] for u, v in best_edges]
