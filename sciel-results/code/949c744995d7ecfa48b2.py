"""Program to construct a large (5,5)-free graph.
Strategy: Start from Paley graph on 37 vertices (verified 5,5-free),
then attempt to expand to 42 vertices using a greedy vertex addition
heuristic with backtracking, checking for K5 and independent 5-sets.
"""

import random
import sys
from itertools import combinations

# Set random seed for reproducibility
random.seed(42)


def _has_k5(adj, n):
    """Check if graph has a clique of size 5."""
    for combo in combinations(range(n), 5):
        valid = True
        for i in range(5):
            for j in range(i + 1, 5):
                if adj[combo[i]][combo[j]] == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return True
    return False


def _has_ind5(adj, n):
    """Check if graph has an independent set of size 5."""
    for combo in combinations(range(n), 5):
        valid = True
        for i in range(5):
            for j in range(i + 1, 5):
                if adj[combo[i]][combo[j]] == 1:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return True
    return False


def _is_55_free(adj, n):
    """Check if graph is (5,5)-free."""
    return not _has_k5(adj, n) and not _has_ind5(adj, n)


def _try_add_vertex(adj, n, new_vertex_adj):
    """Try to add a new vertex with given adjacency to existing vertices."""
    # new_vertex_adj is a boolean array of length n
    # Build new adjacency matrix
    new_n = n + 1
    new_adj = [[0] * new_n for _ in range(new_n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            new_adj[i][j] = adj[i][j]
            new_adj[j][i] = adj[i][j]
    
    for i in range(n):
        new_adj[i][new_n - 1] = 1 if new_vertex_adj[i] else 0
        new_adj[new_n - 1][i] = 1 if new_vertex_adj[i] else 0
    
    return new_adj


def _greedy_expand(adj, n, max_vertices=42, max_iterations=50000):
    """Try to expand the graph to max_vertices using greedy heuristic."""
    if n >= max_vertices:
        return adj, n
    
    iterations = 0
    while n < max_vertices and iterations < max_iterations:
        iterations += 1
        found = False
        
        # Try random patterns for the new vertex
        # Start with more structured patterns
        for attempt in range(100):
            if n + 1 > max_vertices:
                break
            
            # Generate random adjacency for new vertex
            if attempt < 20:
                # Try to connect to roughly half the vertices
                new_adj_list = [random.choice([0, 1]) for _ in range(n)]
            elif attempt < 40:
                # Try mostly connected
                new_adj_list = [random.choice([1, 0]) for _ in range(n)]
            elif attempt < 60:
                # Try mostly disconnected
                new_adj_list = [random.choice([0, 1]) for _ in range(n)]
            else:
                # Random
                new_adj_list = [random.choice([0, 1]) for _ in range(n)]
            
            new_adj = _try_add_vertex(adj, n, new_adj_list)
            
            if _is_55_free(new_adj, n + 1):
                adj = new_adj
                n += 1
                found = True
                break
        
        if not found:
            break
    
    return adj, n


def _try_edge_flips(adj, n, max_vertices=42, max_iterations=10000):
    """Try to improve the graph by flipping edges while maintaining (5,5)-freeness."""
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        
        # Try to flip a random edge
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        old_val = adj[i][j]
        new_val = 1 - old_val
        
        # Create new adjacency
        new_adj = [row[:] for row in adj]
        new_adj[i][j] = new_val
        new_adj[j][i] = new_val
        
        if _is_55_free(new_adj, n):
            adj = new_adj
            # Keep the flip
        else:
            # Revert
            pass
        
        # Occasionally try to add a vertex
        if n < max_vertices and iterations % 50 == 0:
            adj, n = _greedy_expand(adj, n, max_vertices, max_iterations=10000)
            if n >= max_vertices:
                break
    
    return adj, n


def construct():
    """Construct a (5,5)-free graph.
    
    Returns:
        tuple: (n, edges) where n is the number of vertices and edges is a list
               of (i, j) pairs with 0 <= i < j < n.
    """
    # Start with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    
    # Build adjacency matrix
    adj = [[0] * q for _ in range(q)]
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i][j] = 1
                adj[j][i] = 1
    
    n = q
    
    # Try to expand to 42 vertices
    adj, n = _greedy_expand(adj, n, max_vertices=42, max_iterations=100000)
    
    # If we haven't reached 42, try edge flips to potentially enable expansion
    if n < 42:
        adj, n = _try_edge_flips(adj, n, max_vertices=42, max_iterations=20000)
        
        # After edge flips, try greedy expansion again
        adj, n = _greedy_expand(adj, n, max_vertices=42, max_iterations=100000)
    
    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j] == 1:
                edges.append((i, j))
    
    return n, edges


if __name__ == "__main__":
    n, edges = construct()
    print(f"Constructed graph with {n} vertices and {len(edges)} edges")
