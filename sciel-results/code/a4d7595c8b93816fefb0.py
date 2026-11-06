"""
K4-free graphs without independent 6-sets (Ramsey number R(4,6)).

Strategy:
1. Start from known small valid graphs (e.g., Paley(17)).
2. Use a greedy/repair heuristic to grow the graph:
   - Try to add a new vertex connected to a subset of existing vertices.
   - The new vertex must not create a K4 with any existing triangle it joins.
   - The new vertex must not create an independent set of size 6 with existing independent sets.
   - Use local search (edge flips) to fix violations if any arise.
3. Use a time budget to ensure we finish within the limit.

Key insights:
- A vertex v can be added to a graph G if:
  a) For every triangle {a,b,c} in G, v is not connected to all of a,b,c (to avoid K4).
  b) For every independent set of size 5 in G, v is connected to at least one of them (to avoid I6).
- We can precompute triangles and independent sets of size 5 for efficiency.
- If adding a vertex creates violations, we can try to fix them by flipping edges incident to the new vertex.

For larger n, we use a more sophisticated approach:
- Start with a random graph or a known construction.
- Use simulated annealing or local search to minimize the number of K4s and I6s.
- Accept moves that reduce the objective (number of K4s + number of I6s).
"""

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds to leave margin
    
    # Start with the Paley graph on 17 vertices
    best_n = 17
    best_edges = _paley_edges(17)
    
    # Try to grow the graph using the repair heuristic
    current_n = 17
    current_edges = best_edges[:]
    
    # Precompute adjacency for efficiency
    adj = _edges_to_adj(current_n, current_edges)
    
    # Try to add vertices one by one
    for target_n in range(18, 36):
        if time.time() > deadline - 10:
            break
        
        # Try to add a vertex with a specific connection pattern
        success = _try_add_vertex(adj, current_n, target_n)
        if success:
            current_n = target_n
            current_edges = _adj_to_edges(adj, current_n)
            best_n = current_n
            best_edges = current_edges[:]
        else:
            # Try a few random restarts of the addition
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                adj_try = [row[:] for row in adj]
                if _try_add_vertex(adj_try, current_n, target_n, random=True):
                    current_n = target_n
                    current_edges = _adj_to_edges(adj_try, current_n)
                    best_n = current_n
                    best_edges = current_edges[:]
                    break
            if time.time() > deadline - 10:
                break
    
    # Final local search to try to improve
    if time.time() < deadline - 20:
        adj = _edges_to_adj(best_n, best_edges)
        improved = _local_search(adj, best_n, deadline - 15)
        if improved:
            best_edges = _adj_to_edges(adj, best_n)
    
    return {'n': best_n, 'edges': best_edges}


def _paley_edges(n):
    """Return edges of the Paley graph on n vertices (n must be a prime power ≡ 1 mod 4)."""
    if n == 17:
        squares = {x * x % 17 for x in range(1, 17)}
        return [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    # For other n, fall back to a simple construction
    return [[i, j] for i in range(n) for j in range(i + 1, n) if (i * j) % n != 0]


def _edges_to_adj(n, edges):
    """Convert edge list to adjacency matrix (list of lists)."""
    adj = [[0] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    return adj


def _adj_to_edges(adj, n):
    """Convert adjacency matrix to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges


def _count_k4s(adj, n):
    """Count the number of K4s in the graph."""
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                for k in range(j + 1, n):
                    if adj[i][k] and adj[j][k]:
                        for l in range(k + 1, n):
                            if adj[i][l] and adj[j][l] and adj[k][l]:
                                count += 1
    return count


def _count_ind6s(adj, n):
    """Count the number of independent sets of size 6 in the graph."""
    count = 0
    for combo in combinations(range(n), 6):
        is_independent = True
        for a in range(6):
            for b in range(a + 1, 6):
                if adj[combo[a]][combo[b]]:
                    is_independent = False
                    break
            if not is_independent:
                break
        if is_independent:
            count += 1
    return count


def _try_add_vertex(adj, current_n, target_n, random=False):
    """
    Try to add a vertex (target_n - 1) to the graph.
    Returns True if successful, False otherwise.
    Modifies adj in place.
    """
    new_vertex = target_n - 1
    
    # Extend adjacency matrix
    for row in adj:
        row.append(0)
    adj.append([0] * target_n)
    
    # Determine which existing vertices the new vertex should connect to
    if random:
        # Random connection pattern
        for i in range(current_n):
            adj[new_vertex][i] = 1 if random.random() < 0.5 else 0
            adj[i][new_vertex] = adj[new_vertex][i]
    else:
        # Greedy: connect to vertices that don't create immediate K4s
        for i in range(current_n):
            # Check if connecting to i would create a K4
            can_connect = True
            for j in range(i):
                if adj[i][j]:
                    for k in range(j):
                        if adj[i][k] and adj[j][k]:
                            # i, j, k form a triangle; if new_vertex connects to all, K4
                            # So don't connect to i if it's part of a triangle
                            can_connect = False
                            break
                if not can_connect:
                    break
            if can_connect:
                adj[new_vertex][i] = 1
                adj[i][new_vertex] = 1
    
    # Check for violations and try to fix
    k4_count = _count_k4s(adj, target_n)
    ind6_count = _count_ind6s(adj, target_n)
    
    if k4_count == 0 and ind6_count == 0:
        return True
    
    # Try to fix by flipping edges incident to the new vertex
    for _ in range(100):
        if k4_count == 0 and ind6_count == 0:
            return True
        
        # Flip a random edge incident to the new vertex
        i = random.randint(0, current_n - 1)
        adj[new_vertex][i] = 1 - adj[new_vertex][i]
        adj[i][new_vertex] = adj[new_vertex][i]
        
        new_k4 = _count_k4s(adj, target_n)
        new_ind6 = _count_ind6s(adj, target_n)
        
        if new_k4 + new_ind6 < k4_count + ind6_count:
            k4_count = new_k4
            ind6_count = new_ind6
        else:
            # Revert
            adj[new_vertex][i] = 1 - adj[new_vertex][i]
            adj[i][new_vertex] = adj[new_vertex][i]
    
    return k4_count == 0 and ind6_count == 0


def _local_search(adj, n, deadline):
    """
    Local search to eliminate K4s and I6s.
    Returns True if the graph is valid.
    """
    k4_count = _count_k4s(adj, n)
    ind6_count = _count_ind6s(adj, n)
    
    if k4_count == 0 and ind6_count == 0:
        return True
    
    for _ in range(10000):
        if time.time() > deadline:
            break
        
        if k4_count == 0 and ind6_count == 0:
            return True
        
        # Try flipping a random edge
        i = random.randint(0, n - 1)
        j = random.randint(i + 1, n - 1)
        adj[i][j] = 1 - adj[i][j]
        adj[j][i] = adj[i][j]
        
        new_k4 = _count_k4s(adj, n)
        new_ind6 = _count_ind6s(adj, n)
        
        if new_k4 + new_ind6 < k4_count + ind6_count:
            k4_count = new_k4
            ind6_count = new_ind6
        else:
            # Revert
            adj[i][j] = 1 - adj[i][j]
            adj[j][i] = adj[i][j]
    
    return k4_count == 0 and ind6_count == 0
