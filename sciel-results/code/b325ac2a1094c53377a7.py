"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def _generate_graph(n, p, seed):
    """Generate a graph matching the verifier's random_graph(n, p, seed)."""
    from random import Random
    rng = Random(seed)
    adj = [set() for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
    return adj, edges

def _max_clique_greedy(adj, start_vertex, candidates):
    """Find a maximal clique starting from start_vertex using greedy coloring heuristic."""
    # We want to find a clique within the candidate set that includes start_vertex
    # Use a simple greedy approach: keep adding vertices that connect to all current clique members
    clique = [start_vertex]
    current_candidates = set(candidates)
    current_candidates.discard(start_vertex)
    
    while current_candidates:
        # Pick the vertex in current_candidates that has the highest degree within the clique
        # and is connected to all clique members
        best = None
        best_score = -1
        for v in current_candidates:
            if all(u in adj[v] for u in clique):
                score = sum(1 for w in current_candidates if w in adj[v])
                if score > best_score:
                    best_score = score
                    best = v
        if best is None:
            break
        clique.append(best)
        current_candidates.discard(best)
        # Remove candidates that are not connected to best
        current_candidates = {v for v in current_candidates if best in adj[v]}
    
    return clique

def _solve_instance(n, p, seed, time_budget=110.0):
    """Find a clique partition for one graph instance."""
    start_time = time.time()
    adj, edges = _generate_graph(n, p, seed)
    
    if not edges:
        return []
    
    # Track uncovered edges
    uncovered = set(edges)
    partition = []
    
    while uncovered:
        # Check time budget
        if time.time() - start_time > time_budget:
            break
        
        # Find the most constrained edge (both endpoints have fewest uncovered edges)
        # Use the edge where the minimum degree is smallest
        best_edge = None
        best_min_deg = float('inf')
        
        for (u, v) in uncovered:
            deg_u = len(adj[u] & {w for (a, b) in uncovered for w in (a, b) if a == u or b == u})
            # This is expensive; let's use a simpler heuristic
            # Just pick an edge somewhat randomly but prefer edges where both endpoints have fewer options
            pass
        
        # Simpler: pick an edge, then grow a clique
        # Pick edge with minimum sum of degrees in the remaining graph
        # For efficiency, just pick the first edge and grow
        edge = next(iter(uncovered))
        u, v = edge
        
        # Try to find a large clique containing this edge
        # Start with {u, v} and greedily add vertices
        clique = [u, v]
        candidates = adj[u] & adj[v]
        candidates.discard(u)
        candidates.discard(v)
        
        # Greedy expansion
        while candidates:
            # Pick vertex with max connections to other candidates and to clique
            best = None
            best_score = -1
            for w in candidates:
                if all(x in adj[w] for x in clique):
                    score = len(adj[w] & candidates)
                    if score > best_score:
                        best_score = score
                        best = w
            if best is None:
                break
            clique.append(best)
            candidates.discard(best)
            candidates = {w for w in candidates if best in adj[w]}
        
        # Remove all edges within the clique from uncovered
        clique_set = set(clique)
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                a, b = clique[i], clique[j]
                if a < b:
                    uncovered.discard((a, b))
                else:
                    uncovered.discard((b, a))
        
        partition.append(clique)
    
    # Handle any remaining edges as single-edge cliques
    for edge in uncovered:
        partition.append(list(edge))
    
    return partition

def construct():
    deadline = time.time() + 118.0  # 2s margin before 120s limit
    result1 = _solve_instance(32, 0.6, 601, time_budget=55.0)
    if time.time() > deadline:
        # Fallback: just use single-edge cliques
        _, edges1 = _generate_graph(32, 0.6, 601)
        result1 = [[list(e)] for e in edges1]
    result2 = _solve_instance(64, 0.55, 602, time_budget=55.0)
    if time.time() > deadline:
        _, edges2 = _generate_graph(64, 0.55, 602)
        result2 = [[list(e)] for e in edges2]
    return [result1, result2]
# EVOLVE-BLOCK-END
