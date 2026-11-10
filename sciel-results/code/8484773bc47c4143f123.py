"""Rewritten program for clique partitioning of two fixed dense random graphs."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def _generate_edges(n, p, seed):
    """Generate the edge list for a random graph using the same RNG as the verifier."""
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adjacency(n, edges):
    """Build adjacency sets for fast clique verification."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_clique(vertices, adj):
    """Check if all vertices form a clique."""
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if vertices[j] not in adj[vertices[i]]:
                return False
    return True

def _maximal_cliques(n, adj):
    """Find all maximal cliques using Bron-Kerbosch with pivot."""
    cliques = []
    
    def bronk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(set(R))
            return
        
        # Pivot selection: choose vertex in P ∪ X with max neighbors in P
        pivot_candidates = list(P | X)
        if pivot_candidates:
            pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P))
        else:
            return
        
        # Only consider vertices in P that are NOT adjacent to pivot
        candidates = [v for v in P if v not in adj[pivot]]
        
        for v in candidates:
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    
    bronk(set(), set(range(n)), set())
    return cliques

def _greedy_clique_partition(n, edges, adj):
    """
    Greedy algorithm to partition edges into cliques.
    Strategy: repeatedly find the largest clique among remaining edges,
    add it to the partition, and remove its edges.
    """
    # Build edge set for quick removal
    remaining_edges = set(edges)
    if not remaining_edges:
        return []
    
    cliques = []
    
    # Precompute all maximal cliques for reference
    # But for efficiency, we'll use a greedy approach:
    # At each step, find a clique that covers the most remaining edges.
    
    # For dense graphs, we can try to find large cliques.
    # Strategy: for each vertex, find the largest clique containing it using the remaining edges.
    
    while remaining_edges:
        best_clique = None
        best_count = 0
        
        # Try to find a good clique by looking at each vertex
        # For each vertex, consider its neighbors in the remaining graph
        for u in range(n):
            # Get neighbors of u that are still connected via remaining edges
            neighbors = set()
            for v in adj[u]:
                if (min(u, v), max(u, v)) in remaining_edges:
                    neighbors.add(v)
            
            if len(neighbors) < 1:
                continue
            
            # Try to build a clique starting from u
            # Use a simple greedy: repeatedly add the vertex with the most common neighbors
            clique = {u}
            candidates = set(neighbors)
            
            while candidates:
                # Find the vertex in candidates that has the most edges to all current clique members
                # and is connected to all of them
                best_v = None
                best_score = 0
                for v in candidates:
                    # Check if v is connected to all in clique
                    is_connected = all(w in adj[v] for w in clique)
                    if not is_connected:
                        continue
                    # Count how many remaining edges v has to other candidates
                    score = sum(1 for w in candidates if w != v and (min(v, w), max(v, w)) in remaining_edges)
                    if score > best_score:
                        best_score = score
                        best_v = v
                
                if best_v is None:
                    break
                
                clique.add(best_v)
                candidates.remove(best_v)
            
            # Check how many remaining edges this clique covers
            clique_edges = set()
            clique_list = list(clique)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    e = (min(a, b), max(a, b))
                    if e in remaining_edges:
                        clique_edges.add(e)
            
            if len(clique_edges) > best_count:
                best_count = len(clique_edges)
                best_clique = clique_list
        
        if best_clique is None or best_count == 0:
            # Fallback: just take any remaining edge
            e = next(iter(remaining_edges))
            best_clique = [e[0], e[1]]
        
        cliques.append(best_clique)
        
        # Remove edges covered by this clique
        clique_set = set(best_clique)
        for i in range(len(best_clique)):
            for j in range(i + 1, len(best_clique)):
                a, b = best_clique[i], best_clique[j]
                e = (min(a, b), max(a, b))
                remaining_edges.discard(e)
    
    return cliques

def construct():
    """Construct clique partitions for two fixed dense random graphs."""
    results = []
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = _generate_edges(n, p, seed)
        adj = _build_adjacency(n, edges)
        cliques = _greedy_clique_partition(n, edges, adj)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
