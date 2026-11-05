"""Clique partition of two fixed dense random graphs."""
import time
from itertools import combinations

def _build_graph(n, p, seed):
    """Reproduce the exact edge set from the verifier's random_graph logic."""
    from random import Random
    rng = Random(seed)
    edges = set()
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
    return edges, adj

def _greedy_clique_partition(edges, adj, n, time_limit=140.0):
    """
    Partition edges into cliques using a greedy approach.
    At each step, find a clique that covers the most uncovered edges.
    """
    start_time = time.time()
    uncovered = set(edges)
    cliques = []
    
    # Precompute common neighbors for speed
    # For a clique, all vertices must be mutually adjacent.
    # Strategy: pick an uncovered edge (u,v), try to expand it into a maximal clique.
    # Then check if that clique covers more uncovered edges than any other candidate.
    
    # To be efficient, we'll use a heuristic:
    # 1. Find a "good" starting edge (one with high potential for expansion).
    # 2. Expand it greedily into a maximal clique.
    # 3. Add that clique to the partition and remove its edges from uncovered.
    
    # Better heuristic for dense graphs:
    # Since graphs are dense (p=0.6, p=0.55), cliques can be large.
    # We want cliques that cover many edges.
    
    # Let's use a different approach:
    # 1. Find all maximal cliques using a heuristic (Bron-Kerbosch with pivot, but limited).
    # 2. Use greedy set cover: at each step, pick the clique that covers the most uncovered edges.
    
    # However, finding all maximal cliques can be expensive.
    # Alternative: iteratively find a large clique covering many uncovered edges.
    
    # Let's try:
    # 1. While uncovered edges remain:
    #    a. Pick an uncovered edge (u, v) with maximum degree in the subgraph induced by uncovered edges.
    #    b. Try to expand (u, v) into a maximal clique within the uncovered subgraph.
    #    c. Add this clique to the partition.
    #    d. Remove all edges of this clique from uncovered.
    
    # But "maximal clique within uncovered subgraph" is tricky because the subgraph changes.
    
    # Simpler approach:
    # 1. While uncovered edges remain:
    #    a. Pick a random uncovered edge (u, v).
    #    b. Expand (u, v) into a clique by greedily adding vertices that are adjacent to all current clique members.
    #    c. Add this clique to the partition.
    #    d. Remove all edges of this clique from uncovered.
    
    # Even simpler and potentially effective for dense graphs:
    # 1. Find a large clique in the full graph.
    # 2. Add it to the partition.
    # 3. Remove its edges.
    # 4. Repeat.
    
    # Let's implement a heuristic for finding a large clique:
    # Start with a high-degree vertex, add neighbors that are mutually adjacent.
    
    while uncovered:
        if time.time() - start_time > time_limit - 5:
            # Fallback: add remaining edges as 2-vertex cliques
            for e in uncovered:
                cliques.append(list(e))
            break
        
        # Pick an uncovered edge with maximum potential
        # Use a heuristic: pick edge (u,v) maximizing (deg(u) + deg(v)) in uncovered subgraph
        best_edge = None
        best_score = -1
        # To save time, sample a subset of uncovered edges
        sample = list(uncovered)
        if len(sample) > 100:
            import random
            rng = random.Random(42)
            sample = rng.sample(sample, 100)
        
        for e in sample:
            u, v = e
            # Count how many common neighbors u and v have in the uncovered subgraph
            # This is expensive, so use a simpler heuristic
            score = len(adj[u]) + len(adj[v])
            if score > best_score:
                best_score = score
                best_edge = e
        
        if best_edge is None:
            best_edge = next(iter(uncovered))
        
        u, v = best_edge
        # Expand (u, v) into a clique
        clique = {u, v}
        candidates = adj[u] & adj[v]
        
        # Greedily add vertices
        while candidates:
            # Pick candidate with maximum degree in the current clique's common neighborhood
            best_cand = None
            best_cand_score = -1
            for cand in candidates:
                # Check if cand is adjacent to all current clique members
                is_adjacent = all(cand in adj[m] for m in clique)
                if not is_adjacent:
                    continue
                # Score: number of common neighbors with current clique
                score = sum(1 for m in clique if cand in adj[m])
                if score > best_cand_score:
                    best_cand_score = score
                    best_cand = cand
            
            if best_cand is None:
                break
            
            clique.add(best_cand)
            # Update candidates
            candidates = candidates & adj[best_cand]
        
        # Convert to list and add to cliques
        clique_list = list(clique)
        if len(clique_list) >= 2:
            cliques.append(clique_list)
            # Remove all edges of this clique from uncovered
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    e1 = (clique_list[i], clique_list[j])
                    e2 = (clique_list[j], clique_list[i])
                    uncovered.discard(e1)
                    uncovered.discard(e2)
    
    return cliques

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges, adj = _build_graph(n, p, seed)
        cliques = _greedy_clique_partition(edges, adj, n)
        results.append(cliques)
    return results
