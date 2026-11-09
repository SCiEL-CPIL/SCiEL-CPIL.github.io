"""Clique partition of dense random graphs: greedy triangle extraction + edge pairing."""
import time
from random import Random
from collections import defaultdict

def _random_graph(n, p, seed):
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed, deadline):
    edges = _random_graph(n, p, seed)
    # Build adjacency sets for fast triangle checks
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Track which edges are still available
    edge_set = set(edges)
    cliques = []
    
    # Greedy: repeatedly find and extract triangles, then pair remaining edges
    # We'll use a simple approach:
    # 1. While there exist triangles in the remaining edge set, extract them
    # 2. Pair up remaining edges that share a vertex to form 2-cliques (but 2-cliques are just edges)
    # Actually, a "clique" of size 2 is just an edge. So the minimum partition is just all edges.
    # But we can do better by grouping edges into larger cliques.
    # The goal is to minimize the NUMBER of cliques, so we want large cliques.
    
    # Strategy: find maximal cliques greedily.
    # For dense graphs, we can try to find large cliques.
    
    # Let's use a different approach: 
    # 1. Find all triangles first, group edges into triangles
    # 2. For remaining edges, try to form larger cliques
    
    # Actually, the simplest good approach:
    # - Find maximal cliques using a greedy algorithm
    # - Each edge must be in exactly one clique
    
    # Let's think differently. We want to partition edges into cliques minimizing count.
    # This is equivalent to: cover all edges with minimum number of cliques.
    
    # Greedy approach: 
    # 1. Sort edges
    # 2. For each edge, try to extend it into the largest possible clique using remaining edges
    # 3. Remove those edges from the pool
    
    remaining_edges = set(edges)
    cliques = []
    
    # Precompute for speed: for each vertex, the set of neighbors
    # We'll work with the remaining edge set
    
    while remaining_edges:
        if time.time() > deadline:
            # Return what we have plus remaining as individual edges
            for e in remaining_edges:
                cliques.append(list(e))
            break
        
        # Pick an edge, try to build a clique around it
        # Use the first edge in the set (or a heuristic choice)
        # For better results, pick an edge that can be part of the largest clique
        
        # Simple heuristic: pick an edge, then greedily add vertices
        # that are connected to all current clique members
        
        # Let's try: for each edge, compute how many common neighbors it has
        # Pick the edge with the most common neighbors (potential for large clique)
        
        # For efficiency with up to ~2000 edges, let's be smart
        # Actually for n=64, max edges ~2016, this could be slow if done naively
        
        # Let's use a simpler but effective approach:
        # Pick a vertex with high degree, build a clique from its neighborhood
        
        # Alternative: just greedily pick edges and extend
        # For each candidate edge, count potential clique size
        
        # Let's just pick the first edge and extend greedily
        # To be faster, we'll limit the search
        
        # Pick an edge - use a heuristic: pick edge with max common neighbors
        best_edge = None
        best_common = -1
        sample = list(remaining_edges)
        if len(sample) > 200:
            # Sample for speed
            import random
            rng2 = Random(42)
            sample = rng2.sample(sample, 200)
        
        for e in sample:
            u, v = e
            common = len(adj[u] & adj[v])
            # But we need common neighbors that are also connected in remaining edges
            # This is an upper bound, good enough for heuristic
            if common > best_common:
                best_common = common
                best_edge = e
        
        if best_edge is None:
            best_edge = next(iter(remaining_edges))
        
        u, v = best_edge
        # Build clique starting with {u, v}
        clique = [u, v]
        used_in_clique = {u, v}
        
        # Try to add vertices that are connected to all clique members
        # and the edges to those vertices are in remaining_edges
        candidates = adj[u] & adj[v]
        candidates = [w for w in candidates if w not in used_in_clique]
        
        # Sort candidates by degree in remaining graph (heuristic)
        def degree_in_remaining(w):
            count = 0
            for x in used_in_clique:
                if (min(x,w), max(x,w)) in remaining_edges:
                    count += 1
            return count
        
        candidates.sort(key=degree_in_remaining, reverse=True)
        
        for w in candidates:
            # Check if w can be added: w must be connected to all clique members
            # and all those edges must be in remaining_edges
            can_add = True
            for x in clique:
                e_key = (min(x, w), max(x, w))
                if e_key not in remaining_edges:
                    can_add = False
                    break
            if can_add:
                clique.append(w)
                used_in_clique.add(w)
        
        # Remove all edges within this clique from remaining
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                e_key = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                remaining_edges.discard(e_key)
        
        if len(clique) >= 2:
            cliques.append(clique)
    
    return cliques

def construct():
    deadline = time.time() + 100  # Safety margin before 120s limit
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        cliques = _solve_instance(n, p, seed, deadline)
        result.append(cliques)
        if time.time() > deadline - 10:
            # Not enough time for second instance, return edges for second
            edges = _random_graph(n, p, seed)
            result.append([list(e) for e in edges])
            break
    return result
