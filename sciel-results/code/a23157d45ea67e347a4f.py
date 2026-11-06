import time
import random
import numpy as np
from itertools import combinations
from collections import defaultdict

def construct():
    # Define the two instances
    instances = [
        (32, 0.6, 601),
        (64, 0.55, 602)
    ]
    
    results = []
    total_start_time = time.time()
    total_time_limit = 140.0  # Leave 10s safety margin for the evaluator
    
    for idx, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - total_start_time
        remaining_time = total_time_limit - elapsed
        
        # If we don't have enough time, just do the trivial solution
        if remaining_time < 1.0:
            cliques = trivial_partition(n, p, seed)
        else:
            # Allocate time proportionally to problem size, but with a minimum
            # Give at least 5s to each instance
            if idx == 0:
                instance_time_limit = min(remaining_time * 0.45, 50.0)
            else:
                instance_time_limit = min(remaining_time * 0.55, 85.0)
            
            cliques = solve_instance(n, p, seed, instance_time_limit)
            
        results.append(cliques)
        
    return results

def trivial_partition(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append([u, v])
    return edges

def generate_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def solve_instance(n, p, seed, time_limit):
    edges = generate_edges(n, p, seed)
    if not edges:
        return []
        
    # Build adjacency list for residual graph
    # res_adj[u] is a set of neighbors v such that (u,v) is in residual graph
    res_adj = {i: set() for i in range(n)}
    for u, v in edges:
        res_adj[u].add(v)
        res_adj[v].add(u)
        
    cliques = []
    start_time = time.time()
    
    # Track remaining edges count to stop early
    remaining_edges = len(edges)
    
    while remaining_edges > 0:
        current_time = time.time()
        if current_time - start_time > time_limit:
            # Timeout: fall back to trivial for remaining edges
            # Collect all remaining edges
            for u in range(n):
                for v in res_adj[u]:
                    if u < v:
                        cliques.append([u, v])
                        remaining_edges -= 1
            break
        
        # Find a large clique
        clique = find_large_clique(res_adj, n)
        
        if not clique:
            # Should not happen if edges exist, but safety check
            # Find any edge
            found = False
            for u in range(n):
                for v in res_adj[u]:
                    if u < v:
                        clique = [u, v]
                        found = True
                        break
                if found: break
            if not found:
                break
                
        # Remove edges in clique
        clique_edges_removed = 0
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                if v in res_adj[u]:
                    res_adj[u].discard(v)
                    res_adj[v].discard(u)
                    clique_edges_removed += 1
                    remaining_edges -= 1
                
        if clique_edges_removed > 0:
            cliques.append(clique)
        
    return cliques

def find_large_clique(res_adj, n):
    # Heuristic: Try greedy expansion from top K highest degree vertices
    # Calculate degrees
    degrees = {u: len(res_adj[u]) for u in range(n)}
    
    # Sort vertices by degree
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    best_clique = []
    
    # Try top K vertices
    K = min(30, n) # Try more starting points for better coverage
    
    for start_v in sorted_vertices[:K]:
        if degrees[start_v] == 0:
            continue
            
        # Greedy expansion
        C = [start_v]
        P = set(res_adj[start_v])
        
        while P:
            # Pick u in P with max degree in P
            best_u = None
            best_deg = -1
            for u in P:
                # Degree of u in P is |P & res_adj[u]|
                deg = len(P & res_adj[u])
                if deg > best_deg:
                    best_deg = deg
                    best_u = u
            if best_u is None:
                break
            C.append(best_u)
            P = P & res_adj[best_u]
            
        if len(C) > len(best_clique):
            best_clique = C
            
    return best_clique
