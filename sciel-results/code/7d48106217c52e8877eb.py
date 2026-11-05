import numpy as np
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Try to build a C4-free graph on n vertices (n up to 47) with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 145  # 145s out of 150s budget
    
    best_result = None
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple strategies and seeds
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            result = _try_construct_fast(n, required_min_deg, deadline, attempt)
            if result is not None:
                n_found, edges_found = result
                return {'n': n_found, 'edges': edges_found}
        
        # If not found with fast method, try local search with edge swaps
        result = _local_search_fast(n, required_min_deg, deadline)
        if result is not None:
            n_found, edges_found = result
            return {'n': n_found, 'edges': edges_found}
        
        # Track best partial result
        if best_result is None:
            best_result = (n, required_min_deg)
    
    # Fallback: return a valid smaller graph
    # For n=39, min degree >= 0, so empty graph works
    return {'n': 39, 'edges': []}


def _check_c4_fast(adj, u, v):
    """
    Check if adding edge (u,v) creates a C4.
    Uses set intersection for O(deg) performance.
    """
    # Get neighbor sets
    nu = set(np.where(adj[u])[0])
    nv = set(np.where(adj[v])[0])
    
    # Remove u and v from consideration
    nu.discard(v)
    nv.discard(u)
    
    # For each x in N(u) \ {v}, check if N(x) \ {u, v} intersects N(v)
    for x in nu:
        nx = set(np.where(adj[x])[0])
        nx.discard(u)
        nx.discard(v)
        if nx & nv:
            return True
    
    return False


def _try_construct_fast(n, required_min_deg, deadline, seed):
    """
    Greedy construction with efficient C4 checking.
    Uses structured starting point for some seeds.
    """
    random.seed(seed)
    rng = np.random.RandomState(seed)
    
    adj = np.zeros((n, n), dtype=bool)
    
    # For some seeds, use a structured starting point (polarity-based)
    if seed % 3 == 0:
        # Try a structured approach: partition into groups
        # This gives a high initial degree
        group_size = max(1, n // 5)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Connect vertices in different groups
                gi = i // group_size
                gj = j // group_size
                if gi != gj:
                    if not _check_c4_fast(adj, i, j):
                        adj[i, j] = True
                        adj[j, i] = True
    
    max_iterations = 10000
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        
        degrees = adj.sum(axis=1)
        min_deg = degrees.min()
        
        if min_deg >= required_min_deg:
            break
        
        # Find the best edge to add: one that doesn't create C4 and maximally helps
        # Try edges involving the minimum-degree vertex first
        min_deg_vertices = np.where(degrees == min_deg)[0]
        
        added = False
        # Shuffle to add randomness
        min_deg_list = min_deg_vertices.tolist()
        random.shuffle(min_deg_list)
        
        for u in min_deg_list:
            if time.time() > deadline:
                break
            
            # Try adding edges from u to other vertices, preferring those with low degree
            candidates = []
            for v in range(n):
                if v != u and not adj[u, v]:
                    candidates.append(v)
            
            # Sort candidates by degree (lower first)
            candidates.sort(key=lambda v: degrees[v])
            
            for v in candidates:
                if time.time() > deadline:
                    break
                
                if adj[u, v]:
                    continue
                
                if not _check_c4_fast(adj, u, v):
                    adj[u, v] = True
                    adj[v, u] = True
                    added = True
                    break
            
            if added:
                break
        
        if not added:
            # No more edges can be added without creating C4
            # Try a different random restart
            break
    
    degrees = adj.sum(axis=1)
    if degrees.min() >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return (n, edges)
    return None


def _local_search_fast(n, required_min_deg, deadline):
    """
    Local search: start with a greedy solution, then try edge swaps to improve min degree.
    """
    # First, get an initial solution
    for seed in range(10):
        result = _try_construct_fast(n, required_min_deg, deadline, seed)
        if result is not None:
            return result
    
    # Try to build from scratch with a different strategy:
    # Start with a high-degree subgraph and add edges carefully
    
    adj = np.zeros((n, n), dtype=bool)
    
    # Use a different heuristic: build a graph that is close to complete
    # but remove edges to avoid C4s
    
    # Start with complete graph and remove edges that create C4s
    # This is expensive, so use a heuristic instead
    
    # Try: start with a random sparse graph and add edges
    for seed in range(50):
        if time.time() > deadline:
            break
        
        random.seed(seed + 1000)
        adj = np.zeros((n, n), dtype=bool)
        
        # Random initial edges
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.3:
                    if not _check_c4_fast(adj, u, v):
                        adj[u, v] = True
                        adj[v, u] = True
        
        # Now try to add more edges to increase min degree
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            degrees = adj.sum(axis=1)
            min_deg = degrees.min()
            
            if min_deg >= required_min_deg:
                break
            
            min_deg_vertices = np.where(degrees == min_deg)[0]
            random.shuffle(min_deg_vertices.tolist())
            
            added = False
            for u in min_deg_vertices:
                if time.time() > deadline:
                    break
                
                candidates = [v for v in range(n) if v != u and not adj[u, v]]
                candidates.sort(key=lambda v: degrees[v])
                
                for v in candidates:
                    if time.time() > deadline:
                        break
                    if not _check_c4_fast(adj, u, v):
                        adj[u, v] = True
                        adj[v, u] = True
                        added = True
                        break
                
                if added:
                    break
            
            if not added:
                break
        
        degrees = adj.sum(axis=1)
        if degrees.min() >= required_min_deg:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        edges.append([u, v])
            return (n, edges)
    
    return None

# EVOLVE-BLOCK-END
