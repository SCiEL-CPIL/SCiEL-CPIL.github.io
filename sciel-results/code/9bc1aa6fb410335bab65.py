"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def _compute_scores(adj, n):
    """Compute degrees and max triangle degree sum."""
    deg = [0] * n
    for u in range(n):
        for v in adj[u]:
            deg[u] += 1
    # Find all triangles and compute max degree sum
    max_sum = 0
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
    return max_sum, deg

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_set(edges):
    es = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        es.add((u, v))
    return es

def _base_graph(n=16):
    """K_{8,8} plus one edge (0,1) within part A."""
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    edges.append([0, 1])
    return edges

def _local_search(edges, n, time_limit, rng):
    """Local search via edge swaps to minimize max triangle degree sum."""
    adj = _build_adj(edges, n)
    es = _edge_set(edges)
    max_sum, deg = _compute_scores(adj, n)
    best_edges = list(edges)
    best_score = max_sum
    
    # Precompute all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u+1, n):
            all_edges.append((u, v))
    
    # For speed, categorize edges
    # Current edge set
    current_es = set(es)
    
    deadline = time.time() + time_limit
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        # Try random swaps
        for _ in range(200):
            if time.time() > deadline:
                break
            # Pick a random edge to remove
            remove_edge = rng.choice(list(current_es))
            ru, rv = remove_edge
            # Pick a random edge to add (not already present)
            add_edge = rng.choice(all_edges)
            au, av = add_edge
            if add_edge in current_es:
                continue
            # Apply swap
            current_es.discard(remove_edge)
            current_es.add(add_edge)
            
            # Rebuild adj and compute score
            new_adj = [set() for _ in range(n)]
            for (u, v) in current_es:
                new_adj[u].add(v)
                new_adj[v].add(u)
            new_max, new_deg = _compute_scores(new_adj, n)
            
            if new_max < best_score:
                best_score = new_max
                best_edges = [[u, v] for (u, v) in sorted(current_es)]
                max_sum = new_max
                improved = True
                no_improve_count = 0
            else:
                # Revert
                current_es.discard(add_edge)
                current_es.add(remove_edge)
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
    
    return best_edges, best_score

def _simulated_annealing(edges, n, time_limit, rng):
    """SA over edge swaps."""
    adj = _build_adj(edges, n)
    current_es = _edge_set(edges)
    current_max, _ = _compute_scores(adj, n)
    best_edges = list(edges)
    best_score = current_max
    
    all_edges = []
    for u in range(n):
        for v in range(u+1, n):
            all_edges.append((u, v))
    
    deadline = time.time() + time_limit
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline:
        T = max(T * cooling, T_min)
        
        # Pick a random swap
        remove_edge = rng.choice(list(current_es))
        ru, rv = remove_edge
        add_edge = rng.choice(all_edges)
        au, av = add_edge
        if add_edge in current_es:
            continue
        
        # Apply
        current_es.discard(remove_edge)
        current_es.add(add_edge)
        
        new_adj = [set() for _ in range(n)]
        for (u, v) in current_es:
            new_adj[u].add(v)
            new_adj[v].add(u)
        new_max, _ = _compute_scores(new_adj, n)
        
        delta = new_max - current_max
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / max(T, 0.01)):
            current_max = new_max
            if new_max < best_score:
                best_score = new_max
                best_edges = [[u, v] for (u, v) in sorted(current_es)]
        else:
            # Revert
            current_es.discard(add_edge)
            current_es.add(remove_edge)
    
    return best_edges, best_score

def construct():
    n = 16
    rng = random.Random(42)
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Phase 1: Local search from multiple starting points
    best_edges = None
    best_score = float('inf')
    
    # Starting point 1: K_{8,8} + edge(0,1)
    start_edges = _base_graph(n)
    t0 = time.time()
    edges, score = _local_search(start_edges, n, 30, rng)
    if score < best_score:
        best_score = score
        best_edges = edges
    
    # Starting point 2: K_{8,8} + edge(0,7) (different intra-edge)
    start_edges2 = []
    for u in range(8):
        for v in range(8, 16):
            start_edges2.append([u, v])
    start_edges2.append([0, 7])
    t1 = time.time()
    if time.time() < deadline - 60:
        edges, score = _local_search(start_edges2, n, 30, rng)
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Phase 2: SA from best found
    if time.time() < deadline - 30:
        edges, score = _simulated_annealing(best_edges, n, 40, rng)
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Phase 3: More local search from best
    if time.time() < deadline - 10:
        edges, score = _local_search(best_edges, n, 20, rng)
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Final verification
    if best_edges is None:
        best_edges = _base_graph(n)
    
    # Ensure exactly 65 edges
    assert len(best_edges) == 65, f"Expected 65 edges, got {len(best_edges)}"
    
    # Ensure no duplicate edges
    es = _edge_set(best_edges)
    assert len(es) == 65, f"Duplicate edges found: {len(es)} unique vs 65"
    
    return {'n': n, 'edges': best_edges}

# EVOLVE-BLOCK-END
