"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def _build_adj_and_bitmask(edges, n):
    """Build adjacency sets and bitmask representation."""
    adj = [set() for _ in range(n)]
    bitmask = [0] * n
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        bitmask[u] |= (1 << v)
        bitmask[v] |= (1 << u)
    return adj, bitmask

def _score_with_bitmask(adj, bitmask, n):
    """Compute degrees and max triangle degree sum using bitmask for speed."""
    deg = [0] * n
    for u in range(n):
        deg[u] = bitmask[u].bit_count()
    
    max_sum = 0
    for u in range(n):
        for v in adj[u]:
            if v > u:
                common_mask = bitmask[u] & bitmask[v]
                # Only consider w > v to avoid duplicates
                for w in range(v + 1, n):
                    if common_mask & (1 << w):
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
    return max_sum, deg

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
    adj, bitmask = _build_adj_and_bitmask(edges, n)
    max_sum, deg = _score_with_bitmask(adj, bitmask, n)
    best_edges = list(edges)
    best_score = max_sum
    
    # Precompute all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u+1, n):
            all_edges.append((u, v))
    
    current_es = set(_edge_set(edges))
    
    deadline = time.time() + time_limit
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        # Try random swaps
        for _ in range(500):
            if time.time() > deadline:
                break
            # Pick a random edge to remove
            remove_edge = rng.choice(list(current_es))
            # Pick a random edge to add (not already present)
            add_edge = rng.choice(all_edges)
            if add_edge in current_es:
                continue
            
            # Apply swap
            current_es.discard(remove_edge)
            current_es.add(add_edge)
            
            # Rebuild adj and compute score
            new_adj, new_bitmask = _build_adj_and_bitmask(list(current_es), n)
            new_max, _ = _score_with_bitmask(new_adj, new_bitmask, n)
            
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
    """SA over edge swaps with targeted move selection."""
    adj, bitmask = _build_adj_and_bitmask(edges, n)
    current_es = set(_edge_set(edges))
    current_max, deg = _score_with_bitmask(adj, bitmask, n)
    best_edges = list(edges)
    best_score = current_max
    
    all_edges = []
    for u in range(n):
        for v in range(u+1, n):
            all_edges.append((u, v))
    
    deadline = time.time() + time_limit
    T = 5.0
    T_min = 0.05
    cooling = 0.9995
    
    # Precompute worst triangles for targeted moves
    def _get_worst_triangles(adj, bitmask, n, deg):
        """Return list of (u,v,w) triangles sorted by degree sum descending."""
        triangles = []
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    common_mask = bitmask[u] & bitmask[v]
                    for w in range(v + 1, n):
                        if common_mask & (1 << w):
                            s = deg[u] + deg[v] + deg[w]
                            triangles.append((s, u, v, w))
        triangles.sort(reverse=True)
        return triangles
    
    while time.time() < deadline:
        T = max(T * cooling, T_min)
        
        # Get worst triangles for targeted removal
        worst_triangles = _get_worst_triangles(adj, bitmask, n, deg)
        
        # 70% chance: targeted move (remove edge from worst triangle)
        # 30% chance: random swap
        if worst_triangles and rng.random() < 0.7:
            # Pick from top 10 worst triangles
            top_k = min(10, len(worst_triangles))
            _, tu, tv, tw = rng.choice(worst_triangles[:top_k])
            # Remove one of the edges in this triangle
            candidates = [(tu, tv), (tu, tw), (tv, tw)]
            candidates = [c for c in candidates if c in current_es]
            if not candidates:
                candidates = [(min(tu,tv),max(tu,tv)), (min(tu,tw),max(tu,tw)), (min(tv,tw),max(tv,tw))]
                candidates = [c for c in candidates if c in current_es]
            if not candidates:
                remove_edge = rng.choice(list(current_es))
            else:
                remove_edge = rng.choice(candidates)
        else:
            remove_edge = rng.choice(list(current_es))
        
        add_edge = rng.choice(all_edges)
        if add_edge in current_es:
            continue
        
        # Apply
        current_es.discard(remove_edge)
        current_es.add(add_edge)
        
        new_adj, new_bitmask = _build_adj_and_bitmask(list(current_es), n)
        new_max, new_deg = _score_with_bitmask(new_adj, new_bitmask, n)
        
        delta = new_max - current_max
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / max(T, 0.01)):
            current_max = new_max
            adj = new_adj
            bitmask = new_bitmask
            deg = new_deg
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
    
    deadline = time.time() + 135  # Safety margin before 150s limit
    
    # Phase 1: Local search from multiple starting points
    best_edges = None
    best_score = float('inf')
    
    # Starting point 1: K_{8,8} + edge(0,1)
    start_edges = _base_graph(n)
    edges, score = _local_search(start_edges, n, 25, rng)
    if score < best_score:
        best_score = score
        best_edges = edges
    
    # Starting point 2: K_{8,8} + edge(0,7) (different intra-edge)
    start_edges2 = []
    for u in range(8):
        for v in range(8, 16):
            start_edges2.append([u, v])
    start_edges2.append([0, 7])
    if time.time() < deadline - 60:
        edges, score = _local_search(start_edges2, n, 25, rng)
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Starting point 3: K_{8,8} + edge(3,5)
    start_edges3 = []
    for u in range(8):
        for v in range(8, 16):
            start_edges3.append([u, v])
    start_edges3.append([3, 5])
    if time.time() < deadline - 50:
        edges, score = _local_search(start_edges3, n, 20, rng)
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
        edges, score = _local_search(best_edges, n, 15, rng)
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
