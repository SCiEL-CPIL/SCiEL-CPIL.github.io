"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
import math
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

def _soft_max_triangle_sum(adj, bitmask, deg, n, beta=0.5):
    """Compute a smooth surrogate for max triangle degree sum using log-sum-exp."""
    # Collect all triangle degree sums
    sums = []
    for u in range(n):
        for v in adj[u]:
            if v > u:
                common_mask = bitmask[u] & bitmask[v]
                for w in range(v + 1, n):
                    if common_mask & (1 << w):
                        s = deg[u] + deg[v] + deg[w]
                        sums.append(s)
    
    if not sums:
        return 0.0
    
    max_s = max(sums)
    # Log-sum-exp trick for numerical stability
    exp_sums = [math.exp((s - max_s) / beta) for s in sums]
    log_sum = math.log(sum(exp_sums))
    return max_s + beta * log_sum

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

def _quick_score_delta(current_es, remove_edge, add_edge, adj, bitmask, deg, n):
    """Estimate score change from an edge swap without full rebuild."""
    # This is an approximation - we'll still verify with full rebuild
    # but use this for faster rejection of obviously bad moves
    u1, v1 = remove_edge
    u2, v2 = add_edge
    
    # Estimate degree changes
    deg_change = [0] * n
    deg_change[u1] -= 1
    deg_change[v1] -= 1
    deg_change[u2] += 1
    deg_change[v2] += 1
    
    # Rough estimate: if we're removing an edge from high-degree vertices
    # and adding to low-degree, it's likely good
    remove_deg_sum = deg[u1] + deg[v1]
    add_deg_sum = (deg[u2] + 1) + (deg[v2] + 1)
    
    # This is just a heuristic filter, not exact
    return add_deg_sum - remove_deg_sum

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
    max_no_improve = 200
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        # Try random swaps with targeted removal
        # First, identify worst triangles
        worst_triangles = []
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    common_mask = bitmask[u] & bitmask[v]
                    for w in range(v + 1, n):
                        if common_mask & (1 << w):
                            s = deg[u] + deg[v] + deg[w]
                            worst_triangles.append((s, (u, v), (u, w), (v, w)))
        
        worst_triangles.sort(reverse=True)
        
        for _ in range(300):
            if time.time() > deadline:
                break
            
            # 60% targeted, 40% random
            if worst_triangles and rng.random() < 0.6:
                # Pick from top 5 worst triangles
                top_k = min(5, len(worst_triangles))
                _, e1, e2, e3 = rng.choice(worst_triangles[:top_k])
                candidates = [(min(a,b),max(a,b)) for a,b in [e1,e2,e3] if (min(a,b),max(a,b)) in current_es]
                if candidates:
                    remove_edge = rng.choice(candidates)
                else:
                    remove_edge = rng.choice(list(current_es))
            else:
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
                # Update adj, bitmask, deg for next iteration
                adj = new_adj
                bitmask = new_bitmask
                deg = [bitmask[u].bit_count() for u in range(n)]
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
    """SA over edge swaps with soft-max surrogate and targeted move selection."""
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
    T = 3.0
    T_min = 0.02
    cooling = 0.9997
    
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
    
    # Track soft-max objective for smoother acceptance
    current_soft = _soft_max_triangle_sum(adj, bitmask, deg, n, beta=0.8)
    
    while time.time() < deadline:
        T = max(T * cooling, T_min)
        
        # Get worst triangles for targeted removal
        worst_triangles = _get_worst_triangles(adj, bitmask, n, deg)
        
        # 75% chance: targeted move (remove edge from worst triangle)
        # 25% chance: random swap
        if worst_triangles and rng.random() < 0.75:
            # Pick from top 8 worst triangles
            top_k = min(8, len(worst_triangles))
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
        new_soft = _soft_max_triangle_sum(new_adj, new_bitmask, new_deg, n, beta=0.8)
        
        # Use soft-max delta for smoother acceptance
        delta = new_soft - current_soft
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / max(T, 0.01)):
            current_max = new_max
            current_soft = new_soft
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

def _polish_pass(edges, n, time_limit, rng):
    """High-precision polish: iteratively try to eliminate the worst triangle."""
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
    
    while time.time() < deadline:
        # Find the worst triangle
        worst_tri = None
        worst_sum = 0
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    common_mask = bitmask[u] & bitmask[v]
                    for w in range(v + 1, n):
                        if common_mask & (1 << w):
                            s = deg[u] + deg[v] + deg[w]
                            if s > worst_sum:
                                worst_sum = s
                                worst_tri = (u, v, w)
        
        if worst_tri is None or worst_sum <= 22:
            break
        
        tu, tv, tw = worst_tri
        # Try to break this triangle by removing one of its edges
        # and adding an edge that doesn't create a worse triangle
        candidates = [(tu, tv), (tu, tw), (tv, tw)]
        candidates = [c for c in candidates if (min(c[0],c[1]),max(c[0],c[1])) in current_es]
        
        if not candidates:
            break
        
        found_improvement = False
        for remove_edge in candidates:
            if time.time() > deadline:
                break
            # Try adding different edges
            for add_edge in rng.sample(all_edges, min(50, len(all_edges))):
                if add_edge in current_es:
                    continue
                
                current_es.discard(remove_edge)
                current_es.add(add_edge)
                
                new_adj, new_bitmask = _build_adj_and_bitmask(list(current_es), n)
                new_max, new_deg = _score_with_bitmask(new_adj, new_bitmask, n)
                
                if new_max < best_score:
                    best_score = new_max
                    best_edges = [[u, v] for (u, v) in sorted(current_es)]
                    adj = new_adj
                    bitmask = new_bitmask
                    deg = new_deg
                    found_improvement = True
                    break
                else:
                    current_es.discard(add_edge)
                    current_es.add(remove_edge)
            
            if found_improvement:
                break
        
        if not found_improvement:
            # No improvement found for this worst triangle, try a few more random swaps
            for _ in range(20):
                if time.time() > deadline:
                    break
                remove_edge = rng.choice(list(current_es))
                add_edge = rng.choice(all_edges)
                if add_edge in current_es:
                    continue
                
                current_es.discard(remove_edge)
                current_es.add(add_edge)
                
                new_adj, new_bitmask = _build_adj_and_bitmask(list(current_es), n)
                new_max, new_deg = _score_with_bitmask(new_adj, new_bitmask, n)
                
                if new_max < best_score:
                    best_score = new_max
                    best_edges = [[u, v] for (u, v) in sorted(current_es)]
                    adj = new_adj
                    bitmask = new_bitmask
                    deg = new_deg
                    found_improvement = True
                    break
                else:
                    current_es.discard(add_edge)
                    current_es.add(remove_edge)
            
            if not found_improvement:
                break
    
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
    edges, score = _local_search(start_edges, n, 20, rng)
    if score < best_score:
        best_score = score
        best_edges = edges
    
    # Starting point 2: K_{8,8} + edge(0,7) (different intra-edge)
    start_edges2 = []
    for u in range(8):
        for v in range(8, 16):
            start_edges2.append([u, v])
    start_edges2.append([0, 7])
    if time.time() < deadline - 55:
        edges, score = _local_search(start_edges2, n, 20, rng)
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Starting point 3: K_{8,8} + edge(3,5)
    start_edges3 = []
    for u in range(8):
        for v in range(8, 16):
            start_edges3.append([u, v])
    start_edges3.append([3, 5])
    if time.time() < deadline - 45:
        edges, score = _local_search(start_edges3, n, 15, rng)
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Phase 2: SA from best found
    if time.time() < deadline - 35:
        edges, score = _simulated_annealing(best_edges, n, 30, rng)
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Phase 3: Polish pass with targeted triangle elimination
    if time.time() < deadline - 15:
        edges, score = _polish_pass(best_edges, n, 12, rng)
        if score < best_score:
            best_score = score
            best_edges = edges
    
    # Phase 4: Final quick local search
    if time.time() < deadline - 5:
        edges, score = _local_search(best_edges, n, 8, rng)
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
