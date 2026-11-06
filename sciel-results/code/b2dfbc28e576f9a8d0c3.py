"""Rewritten program: local search + simulated annealing to minimize max triangle degree sum."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _compute_score(n, adj, edge_set):
    """Compute the maximum degree sum over all triangles."""
    max_sum = 0
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            common = adj[u] & adj[v]
            for w in common:
                if w <= v:
                    continue
                s = len(adj[u]) + len(adj[v]) + len(adj[w])
                if s > max_sum:
                    max_sum = s
    return max_sum

def _find_all_triangles(n, adj):
    """Return list of triangles as tuples (u,v,w) with u<v<w."""
    triangles = []
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            common = adj[u] & adj[v]
            for w in common:
                if w > v:
                    triangles.append((u, v, w))
    return triangles

def _initial_k88_plus_one(n=16):
    """K_{8,8} plus one internal edge: 64 cross edges + 1 = 65 edges."""
    edges = [[u, v] for u in range(8) for v in range(8, 16)]
    edges.append([0, 1])
    return edges

def _local_search(n, edges, max_time=100.0, max_iters=200000):
    """Local search: try adding/removing edges to reduce max triangle degree sum."""
    start = time.time()
    adj = _build_adjacency(n, edges)
    edge_set = set((min(u,v), max(u,v)) for u, v in edges)
    current_score = _compute_score(n, adj, edge_set)
    best_score = current_score
    best_edges = list(edges)
    
    # Precompute all possible edges
    all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Triangles with their degree sums
    def get_triangles_and_sums(adj):
        tris = []
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                common = adj[u] & adj[v]
                for w in common:
                    if w > v:
                        s = len(adj[u]) + len(adj[v]) + len(adj[w])
                        tris.append((u, v, w, s))
        return tris
    
    tris = get_triangles_and_sums(adj)
    current_max = max((t[3] for t in tris), default=0)
    
    iters = 0
    no_improve = 0
    
    while iters < max_iters:
        if time.time() - start > max_time:
            break
        iters += 1
        
        # Find the worst triangle
        if not tris:
            break
        worst_idx = max(range(len(tris)), key=lambda i: tris[i][3])
        worst_tri = tris[worst_idx]
        worst_sum = worst_tri[3]
        
        if worst_sum <= best_score:
            # Try to improve further
            pass
        
        # Strategy: try to reduce degree of one vertex in the worst triangle
        # by removing an edge incident to that vertex (not in the triangle)
        # or by moving an edge
        improved = False
        
        # Try removing an edge incident to a high-degree vertex in the worst triangle
        for vi in range(3):
            v = worst_tri[vi]
            if len(adj[v]) < 2:
                continue
            # Find edges to remove from v (not in the triangle)
            tri_verts = set(worst_tri[:3])
            candidates = [u for u in adj[v] if u not in tri_verts]
            random.shuffle(candidates)
            for u in candidates:
                # Check: removing edge (u,v) - does it keep exactly 65 edges? No, we need 65.
                # So we need to swap: remove one edge, add another.
                # Try swapping: remove (u,v), add (a,b) where a,b are non-adjacent
                # and (a,b) is not (u,v)
                # For simplicity, try: remove (u,v), add a random non-edge
                # But we need to check the score doesn't increase
                # This is complex; let's try a simpler approach:
                # Just try to find a swap that reduces the score
                break
            # Simpler: try all swaps involving v
            # Remove (v, u) for u in adj[v], add (v, w) for w not in adj[v]
            # or remove (v,u) add (a,b) unrelated
            # Let's try: for each neighbor u of v (not in triangle), 
            #   for each non-neighbor w of v, try swap (v,u)->(v,w)
            non_neighbors = [w for w in range(n) if w != v and w not in adj[v]]
            random.shuffle(non_neighbors)
            for u in candidates[:5]:  # limit for speed
                for w in non_neighbors[:5]:
                    if u == w:
                        continue
                    # Swap: remove (v,u), add (v,w)
                    adj[v].discard(u)
                    adj[u].discard(v)
                    edge_set.discard((min(v,u), max(v,u)))
                    adj[v].add(w)
                    adj[w].add(v)
                    edge_set.add((min(v,w), max(v,w)))
                    
                    new_tris = get_triangles_and_sums(adj)
                    new_max = max((t[3] for t in new_tris), default=0)
                    
                    if new_max < worst_sum:
                        # Accept
                        tris = new_tris
                        worst_sum = new_max
                        if new_max < best_score:
                            best_score = new_max
                            best_edges = [[min(a,b), max(a,b)] for a, b in edge_set]
                        improved = True
                        no_improve = 0
                        break
                    else:
                        # Revert
                        adj[v].discard(w)
                        adj[w].discard(v)
                        edge_set.discard((min(v,w), max(v,w)))
                        adj[v].add(u)
                        adj[u].add(v)
                        edge_set.add((min(v,u), max(v,u)))
                
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
            if no_improve > 5000:
                break
    
    return best_score, best_edges

def _simulated_annealing(n, edges, max_time=80.0, max_iters=500000):
    """Simulated annealing to minimize max triangle degree sum."""
    start = time.time()
    adj = _build_adjacency(n, edges)
    edge_set = set((min(u,v), max(u,v)) for u, v in edges)
    
    def get_max_tri_sum(adj):
        max_sum = 0
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                common = adj[u] & adj[v]
                for w in common:
                    if w > v:
                        s = len(adj[u]) + len(adj[v]) + len(adj[w])
                        if s > max_sum:
                            max_sum = s
        return max_sum
    
    current_score = get_max_tri_sum(adj)
    best_score = current_score
    best_edges = [[min(u,v), max(u,v)] for u, v in edge_set]
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
    possible_set = set(all_possible)
    
    iters = 0
    while iters < max_iters:
        if time.time() - start > max_time:
            break
        iters += 1
        
        # Generate a random move: swap an edge (remove one, add one)
        if random.random() < 0.3:
            # Remove a random existing edge, add a random non-edge
            if not edge_set:
                continue
            rem_u, rem_v = random.choice(list(edge_set))
            # Find a non-edge to add
            added = False
            for _ in range(50):
                add_u, add_v = random.choice(all_possible)
                if (add_u, add_v) not in edge_set:
                    # Apply swap
                    adj[rem_u].discard(rem_v)
                    adj[rem_v].discard(rem_u)
                    edge_set.discard((rem_u, rem_v))
                    adj[add_u].add(add_v)
                    adj[add_v].add(add_u)
                    edge_set.add((add_u, add_v))
                    added = True
                    break
            if not added:
                continue
        else:
            # Move one endpoint of an edge
            if not edge_set:
                continue
            rem_u, rem_v = random.choice(list(edge_set))
            new_v = random.randint(0, n-1)
            if new_v == rem_u or new_v == rem_v:
                continue
            if (rem_u, new_v) in edge_set or (new_v, rem_u) in edge_set:
                continue
            # Swap (rem_u, rem_v) -> (rem_u, new_v)
            adj[rem_u].discard(rem_v)
            adj[rem_v].discard(rem_u)
            edge_set.discard((rem_u, rem_v))
            adj[rem_u].add(new_v)
            adj[new_v].add(rem_u)
            edge_set.add((rem_u, new_v))
        
        new_score = get_max_tri_sum(adj)
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = [[min(u,v), max(u,v)] for u, v in edge_set]
        
        T = max(T * cooling, T_min)
    
    return best_score, best_edges

def construct():
    n = 16
    deadline = time.time() + 140.0  # 140s deadline (10s safety margin)
    
    # Start with K_{8,8} + one edge
    edges = _initial_k88_plus_one(n)
    
    # Phase 1: Local search
    ls_start = time.time()
    score, edges = _local_search(n, edges, max_time=min(40.0, deadline - ls_start - 10))
    
    # Phase 2: Simulated annealing
    sa_start = time.time()
    remaining = deadline - sa_start - 5
    if remaining > 5:
        score, edges = _simulated_annealing(n, edges, max_time=min(remaining, 90.0))
    
    # Verify: exactly 65 edges
    if len(edges) != 65:
        # Fallback to initial
        edges = _initial_k88_plus_one(n)
    
    return {'n': 16, 'edges': edges}

# EVOLVE-BLOCK-END
