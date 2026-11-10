import time
import random

_T0 = time.time()
_DEADLINE = _T0 + 145.0  # 5s safety margin

def _random_graph(n, p, seed):
    """Reproduce the verifier's edge list exactly."""
    rng = random.Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

def _build_adjacency(edges, n):
    """Build adjacency list as bitmasks for fast operations."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_clique(vertices, adj):
    """Check if a set of vertices forms a clique using bitmasks."""
    if not vertices:
        return True
    # Convert vertices to a bitmask
    mask = 0
    for v in vertices:
        mask |= (1 << v)
    
    for v in vertices:
        # All other vertices in the clique must be neighbors of v
        # adj[v] contains v's neighbors. We need to check if (mask without v) is a subset of adj[v]
        if (mask & ~(1 << v)) & ~adj[v]:
            return False
    return True

def _greedy_partition(edges, n, adj, rng, time_limit):
    """
    Greedy clique partition with randomized vertex selection.
    """
    if not edges:
        return []
    
    # remaining edges as a set of (u, v) with u < v
    remaining = set((min(u, v), max(u, v)) for u, v in edges)
    cliques = []
    
    # Current degree of each vertex in the remaining graph
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    start_time = time.time()
    
    while remaining:
        if time.time() > time_limit:
            # Fallback: emit remaining edges as 2-cliques
            for e in remaining:
                cliques.append([e[0], e[1]])
            return cliques
        
        # Find a vertex with high degree to start the clique
        # Pick randomly from the top 5 highest degree vertices to introduce randomness
        # We need vertices that are actually in the remaining graph (deg > 0)
        active_vertices = [v for v in range(n) if deg[v] > 0]
        if not active_vertices:
            break
            
        # Sort by degree descending, take top 5 (or all if < 5)
        active_vertices.sort(key=lambda v: deg[v], reverse=True)
        top_k = active_vertices[:min(5, len(active_vertices))]
        start_v = rng.choice(top_k)
        
        # Build initial clique
        C = [start_v]
        
        # Candidates are neighbors of start_v in the remaining graph
        # We need to find neighbors of start_v such that the edge (start_v, u) is in remaining
        # Since deg[start_v] > 0, there is at least one.
        candidates = set()
        for u, v in remaining:
            if u == start_v:
                candidates.add(v)
            elif v == start_v:
                candidates.add(u)
        
        # Greedily extend the clique
        while candidates:
            # Find candidate with max degree within the candidate set
            # We need to count how many edges exist between candidate u and other candidates
            # This is expensive if done naively.
            # Optimization: Since the graph is dense, we can just pick the one with highest global degree 
            # among candidates, or use a simpler heuristic.
            
            # Let's use a simple heuristic: pick candidate u that maximizes the number of common neighbors 
            # with the current clique? No, just max degree in the remaining graph is a good proxy.
            
            best_u = -1
            best_deg = -1
            # To add randomness, we can shuffle candidates if degrees are equal
            # Or just pick max deg, break ties randomly
            max_deg_in_cand = -1
            best_cands = []
            
            for u in candidates:
                if deg[u] > max_deg_in_cand:
                    max_deg_in_cand = deg[u]
                    best_cands = [u]
                elif deg[u] == max_deg_in_cand:
                    best_cands.append(u)
            
            if not best_cands:
                break
                
            best_u = rng.choice(best_cands)
            C.append(best_u)
            
            # Update candidates: keep only those connected to best_u in remaining graph
            new_candidates = set()
            for w in candidates:
                if w == best_u:
                    continue
                # Check if edge (best_u, w) is in remaining
                e = (min(best_u, w), max(best_u, w))
                if e in remaining:
                    new_candidates.add(w)
            candidates = new_candidates
            
            # Optional: Stop if clique size is large enough? 
            # No, try to maximize.
            
        # Ensure C is a clique (it should be by construction, but verify just in case)
        if len(C) >= 2 and _is_clique(C, adj):
            cliques.append(C)
            # Remove all edges within C from remaining
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, w = C[i], C[j]
                    e = (min(u, w), max(u, w))
                    remaining.discard(e)
                    deg[u] -= 1
                    deg[w] -= 1
        else:
            # Fallback: take one edge
            if remaining:
                u, w = next(iter(remaining))
                cliques.append([u, w])
                remaining.discard((min(u, w), max(u, w)))
                deg[u] -= 1
                deg[w] -= 1

    return cliques

def _local_improve(cliques, adj, n, time_limit):
    """
    Try to merge cliques if their union is also a clique.
    """
    changed = True
    start_time = time.time()
    while changed and time.time() < time_limit:
        changed = False
        new_cliques = []
        used = [False] * len(cliques)
        
        for i in range(len(cliques)):
            if used[i]:
                continue
            Ci = cliques[i]
            merged = False
            # Try to merge Ci with any other clique Cj
            for j in range(i + 1, len(cliques)):
                if used[j]:
                    continue
                Cj = cliques[j]
                # Check if Ci U Cj is a clique
                # Optimization: Check if all vertices in Cj are connected to all vertices in Ci
                # and Cj is a clique (it is) and Ci is a clique (it is)
                # We just need to check cross edges.
                is_mergeable = True
                for u in Cj:
                    for v in Ci:
                        if not (adj[u] & (1 << v)):
                            is_mergeable = False
                            break
                    if not is_mergeable:
                        break
                
                if is_mergeable:
                    # Merge
                    new_cliques.append(Ci + Cj)
                    used[i] = True
                    used[j] = True
                    changed = True
                    merged = True
                    break
            
            if not merged:
                new_cliques.append(Ci)
        
        cliques = new_cliques
        
    return cliques

def _solve_instance(edges, n, time_limit):
    if not edges:
        return []
        
    adj = _build_adjacency(edges, n)
    
    best_cliques = None
    best_count = float('inf')
    
    # Run multiple restarts
    # Estimate number of restarts based on time
    # Each restart takes roughly O(E * K^2) where K is clique size
    # Let's just loop until time runs out
    
    rng_seed = 42
    restart = 0
    
    while time.time() < time_limit:
        rng = random.Random(rng_seed + restart)
        
        # Allocate some time for this restart
        # If we have plenty of time, give it more. If little, give it little.
        remaining_time = time_limit - time.time()
        if remaining_time < 1.0:
            break
            
        # Give each restart a slice of time, but ensure we can finish
        # For small graphs, restarts are fast.
        # Let's just run one restart and see how long it takes.
        # To be safe, we'll run a restart with a sub-limit.
        
        # Simple strategy: Run restart with 50ms limit if we have < 5s, else 200ms limit?
        # Actually, just run the greedy partition. It's fast.
        # The local improve is also fast.
        
        cliques = _greedy_partition(edges, n, adj, rng, time_limit)
        
        if not cliques:
            continue
            
        # Local improvement
        cliques = _local_improve(cliques, adj, n, time_limit)
        
        # Validate
        if _validate_partition(cliques, edges, adj):
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques
                
        restart += 1
        
    if best_cliques is None:
        # Fallback: single greedy pass without local improve or strict validation
        rng = random.Random(0)
        best_cliques = _greedy_partition(edges, n, adj, rng, time_limit)
        
    return best_cliques

def _validate_partition(cliques, edges, adj):
    """
    Check if the partition is valid:
    1. Each clique is a clique in the graph.
    2. Every edge is covered exactly once.
    """
    if not cliques:
        return len(edges) == 0
        
    seen_edges = set()
    for C in cliques:
        if len(C) < 2:
            return False
        # Check if C is a clique
        if not _is_clique(C, adj):
            return False
        # Add edges
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                u, v = C[i], C[j]
                e = (min(u, v), max(u, v))
                if e in seen_edges:
                    return False
                seen_edges.add(e)
                
    # Check if all edges are covered
    if len(seen_edges) != len(edges):
        return False
        
    # Check if seen_edges is exactly the set of edges
    # (Since we only add edges from the graph, and count matches, it should be exact)
    # But to be safe:
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    if seen_edges != edge_set:
        return False
        
    return True

def construct():
    """
    Entry point: construct clique partitions for two fixed dense random graphs.
    Returns [cliques_for_graph1, cliques_for_graph2].
    """
    # Graph 1
    n1, p1, seed1 = 32, 0.6, 601
    edges1 = _random_graph(n1, p1, seed1)
    
    # Graph 2
    n2, p2, seed2 = 64, 0.55, 602
    edges2 = _random_graph(n2, p2, seed2)
    
    # Allocate time budget
    # Total limit 145s.
    # Graph 1 is smaller, should be faster.
    # Graph 2 is larger, needs more time.
    
    # Let's give 40s to Graph 1 and 100s to Graph 2.
    # Adjust dynamically if Graph 1 finishes early.
    
    time_limit_1 = _T0 + 40.0
    time_limit_2 = _T0 + 145.0
    
    res1 = _solve_instance(edges1, n1, time_limit_1)
    
    # Update time limit for graph 2 based on remaining time
    time_limit_2 = _T0 + 145.0
    
    res2 = _solve_instance(edges2, n2, time_limit_2)
    
    return [res1, res2]
