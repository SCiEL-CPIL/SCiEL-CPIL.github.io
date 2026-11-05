"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _score(graph_edges, n=16):
    """Compute the largest triangle degree sum."""
    # Build adjacency sets and degrees
    adj = [set() for _ in range(n)]
    deg = [0] * n
    for u, v in graph_edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    # Find all triangles
    max_sum = 0
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[u] & adj[v]:
                if w > v:
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
    return max_sum

def _random_walk_optimize(n=16, target_edges=65, max_iters=50000, time_limit=100):
    """
    Start from a balanced random graph with target_edges edges,
    then use local search (edge swaps) to minimize the max triangle degree sum.
    """
    start_time = time.time()
    
    # Generate a random graph with exactly target_edges edges
    all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(all_possible)
    edges = list(set(all_possible[:target_edges]))
    # Ensure exactly target_edges
    while len(edges) < target_edges:
        # Add a random missing edge
        missing = [e for e in all_possible if e not in set(map(tuple, edges))]
        if not missing:
            break
        edges.append(list(random.choice(missing)))
    edges = [list(e) for e in edges[:target_edges]]
    
    current_score = _score(edges, n)
    best_edges = [e[:] for e in edges]
    best_score = current_score
    
    edge_set = set(map(tuple, edges))
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random edge to remove
        if not edges:
            break
        idx = random.randint(0, len(edges)-1)
        u, v = edges[idx]
        
        # Find a non-edge to add
        # Try a few random non-edges
        for _ in range(20):
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            if a == b:
                continue
            if (a, b) in edge_set or (b, a) in edge_set:
                continue
            if (u, v) == (a, b) or (u, v) == (b, a):
                continue
            
            # Try the swap
            new_edge_set = set(edge_set)
            new_edge_set.remove((u, v))
            new_edge_set.add((a, b))
            new_edges = [list(e) for e in new_edge_set]
            
            new_score = _score(new_edges, n)
            
            if new_score < current_score:
                # Accept
                edge_set = new_edge_set
                edges = new_edges
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = [e[:] for e in edges]
                break
            else:
                # Reject, try another non-edge
                continue
    
    return best_edges, best_score

def _try_balanced_construction():
    """
    Try a construction based on Turan-like structure with careful edge placement.
    K_{8,8} has 64 edges and no triangles. We need 65 edges.
    Adding one edge within a part creates triangles involving that edge.
    
    Alternative: start with a different base.
    Consider a graph that is close to balanced but with edges distributed to minimize triangle degree sums.
    
    Let's try: two parts of size 8. Instead of adding an edge within a part,
    we can remove one cross edge and add two within-part edges? No, that changes count.
    
    Actually, let's just use local search from multiple random starts and also from the seed.
    """
    pass

def construct():
    """
    Construct a graph on 16 vertices with 65 edges minimizing the max triangle degree sum.
    """
    random.seed(42)
    
    n = 16
    target_edges = 65
    
    best_score = float('inf')
    best_edges = None
    
    # Start 1: The seed construction (K_{8,8} + one internal edge)
    edges_seed = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    score_seed = _score(edges_seed, n)
    if score_seed < best_score:
        best_score = score_seed
        best_edges = [e[:] for e in edges_seed]
    
    # Start 2-5: Random walks from different random initial graphs
    for attempt in range(5):
        try:
            edges_opt, score_opt = _random_walk_optimize(n, target_edges, max_iters=20000, time_limit=15)
            if score_opt < best_score:
                best_score = score_opt
                best_edges = [e[:] for e in edges_opt]
        except Exception:
            continue
    
    # Also try a more structured approach: 
    # Partition into two sets A and B of size 8.
    # Include all cross edges (64) plus one internal edge.
    # The seed does this. Let's try different internal edges or different partitions.
    
    # Try all possible single internal edges in a K_{8,8} base
    # But K_{8,8} is fixed by the partition. Different partitions are isomorphic for scoring purposes?
    # The degree sum depends on which vertices get the extra degree.
    # In K_{8,8} + edge (u,v) within A:
    # deg(u) = 9, deg(v) = 9, all other A vertices = 8, all B vertices = 8.
    # Triangles: only those containing edge (u,v) and a vertex w in B.
    # For each w in B, (u,v,w) is a triangle. deg(u)+deg(v)+deg(w) = 9+9+8 = 26.
    # So score is 26.
    
    # Can we do better?
    # If we don't use K_{8,8} as base, maybe a more balanced degree sequence helps.
    # 65 edges on 16 vertices: avg degree = 130/16 = 8.125.
    # Ideal degree sequence: ten vertices of degree 8, six vertices of degree 9? 
    # Sum = 10*8 + 6*9 = 80 + 54 = 134. No, sum must be 130.
    # 14 vertices of degree 8, 2 vertices of degree 9: 14*8+2*9 = 112+18=130. Yes.
    # Or 15 of degree 8, 1 of degree 10: 120+10=130.
    # Or 13 of degree 8, 3 of degree 9, ... 13*8+3*9=104+27=131. No.
    # So possible: (9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,8) or (10,8,8,...) etc.
    
    # Let's try to construct a graph with degree sequence (9,9,8,...,8) and few triangles.
    # This is hard to do directly. Local search should find good configurations.
    
    if best_edges is None:
        # Fallback to seed
        best_edges = edges_seed
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
