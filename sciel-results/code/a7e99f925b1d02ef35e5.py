"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START

def _creates_c6(u, v, adj):
    """
    Check if adding edge (u, v) to the graph represented by `adj` creates a C6.
    This is equivalent to checking if there is a simple path of length 5 between u and v in the current graph.
    Path: u - a - b - c - d - v
    """
    nu_mask = adj[u]
    nv_mask = adj[v]
    
    # We iterate through a in N(u)
    am = nu_mask
    while am:
        lb_a = am & (-am)
        ai = lb_a.bit_length() - 1
        am ^= lb_a
        
        # b in N(a) \ {u, v, a}
        # Note: a is not in N(a) for simple graph. u and v must be excluded.
        bm = adj[ai] & ~(1 << u) & ~(1 << v) & ~lb_a
        while bm:
            lb_b = bm & (-bm)
            bi = lb_b.bit_length() - 1
            bm ^= lb_b
            
            # c in N(b) \ {u, v, a, b}
            cm = adj[bi] & ~(1 << u) & ~(1 << v) & ~lb_a & ~lb_b
            while cm:
                lb_c = cm & (-cm)
                ci = lb_c.bit_length() - 1
                cm ^= lb_c
                
                # d in N(c) \ {u, v, a, b, c}
                # And d must be in N(v)
                # Candidates for d are neighbors of c that are also neighbors of v.
                cand_d = adj[ci] & nv_mask
                # We must ensure d is distinct from u, a, b, c.
                # d != c is guaranteed (simple graph).
                # d != v is guaranteed (d in N(v)).
                # We must exclude u, a, b.
                cand_d &= ~(1 << u)
                cand_d &= ~lb_a
                cand_d &= ~lb_b
                
                if cand_d:
                    return True
                    
    return False

def _count_edges(adj, n):
    count = 0
    for i in range(n):
        count += adj[i].bit_count()
    return count // 2

def _edges_from_adj(adj, n):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    return edges

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_adj = [0] * n
    best_count = 0
    
    # Try multiple random orderings
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        random.seed(42 + restart)
        adj = [0] * n
        
        # Shuffle the order of edges to add
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if not ((adj[u] >> v) & 1):
                if not _creates_c6(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        count = _count_edges(adj, n)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            
        # If we have plenty of time, try a local search improvement
        # Simple 1-for-1 swap: remove an edge, try to add another
        if time.time() < deadline - 20:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                current_edges = _count_edges(adj, n)
                # Try removing each edge
                for u, v in all_pairs:
                    if time.time() > deadline - 5:
                        break
                    if not ((adj[u] >> v) & 1):
                        continue
                    
                    # Remove edge (u,v)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    # Try adding any non-edge
                    found_improvement = False
                    for x, y in all_pairs:
                        if time.time() > deadline - 5:
                            break
                        if (x == u and y == v) or (x == v and y == u):
                            continue
                        if (adj[x] >> y) & 1:
                            continue
                        
                        if not _creates_c6(x, y, adj):
                            adj[x] |= (1 << y)
                            adj[y] |= (1 << x)
                            
                            # Greedy add remaining edges
                            for w, z in all_pairs:
                                if time.time() > deadline - 5:
                                    break
                                if not ((adj[w] >> z) & 1):
                                    if not _creates_c6(w, z, adj):
                                        adj[w] |= (1 << z)
                                        adj[z] |= (1 << w)
                            
                            new_count = _count_edges(adj, n)
                            if new_count > current_edges:
                                improved = True
                                found_improvement = True
                                break
                            else:
                                # Revert
                                adj[x] &= ~(1 << y)
                                adj[y] &= ~(1 << x)
                                # Restore original edge
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                    
                    if found_improvement:
                        break
                    else:
                        # Restore original edge if we didn't break out
                        if not ((adj[u] >> v) & 1):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                
                if improved and count < _count_edges(adj, n):
                    best_count = _count_edges(adj, n)
                    best_adj = adj[:]

    return _edges_from_adj(best_adj, n)

# EVOLVE-BLOCK-END
