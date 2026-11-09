"""C4-free graph with high minimum degree (R(C4,K1,39) lower bound)."""
import time
import random

# EVOLVE-BLOCK-START

def _can_add_edge(adj, u, v):
    """
    Check if adding edge (u, v) creates a C4.
    A C4 is formed if there exist x, y such that:
    u-x, x-v, v-y, y-u are edges (in the final graph with u-v added).
    Since u-v is not yet in adj, we need:
    - x is a neighbor of v (currently)
    - y is a neighbor of u (currently)
    - x != y
    - edge (x, y) exists (currently)
    
    Equivalently: there exists x in N(v) such that N(x) intersects N(u) (excluding x itself, but x not in N(x)).
    Actually: for any x in N(v), if there exists y in N(u) with edge(x,y), then C4 u-y-x-v-u.
    y must be != x. Since x in N(v) and we're checking edge(x,y), y is in N(x).
    We need y in N(u) as well. So N(x) & N(u) must be non-empty.
    Note: x is not in N(u) necessarily. If x in N(u), then u-x is edge. But then y=x would be in N(u) and N(x)? No, x not in N(x).
    So if N(x) & N(u) != 0, there is some y in N(u) connected to x.
    The C4 is u-y-x-v-u.
    Edges: u-y (y in N(u)), y-x (edge), x-v (x in N(v)), v-u (to be added).
    This is valid if u, y, x, v are distinct.
    u != y (simple graph), u != x (x in N(v), if x=u then u in N(v) means v-u edge exists, but we are adding it, so not yet. Wait, if x=u, then u in N(v). But we are adding u-v. So currently u not in N(v). So x != u.)
    y != v (y in N(u), if y=v then v in N(u), meaning u-v edge exists, but we are adding it, so not yet. So y != v.)
    x != v (x in N(v), simple graph no self loop).
    y != x (as discussed, x not in N(x), so y != x if y in N(x)).
    So u, y, x, v are distinct.
    """
    nu = adj[u]
    nv = adj[v]
    
    # Iterate over neighbors of v
    x = nv
    while x:
        lsb = x & -x
        xi = lsb.bit_length() - 1
        x ^= lsb
        
        # Check if N(xi) intersects N(u)
        if adj[xi] & nu:
            return False
            
    return True

def _greedy_build(n, rng, order_seed=None):
    """
    Build a C4-free graph by greedily adding edges.
    """
    adj = [0] * n
    edges_to_try = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        if _can_add_edge(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    degrees = [a.bit_count() for a in adj]
    min_deg = min(degrees) if degrees else 0
    
    return adj, min_deg

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if (ai >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    deadline = time.time() + 140.0  # safety margin under 150s
    best_n = 39
    best_edges = []
    best_adj = None
    
    # Try to maximize n from 47 down
    for n in range(47, 38, -1):
        target_min_deg = n - 39
        found = False
        
        # Try multiple seeds
        for seed in range(500):
            if time.time() > deadline:
                break
            
            rng = random.Random(seed)
            adj, min_deg = _greedy_build(n, rng)
            
            if min_deg >= target_min_deg:
                edges = _edges_from_adj(adj, n)
                
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    best_adj = adj
                found = True
                
                # If we found n=47, return immediately
                if n == 47:
                    return {'n': n, 'edges': edges}
        
        if found:
            # Since we iterate n downwards, this is the largest n found
            return {'n': best_n, 'edges': best_edges}
        
        # If not found for this n, continue to smaller n
    
    # Fallback
    if best_adj is not None:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}

# EVOLVE-BLOCK-END
