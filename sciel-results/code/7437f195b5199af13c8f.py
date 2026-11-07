"""Rewritten program: seed with K7,8, then local-search to add intra-part edges."""
# EVOLVE-BLOCK-START
import time
import random

N = 15

def _is_valid(adj, deg):
    """Check if graph (adjacency matrix) has no violating 3-path s-u-v-t."""
    for u in range(N):
        for v in range(u + 1, N):
            if not adj[u][v]:
                continue
            # For each s in N(u), t in N(v), s!=t, s!=v, t!=u, check deg(s)!=deg(t)
            for s in range(N):
                if s == u or s == v or not adj[s][u]:
                    continue
                for t in range(N):
                    if t == u or t == v or t == s or not adj[t][v]:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _try_add_edge_fast(adj, deg, u, v):
    """Try adding edge (u,v); return (new_adj, new_deg) if valid, else None.
    Uses bitmask for speed."""
    if u == v or adj[u][v]:
        return None
    
    # Build neighbor bitmasks for u and v (excluding each other)
    nbr_u_mask = 0
    nbr_v_mask = 0
    for s in range(N):
        if s == u and s == v:
            continue
        if s != v and adj[s][u]:
            nbr_u_mask |= (1 << s)
        if s != u and adj[s][v]:
            nbr_v_mask |= (1 << s)
    
    # Check: for any s in nbr_u, t in nbr_v, s!=t, deg[s]==deg[t]
    # This creates path s-u-v-t
    for s in range(N):
        if not (nbr_u_mask & (1 << s)):
            continue
        ds = deg[s]
        for t in range(N):
            if t == s or not (nbr_v_mask & (1 << t)):
                continue
            if ds == deg[t]:
                return None
    
    # Also check paths s-v-u-t: s in nbr_v, t in nbr_u, s!=t
    for s in range(N):
        if not (nbr_v_mask & (1 << s)):
            continue
        ds = deg[s]
        for t in range(N):
            if t == s or not (nbr_u_mask & (1 << t)):
                continue
            if ds == deg[t]:
                return None
    
    # Build new adj and deg
    new_adj = [row[:] for row in adj]
    new_adj[u][v] = new_adj[v][u] = 1
    new_deg = deg[:]
    new_deg[u] += 1
    new_deg[v] += 1
    
    # Full recheck because degree changes may invalidate other edges
    if _is_valid(new_adj, new_deg):
        return (new_adj, new_deg)
    return None

def _try_swap_fast(adj, deg, remove, add):
    """Remove edge `remove`, add edge `add`. Return (new_adj, new_deg) if valid."""
    (ru, rv) = remove
    (au, av) = add
    if au == av or adj[au][av]:
        return None
    if not adj[ru][rv]:
        return None
    
    new_adj = [row[:] for row in adj]
    new_adj[ru][rv] = new_adj[rv][ru] = 0
    new_adj[au][av] = new_adj[av][au] = 1
    new_deg = deg[:]
    new_deg[ru] -= 1
    new_deg[rv] -= 1
    new_deg[au] += 1
    new_deg[av] += 1
    
    if _is_valid(new_adj, new_deg):
        return (new_adj, new_deg)
    return None

def _edge_count(adj):
    return sum(sum(row) for row in adj) // 2

def construct():
    start = time.time()
    deadline = start + 80  # 10s safety margin
    
    # Seed: K7,8 bipartite
    # Part A = {0..6} (7 vertices), Part B = {7..14} (8 vertices)
    adj = [[0]*N for _ in range(N)]
    deg = [0]*N
    for a in range(7):
        for b in range(7, 15):
            adj[a][b] = adj[b][a] = 1
    for a in range(7):
        deg[a] = 8
    for b in range(7, 15):
        deg[b] = 7
    # Verify seed is valid
    assert _is_valid(adj, deg), "Seed K7,8 should be valid"
    
    best_adj = [row[:] for row in adj]
    best_deg = deg[:]
    best_count = _edge_count(best_adj)
    
    # Local search: try adding edges (especially intra-part), then swaps
    # Intra-part edges: within A (0..6) and within B (7..14)
    random.seed(42)
    
    # Phase 1: greedily try to add any missing edge
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        for u in range(N):
            for v in range(u+1, N):
                if adj[u][v]:
                    continue
                res = _try_add_edge_fast(adj, deg, u, v)
                if res is not None:
                    adj, deg = res
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                    best_count = _edge_count(best_adj)
                    improved = True
                    break
            if improved:
                break
    
    # Phase 2: targeted swaps - try all (non_edge, edge) pairs systematically
    all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
    non_edges = [(u,v) for (u,v) in all_edges if not adj[u][v]]
    edges_present = [(u,v) for (u,v) in all_edges if adj[u][v]]
    
    # Try systematic 1-swap: for each non-edge, try removing each edge
    for add in non_edges:
        if time.time() > deadline - 5:
            break
        for remove in edges_present:
            if remove == add:
                continue
            res = _try_swap_fast(adj, deg, remove, add)
            if res is not None:
                adj, deg = res
                cnt = _edge_count(adj)
                if cnt > best_count:
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                    best_count = cnt
                # Update lists after a successful swap
                non_edges = [(u,v) for (u,v) in all_edges if not adj[u][v]]
                edges_present = [(u,v) for (u,v) in all_edges if adj[u][v]]
                break
    
    # Phase 3: random swaps to escape local optima
    for iteration in range(500000):
        if time.time() > deadline - 5:
            break
        # Pick a random non-edge to add
        non_edges = [(u,v) for (u,v) in all_edges if not adj[u][v]]
        if not non_edges:
            break
        add = random.choice(non_edges)
        # Pick a random edge to remove (not the same)
        edges_present = [(u,v) for (u,v) in all_edges if adj[u][v]]
        if not edges_present:
            break
        remove = random.choice(edges_present)
        if remove == add:
            continue
        res = _try_swap_fast(adj, deg, remove, add)
        if res is not None:
            adj, deg = res
            cnt = _edge_count(adj)
            if cnt > best_count:
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
                best_count = cnt
    
    # Phase 4: final greedy add pass
    improved = True
    while improved and (time.time() < deadline - 2):
        improved = False
        for u in range(N):
            for v in range(u+1, N):
                if adj[u][v]:
                    continue
                res = _try_add_edge_fast(adj, deg, u, v)
                if res is not None:
                    adj, deg = res
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                    best_count = _edge_count(best_adj)
                    improved = True
                    break
            if improved:
                break
    
    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if best_adj[u][v]:
                edges.append([int(u), int(v)])
    return edges
# EVOLVE-BLOCK-END
