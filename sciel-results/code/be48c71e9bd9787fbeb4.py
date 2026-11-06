"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _has_k33(adj):
    """Check if graph with adjacency matrix adj (n x n boolean) contains K3,3."""
    n = adj.shape[0]
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                common = adj[a] & adj[b] & adj[c]
                common[a] = False
                common[b] = False
                common[c] = False
                if np.sum(common) >= 3:
                    return True
    return False

def _is_add_valid(adj, u, v):
    """Fast check: after adding edge (u,v), does any K3,3 appear?
    Only pairs in N(u) ∪ N(v) can gain a new common neighbor.
    """
    n = adj.shape[0]
    # Candidates: all pairs (x, y) where x in N(u)∪N(v) and y in N(u)∪N(v), x != y
    # After adding (u,v), u and v become common neighbors of any pair (x,y) with x,y in N(u)∩N(v)
    # More precisely: a new K3,3 must use the new edge. The new edge contributes u and v as
    # potential common neighbors. So we need: exists pair (x,y) such that |N(x)∩N(y)| >= 3
    # and one of u or v is in N(x)∩N(y) but wasn't before (i.e., x,y are both neighbors of u or both of v).
    # Actually simpler: after adding (u,v), for any pair (x,y) not equal to {u,v},
    # the common neighbors can only increase if x is neighbor of u or v, and y is neighbor of u or v.
    # The new common neighbor is u (if x,y both adjacent to u) or v (if x,y both adjacent to v).
    # So check: for each pair (x,y) where x,y ∈ N(u)∪N(v), x≠y, x≠u, y≠v,
    # count common neighbors of x and y. If >= 3, violation.
    
    # Build candidate set
    nu = np.where(adj[u])[0]
    nv = np.where(adj[v])[0]
    candidates = set(nu.tolist()) | set(nv.tolist())
    candidates.discard(u)
    candidates.discard(v)
    cand_list = list(candidates)
    
    for i in range(len(cand_list)):
        x = cand_list[i]
        for j in range(i+1, len(cand_list)):
            y = cand_list[j]
            # Count common neighbors of x and y
            cn = adj[x] & adj[y]
            # Exclude x and y themselves (not self-loops in simple graph)
            s = np.sum(cn)
            if s >= 3:
                return False
    return True

def _build_edges_from_adj(adj):
    """Convert adjacency matrix to edge list."""
    n = adj.shape[0]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edges.append([i, j])
    return edges

def _try_add_edge(adj, u, v):
    """Try to add edge (u,v). Returns True if graph remains K3,3-free."""
    if adj[u, v]:
        return True
    if _is_add_valid(adj, u, v):
        adj[u, v] = True
        adj[v, u] = True
        return True
    return False

def _greedy_fill(adj, deadline):
    """Greedily add edges in random order until no more can be added or deadline."""
    n = adj.shape[0]
    remaining = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]
    random.shuffle(remaining)
    for u, v in remaining:
        if time.time() > deadline:
            break
        _try_add_edge(adj, u, v)
    return adj

def _local_search(adj, deadline, max_iter=5000):
    """Local search: try to add edges, if stuck try removing one and re-adding."""
    n = adj.shape[0]
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        # Try to add any missing edge
        remaining = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]
        if not remaining:
            break
        random.shuffle(remaining)
        added = False
        for u, v in remaining:
            if time.time() > deadline:
                break
            if _try_add_edge(adj, u, v):
                added = True
                break
        if not added:
            # Stuck: try removing a random existing edge to unlock
            existing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j]]
            if not existing:
                break
            random.shuffle(existing)
            for u, v in existing[:5]:
                if time.time() > deadline:
                    break
                adj[u, v] = False
                adj[v, u] = False
                added2 = False
                remaining2 = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]
                random.shuffle(remaining2)
                for u2, v2 in remaining2:
                    if time.time() > deadline:
                        break
                    if _try_add_edge(adj, u2, v2):
                        added2 = True
                        break
                if not added2:
                    adj[u, v] = True
                    adj[v, u] = True
    return adj

def _projective_plane_seed(n):
    """Create a dense K3,3-free seed using projective plane incidence structure.
    PG(2,3) has 13 points and 13 lines. We map to 20 vertices by using
    a bipartite-like structure or direct construction.
    Alternative: use known extremal constructions for small n.
    For n=20, we use a greedy construction starting from a high-density base.
    """
    adj = np.zeros((n, n), dtype=bool)
    # Use a random dense start and repair
    p = 0.55
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                adj[i, j] = True
                adj[j, i] = True
    # Repair: remove edges to eliminate K3,3
    while _has_k33(adj):
        # Find a violating triple
        found = False
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    common = adj[a] & adj[b] & adj[c]
                    common[a] = False
                    common[b] = False
                    common[c] = False
                    if np.sum(common) >= 3:
                        common_vertices = [v for v in range(n) if common[v]]
                        side1 = random.choice([a, b, c])
                        side2 = random.choice(common_vertices)
                        adj[side1, side2] = False
                        adj[side2, side1] = False
                        found = True
                        break
                if found:
                    break
            if found:
                break
    return adj

def construct():
    n = 20
    deadline = time.time() + 110  # safety margin
    
    best_edges = None
    best_count = 0
    
    # Start with projective-plane-inspired seed
    adj_seed = _projective_plane_seed(n)
    seed_edges = _build_edges_from_adj(adj_seed)
    best_count = len(seed_edges)
    best_edges = seed_edges
    
    # Also try K2,18 as a fallback baseline
    adj_k218 = np.zeros((n, n), dtype=bool)
    for a in range(2):
        for b in range(2, n):
            adj_k218[a, b] = True
            adj_k218[b, a] = True
    k218_edges = _build_edges_from_adj(adj_k218)
    if len(k218_edges) > best_count:
        best_count = len(k218_edges)
        best_edges = k218_edges
    
    # Multiple restarts with random initial graphs and greedy fill
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Random initial graph with moderate-to-high density
        p = random.uniform(0.4, 0.65)
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i, j] = True
                    adj[j, i] = True
        
        # Repair: remove edges to eliminate K3,3
        while _has_k33(adj):
            found = False
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        common = adj[a] & adj[b] & adj[c]
                        common[a] = False
                        common[b] = False
                        common[c] = False
                        if np.sum(common) >= 3:
                            common_vertices = [v for v in range(n) if common[v]]
                            side1 = random.choice([a, b, c])
                            side2 = random.choice(common_vertices)
                            adj[side1, side2] = False
                            adj[side2, side1] = False
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        # Greedy fill with fast validity check
        adj = _greedy_fill(adj, deadline)
        
        # Local search
        adj = _local_search(adj, deadline, max_iter=300)
        
        edges = _build_edges_from_adj(adj)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    # Final greedy pass on best
    if best_edges is not None and time.time() < deadline:
        adj_best = np.zeros((n, n), dtype=bool)
        for u, v in best_edges:
            adj_best[u, v] = True
            adj_best[v, u] = True
        adj_best = _greedy_fill(adj_best, deadline)
        adj_best = _local_search(adj_best, deadline, max_iter=200)
        final_edges = _build_edges_from_adj(adj_best)
        if len(final_edges) > best_count:
            best_edges = final_edges
    
    return best_edges
# EVOLVE-BLOCK-END
