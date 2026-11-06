"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 20
DEADLINE_FRACTION = 0.85  # safety margin

def _has_k33(adj):
    """Check if graph with adjacency list has a K3,3 subgraph using bitmasks."""
    # Convert to bitmask representation
    masks = []
    for s in adj:
        m = 0
        for v in s:
            m |= (1 << v)
        masks.append(m)
    
    for a in range(N):
        ma = masks[a]
        for b in range(a + 1, N):
            common_ab = ma & masks[b]
            if common_ab.bit_count() < 3:
                continue
            for c in range(b + 1, N):
                common_abc = common_ab & masks[c]
                if common_abc.bit_count() >= 3:
                    return True
    return False

def _try_add_edge_fast(adj, masks, u, v):
    """Try to add edge (u,v). Uses incremental check for speed."""
    if u == v or v in adj[u]:
        return False
    
    # Adding (u,v) can only create violations for triples where u or v
    # becomes a NEW common neighbor of the other two vertices.
    
    # Case 1: v becomes a common neighbor of some pair (x,y) where x,y != u,v
    # For triple (u, x, y): after adding (u,v), v is in N(u). 
    # Common neighbors of u,x,y = N(u)∩N(x)∩N(y). v is in this iff v∈N(x)∩N(y).
    # Before: |N(u)∩N(x)∩N(y)| = k. After: k + (1 if v∈N(x)∩N(y) else 0).
    # Violation iff k >= 2 and v∈N(x)∩N(y).
    
    nu = masks[u]
    nv = masks[v]
    
    # Check triples (u, x, y) for x < y, x,y != v
    for x in range(N):
        if x == u or x == v:
            continue
        nx = masks[x]
        # v must be in N(x) for v to be a common neighbor of u,x,y (since v will be in N(u))
        if not (nv & (1 << x)):
            continue
        common_ux = nu & nx
        for y in range(x + 1, N):
            if y == u or y == v:
                continue
            ny = masks[y]
            if not (nv & (1 << y)):
                continue
            common_uxy = common_ux & ny
            if common_uxy.bit_count() >= 2:
                return False
    
    # Case 2: u becomes a common neighbor of some pair (x,y) where x,y != u,v
    for x in range(N):
        if x == u or x == v:
            continue
        nx = masks[x]
        if not (nu & (1 << x)):
            continue
        common_vx = nv & nx
        for y in range(x + 1, N):
            if y == u or y == v:
                continue
            ny = masks[y]
            if not (nu & (1 << y)):
                continue
            common_vxy = common_vx & ny
            if common_vxy.bit_count() >= 2:
                return False
    
    # Also check triples (u, v, x): common neighbors = N(u)∩N(v)∩N(x)
    # This doesn't change when adding (u,v), so no new violation.
    
    # Add the edge
    adj[u].add(v)
    adj[v].add(u)
    masks[u] |= (1 << v)
    masks[v] |= (1 << u)
    return True

def _remove_edge(adj, masks, u, v):
    adj[u].discard(v)
    adj[v].discard(u)
    masks[u] &= ~(1 << v)
    masks[v] &= ~(1 << u)

def _edge_count(adj):
    return sum(len(s) for s in adj) // 2

def _make_adj_from_edges(edges):
    adj = [set() for _ in range(N)]
    masks = [0] * N
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return adj, masks

def _add_only_search(adj, masks, deadline):
    """Greedy add-only local search."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        pairs = [(u, v) for u in range(N) for v in range(u + 1, N) if v not in adj[u]]
        random.shuffle(pairs)
        for u, v in pairs:
            if _try_add_edge_fast(adj, masks, u, v):
                improved = True
            if time.time() > deadline:
                return adj, masks
    return adj, masks

def _swap_search(adj, masks, deadline):
    """Edge swap local search: remove one edge, add another."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        existing = [(u, v) for u in range(N) for v in range(u + 1, N) if v in adj[u]]
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) if v not in adj[u]]
        random.shuffle(existing)
        random.shuffle(missing)
        
        for u1, v1 in existing:
            if time.time() > deadline:
                return adj, masks
            for u2, v2 in missing:
                if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                    continue
                # Try the swap
                _remove_edge(adj, masks, u1, v1)
                if _try_add_edge_fast(adj, masks, u2, v2):
                    improved = True
                    break
                else:
                    # Revert the add (it wasn't added) and re-add removed edge
                    _remove_edge(adj, masks, u2, v2)  # no-op if not added
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    masks[u1] |= (1 << v1)
                    masks[v1] |= (1 << u1)
            if improved:
                break
    return adj, masks

def _two_swap_search(adj, masks, deadline):
    """Try removing 2 edges and adding 2 edges to escape local optima."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        existing = [(u, v) for u in range(N) for v in range(u + 1, N) if v in adj[u]]
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) if v not in adj[u]]
        random.shuffle(existing)
        random.shuffle(missing)
        
        for i in range(min(len(existing), 50)):
            if time.time() > deadline:
                return adj, masks
            for j in range(i + 1, min(len(existing), 50)):
                if time.time() > deadline:
                    return adj, masks
                u1, v1 = existing[i]
                u2, v2 = existing[j]
                for k in range(min(len(missing), 30)):
                    if time.time() > deadline:
                        return adj, masks
                    for l in range(k + 1, min(len(missing), 30)):
                        if time.time() > deadline:
                            return adj, masks
                        a1, b1 = missing[k]
                        a2, b2 = missing[l]
                        # Check no conflicts
                        if (a1, b1) == (u1, v1) or (a1, b1) == (v1, u1):
                            continue
                        if (a2, b2) == (u1, v1) or (a2, b2) == (v1, u1):
                            continue
                        if (a1, b1) == (u2, v2) or (a1, b1) == (v2, u2):
                            continue
                        if (a2, b2) == (u2, v2) or (a2, b2) == (v2, u2):
                            continue
                        if (a1, b1) == (a2, b2) or (a1, b1) == (b2, a2):
                            continue
                        
                        # Remove both existing edges
                        _remove_edge(adj, masks, u1, v1)
                        _remove_edge(adj, masks, u2, v2)
                        
                        # Try to add both new edges
                        added1 = _try_add_edge_fast(adj, masks, a1, b1)
                        if added1:
                            added2 = _try_add_edge_fast(adj, masks, a2, b2)
                            if added2:
                                improved = True
                                break
                            else:
                                _remove_edge(adj, masks, a1, b1)
                        # Revert if not successful
                        if not improved:
                            adj[u1].add(v1)
                            adj[v1].add(u1)
                            masks[u1] |= (1 << v1)
                            masks[v1] |= (1 << u1)
                            adj[u2].add(v2)
                            adj[v2].add(u2)
                            masks[u2] |= (1 << v2)
                            masks[v2] |= (1 << u2)
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return adj, masks

def _structured_initial_graphs():
    """Generate several structured initial graphs."""
    graphs = []
    
    # 1. K2,18 (36 edges)
    edges1 = [[a, b] for a in range(2) for b in range(2, 20)]
    graphs.append(edges1)
    
    # 2. Blow-up of C5: partition 20 vertices into 5 groups of 4, connect consecutive groups
    edges2 = []
    for i in range(5):
        for j in [(i + 1) % 5]:
            for a in range(i * 4, (i + 1) * 4):
                for b in range(j * 4, (j + 1) * 4):
                    edges2.append([a, b])
    graphs.append(edges2)
    
    # 3. Blow-up of C5 with 4 groups: 4,4,4,4,4 -> same as above but let's try different partition
    # Actually C5 blow-up with 5 groups of 4 gives 5*4*4 = 80 edges (but each edge counted once)
    # Wait: 5 groups, each pair of consecutive groups has 4*4=16 edges, 5 pairs = 80 edges
    # But C5 is bipartite? No. Let me reconsider.
    # C5 has 5 vertices, 5 edges. Blow-up with 4 copies of each: 5*4*4 = 80 edges.
    # Is the blow-up of C5 K3,3-free? C5 is K3,3-free. The blow-up of a K3,3-free graph
    # is K3,3-free if the blow-up is "proper" (no K3 in the original creates K3,3).
    # Actually, the lexicographic product C5[4] (replace each vertex with K4) would have
    # K3,3. But the "blow-up" where we replace each vertex with an independent set and
    # connect all vertices in consecutive groups: this is the strong product? No.
    # Let me just check: in this graph, can we find K3,3?
    # Pick 3 vertices from group i and 3 from group j. They're connected only if i,j are consecutive.
    # For K3,3, we need 3 vertices in one part all connected to 3 in the other.
    # If we pick 3 from group i, they're only connected to groups (i-1)%5 and (i+1)%5.
    # So the 3 neighbors must come from at most 2 groups. If from 2 groups, say group a and b,
    # then the 3 neighbors in group a are all connected to the 3 in group i, and same for b.
    # But we need 3 neighbors total, all connected to all 3 in group i.
    # If we pick 2 from group a and 1 from group b, that's 3 neighbors, all connected to group i.
    # Then we need 3 vertices in the "other part" all connected to these 3 neighbors.
    # The neighbors are in groups a and b. A vertex in group c is connected to them only if
    # c is adjacent to a or b in C5.
    # This is getting complex. Let me just include it and let the search verify.
    
    # 4. Random moderate density
    random.seed(42)
    edges4 = []
    for u in range(N):
        for v in range(u + 1, N):
            if random.random() < 0.4:
                edges4.append([u, v])
    graphs.append(edges4)
    
    # 5. Another random
    random.seed(123)
    edges5 = []
    for u in range(N):
        for v in range(u + 1, N):
            if random.random() < 0.5:
                edges5.append([u, v])
    graphs.append(edges5)
    
    return graphs

def construct():
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION
    
    best_adj = None
    best_masks = None
    best_count = 0
    
    def _update_best(adj, masks, count):
        nonlocal best_adj, best_masks, best_count
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]
            best_masks = list(masks)
    
    initial_graphs = _structured_initial_graphs()
    
    for edges in initial_graphs:
        if time.time() > deadline:
            break
        adj, masks = _make_adj_from_edges(edges)
        # Verify it's K3,3-free
        if _has_k33(adj):
            # Remove edges until valid (shouldn't happen for our structured graphs)
            # Just skip this one
            continue
        count = _edge_count(adj)
        _update_best(adj, masks, count)
        
        # Add-only search
        adj, masks = _add_only_search(adj, masks, deadline)
        count = _edge_count(adj)
        _update_best(adj, masks, count)
        
        # Swap search
        adj, masks = _swap_search(adj, masks, deadline)
        count = _edge_count(adj)
        _update_best(adj, masks, count)
        
        # Two-swap search
        adj, masks = _two_swap_search(adj, masks, deadline)
        count = _edge_count(adj)
        _update_best(adj, masks, count)
    
    # Random restarts
    restarts = 0
    while time.time() < deadline and restarts < 30:
        restarts += 1
        adj = [set() for _ in range(N)]
        masks = [0] * N
        all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if _try_add_edge_fast(adj, masks, u, v):
                pass
            if time.time() > deadline:
                break
        
        count = _edge_count(adj)
        _update_best(adj, masks, count)
        
        adj, masks = _add_only_search(adj, masks, deadline)
        count = _edge_count(adj)
        _update_best(adj, masks, count)
        
        adj, masks = _swap_search(adj, masks, deadline)
        count = _edge_count(adj)
        _update_best(adj, masks, count)
        
        adj, masks = _two_swap_search(adj, masks, deadline)
        count = _edge_count(adj)
        _update_best(adj, masks, count)
    
    if best_adj is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    edges = []
    for u in range(N):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
