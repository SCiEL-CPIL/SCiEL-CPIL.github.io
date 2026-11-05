"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

# Deadline guard: must finish well before 150s. Use 140s as hard stop.
DEADLINE = 140.0

def _is_k23_free(adj, n):
    """Check if graph with adjacency matrix adj is K2,3-free.
    Two vertices u,v have a common neighborhood N(u) ∩ N(v).
    If |N(u) ∩ N(v)| >= 3, then K2,3 exists.
    """
    for u in range(n):
        for v in range(u + 1, n):
            # Count common neighbors
            common = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    common += 1
                    if common >= 3:
                        return False
    return True

def _edge_count(adj, n):
    return sum(sum(row) for row in adj) // 2

def _build_adj(edges, n):
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def _random_add_edge(adj, n, rng):
    """Try to add a random edge that doesn't violate K2,3-free."""
    # Generate all non-edges
    non_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v]:
                non_edges.append((u, v))
    if not non_edges:
        return False
    # Shuffle
    rng.shuffle(non_edges)
    for u, v in non_edges:
        # Check if adding (u,v) creates K2,3
        # We need to check that for every pair of vertices (a,b), the number of common neighbors doesn't exceed 2.
        # Adding edge (u,v) increases common neighbor count for:
        # - pair (u, x) where x is neighbor of v: common neighbors of u and x gain v
        # - pair (v, x) where x is neighbor of u: common neighbors of v and x gain u
        # - pair (u, v): they become adjacent; their common neighbors are N(u) ∩ N(v)
        # Actually, we just need to verify that no pair has >= 3 common neighbors after adding.
        # Efficient check: for each vertex w, if w is neighbor of u and neighbor of v, then... 
        # Let's just do a full check but optimized: only check pairs involving u or v.
        
        # Common neighbors of u and v after adding: N(u) ∩ N(v)
        # But wait, u and v are now adjacent. The common neighbors are vertices adjacent to both.
        # Adding (u,v) doesn't change N(u) ∩ N(v) for the purpose of counting common neighbors of OTHER pairs?
        # Let me think again.
        
        # When we add edge (u,v):
        # For any pair (a,b) not involving u or v: their common neighbors don't change.
        # For pair (u, v): their common neighbors are N(u) ∩ N(v) (unchanged by adding the edge itself, since u is not in N(u)).
        # For pair (u, x) where x != v: common neighbors are N(u) ∩ N(x). Adding (u,v) means v is now in N(u). 
        #   So if x is adjacent to v, then v becomes a common neighbor of u and x.
        #   So |N(u) ∩ N(x)| increases by 1 if adj[v][x] is True.
        # For pair (v, x) where x != u: similarly, u becomes a common neighbor if adj[u][x] is True.
        
        # So we need to check:
        # 1. For pair (u,v): |N(u) ∩ N(v)| < 3 (must be <= 2)
        # 2. For each neighbor x of v (x != u): |N(u) ∩ N(x)| + (1 if adj[u][x] else 0) < 3
        #    Wait, if x is neighbor of v, then after adding (u,v), v is in N(u) and v is in N(x) (since x~v). 
        #    So v is a common neighbor of u and x. Previously, was v in N(u) ∩ N(x)? Only if u~v was already true, which it isn't.
        #    So the count increases by exactly 1 for each x in N(v) \ {u}.
        # 3. Similarly for each neighbor x of u (x != v): count increases by 1.
        
        # Let's implement this check.
        
        # First, common neighbors of u and v
        cn_uv = 0
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                cn_uv += 1
                if cn_uv >= 3:
                    break
        if cn_uv >= 3:
            continue
        
        # Check pairs (u, x) for x in N(v), x != u
        ok = True
        for x in range(n):
            if x == u or x == v:
                continue
            if adj[v][x]:
                # Count current common neighbors of u and x
                cnt = 0
                for w in range(n):
                    if w != u and w != x and adj[u][w] and adj[x][w]:
                        cnt += 1
                        if cnt >= 2:  # will be 3 after adding
                            break
                if cnt >= 2:
                    ok = False
                    break
        
        if not ok:
            continue
        
        # Check pairs (v, x) for x in N(u), x != v
        for x in range(n):
            if x == u or x == v:
                continue
            if adj[u][x]:
                cnt = 0
                for w in range(n):
                    if w != v and w != x and adj[v][w] and adj[x][w]:
                        cnt += 1
                        if cnt >= 2:
                            break
                if cnt >= 2:
                    ok = False
                    break
            if not ok:
                break
        
        if ok:
            adj[u][v] = True
            adj[v][u] = True
            return True
    return False

def _greedy_fill(adj, n, rng):
    """Greedily add edges until no more can be added."""
    improved = True
    while improved and time.time() < DEADLINE - 10:
        improved = False
        # Try adding random non-edges
        for _ in range(1000):
            if _random_add_edge(adj, n, rng):
                improved = True
    return adj

def construct():
    n = 20
    rng = random.Random(42)
    start_time = time.time()
    
    best_edges = []
    best_count = 0
    
    # Strategy: Start with a random graph and greedily fill, then do local search.
    # Also try structured starts.
    
    # Approach 1: Start empty, greedily fill
    adj = [[False] * n for _ in range(n)]
    _greedy_fill(adj, n, rng)
    count = _edge_count(adj, n)
    if count > best_count:
        best_count = count
        best_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    best_edges.append([u, v])
    
    # Approach 2: Start with random dense-ish graph, then repair and fill
    for trial in range(20):
        if time.time() > DEADLINE - 15:
            break
        # Create a random graph with p=0.5
        adj = [[False] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < 0.5:
                    adj[u][v] = True
                    adj[v][u] = True
        
        # Remove edges that cause K2,3
        # Iteratively remove problematic edges
        changed = True
        while changed and time.time() < DEADLINE - 10:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not adj[u][v]:
                        continue
                    # Check if removing this edge is needed: is there a K2,3 that uses this edge?
                    # Actually, let's check if the graph is K2,3-free. If not, find a violation and remove an edge.
                    # This is expensive. Let's do a different approach:
                    pass
        
        # Alternative: just greedily add from this starting point
        # First, make sure it's valid by removing edges
        # Let's just check validity and if invalid, remove random edges until valid
        while not _is_k23_free(adj, n) and time.time() < DEADLINE - 10:
            # Find a violating pair
            removed = False
            for u in range(n):
                for v in range(u + 1, n):
                    common = [w for w in range(n) if w != u and w != v and adj[u][w] and adj[v][w]]
                    if len(common) >= 3:
                        # Remove a random edge incident to one of the common neighbors
                        w = rng.choice(common)
                        # Remove edge (u,w) or (v,w)
                        if adj[u][w]:
                            adj[u][w] = False
                            adj[w][u] = False
                            removed = True
                            break
                        elif adj[v][w]:
                            adj[v][w] = False
                            adj[w][v] = False
                            removed = True
                            break
                if removed:
                    break
            if not removed:
                break
        
        # Now greedily fill
        _greedy_fill(adj, n, rng)
        count = _edge_count(adj, n)
        if count > best_count:
            best_count = count
            best_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u][v]:
                        best_edges.append([u, v])
    
    # Approach 3: Local search on best graph found
    if time.time() < DEADLINE - 10:
        adj = _build_adj(best_edges, n)
        # Try to improve by swapping edges
        for _ in range(5000):
            if time.time() > DEADLINE - 5:
                break
            # Pick two random edges to remove
            edges_list = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u][v]:
                        edges_list.append((u, v))
            if len(edges_list) < 2:
                break
            e1, e2 = rng.sample(edges_list, 2)
            # Remove them
            adj[e1[0]][e1[1]] = False
            adj[e1[1]][e1[0]] = False
            adj[e2[0]][e2[1]] = False
            adj[e2[1]][e2[0]] = False
            
            # Try to add two new edges
            added = 0
            for _ in range(200):
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v or adj[u][v]:
                    continue
                # Check if we can add (u,v)
                # Use the same check as before
                cn_uv = 0
                for w in range(n):
                    if w != u and w != v and adj[u][w] and adj[v][w]:
                        cn_uv += 1
                        if cn_uv >= 3:
                            break
                if cn_uv >= 3:
                    continue
                ok = True
                for x in range(n):
                    if x == u or x == v:
                        continue
                    if adj[v][x]:
                        cnt = 0
                        for w in range(n):
                            if w != u and w != x and adj[u][w] and adj[x][w]:
                                cnt += 1
                                if cnt >= 2:
                                    break
                        if cnt >= 2:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                for x in range(n):
                    if x == u or x == v:
                        continue
                    if adj[u][x]:
                        cnt = 0
                        for w in range(n):
                            if w != v and w != x and adj[v][w] and adj[x][w]:
                                cnt += 1
                                if cnt >= 2:
                                    break
                        if cnt >= 2:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    adj[u][v] = True
                    adj[v][u] = True
                    added += 1
                    if added == 2:
                        break
            
            if added < 2:
                # Revert
                adj[e1[0]][e1[1]] = True
                adj[e1[1]][e1[0]] = True
                adj[e2[0]][e2[1]] = True
                adj[e2[1]][e2[0]] = True
            else:
                count = _edge_count(adj, n)
                if count > best_count:
                    best_count = count
                    best_edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj[u][v]:
                                best_edges.append([u, v])
    
    return best_edges
# EVOLVE-BLOCK-END
