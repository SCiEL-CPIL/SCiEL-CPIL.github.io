"""Rewritten program: multi-start local search with diverse seeds."""
# EVOLVE-BLOCK-START
import time
import random

N = 15

def _conflict_count_for_edge(adj, deg, u, v):
    """Count violations on edge (u,v). A violation is a pair (s,t) with
    s in N(u)\\{v}, t in N(v)\\{u}, s!=t, deg[s]==deg[t]."""
    cnt = 0
    for s in range(N):
        if s == u or s == v:
            continue
        if not adj[s][u]:
            continue
        ds = deg[s]
        for t in range(N):
            if t == u or t == v or t == s:
                continue
            if adj[t][v] and deg[t] == ds:
                cnt += 1
    return cnt

def _total_conflicts(adj, deg):
    """Sum of conflict counts over all edges."""
    total = 0
    for u in range(N):
        for v in range(u + 1, N):
            if adj[u][v]:
                total += _conflict_count_for_edge(adj, deg, u, v)
    return total

def _is_valid_fast(adj, deg):
    """Check validity by verifying total conflict count is zero."""
    return _total_conflicts(adj, deg) == 0

def _try_add_edge(adj, deg, u, v):
    """Try adding edge (u,v); return (new_adj, new_deg) if valid, else None."""
    if u == v or adj[u][v]:
        return None
    # Quick check: after adding (u,v), deg[u] and deg[v] increase by 1.
    # Only edges incident to u or v (and the new edge itself) can gain conflicts.
    new_deg = deg[:]
    new_deg[u] += 1
    new_deg[v] += 1
    # Check the new edge (u,v) for conflicts
    if _conflict_count_for_edge(adj, new_deg, u, v) > 0:
        return None
    # Check all existing edges incident to u or v
    for w in range(N):
        if w == u or w == v:
            continue
        if adj[u][w]:
            if _conflict_count_for_edge(adj, new_deg, u, w) > 0:
                return None
        if adj[v][w]:
            if _conflict_count_for_edge(adj, new_deg, v, w) > 0:
                return None
    new_adj = [row[:] for row in adj]
    new_adj[u][v] = new_adj[v][u] = 1
    return (new_adj, new_deg)

def _try_swap(adj, deg, remove, add):
    """Remove edge `remove`, add edge `add`. Return (new_adj, new_deg) if valid."""
    (ru, rv) = remove
    (au, av) = add
    if au == av or adj[au][av]:
        return None
    if not adj[ru][rv]:
        return None
    new_deg = deg[:]
    new_deg[ru] -= 1
    new_deg[rv] -= 1
    new_deg[au] += 1
    new_deg[av] += 1
    # Build the tentative adjacency (only needed for checking)
    # Edges affected: all edges incident to ru, rv, au, av, plus the new edge
    affected = set()
    for w in range(N):
        if w == ru or w == rv:
            if adj[ru][w] or adj[rv][w]:
                affected.add((min(ru,rv), max(ru,rv)) if (min(ru,rv)==w or max(ru,rv)==w) else (min(ru,w),max(ru,w)) if adj[ru][w] else (min(rv,w),max(rv,w)))
    # Simpler: just check all edges that could be affected
    # An edge (x,y) is affected if x or y is in {ru,rv,au,av}
    # (because degree changes for those vertices)
    # Plus the new edge (au,av) itself
    # And the removed edge is gone so no need to check it
    check_edges = set()
    for w in range(N):
        for x in (ru, rv, au, av):
            if w != x and adj[x][w]:
                check_edges.add((min(x, w), max(x, w)))
    check_edges.add((min(au, av), max(au, av)))
    # Remove the edge being removed from checks
    check_edges.discard((min(ru, rv), max(ru, rv)))
    for (x, y) in check_edges:
        if x == y:
            continue
        if _conflict_count_for_edge(adj, new_deg, x, y) > 0:
            # But if the edge doesn't exist in new graph, skip
            # Edge exists in new graph if: it existed before and wasn't removed,
            # OR it's the new edge (au,av)
            if (x, y) == (min(ru,rv), max(ru,rv)):
                continue  # removed edge, not present
            if adj[x][y] or (x == au and y == av) or (x == av and y == au):
                return None
    new_adj = [row[:] for row in adj]
    new_adj[ru][rv] = new_adj[rv][ru] = 0
    new_adj[au][av] = new_adj[av][au] = 1
    return (new_adj, new_deg)

def _edge_count(adj):
    return sum(sum(row) for row in adj) // 2

def _greedy_add(adj, deg, deadline):
    """Repeatedly try to add edges until no more can be added or time runs out."""
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for u in range(N):
            for v in range(u+1, N):
                if adj[u][v]:
                    continue
                res = _try_add_edge(adj, deg, u, v)
                if res is not None:
                    adj, deg = res
                    improved = True
                    break
            if improved:
                break
    return adj, deg

def _local_search(adj, deg, deadline, rng, max_iter=200000):
    """Random 2-swap local search with violation-driven removal. Returns (adj, deg, count)."""
    all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
    count = _edge_count(adj)
    stall_count = 0
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        # Rebuild lists (cache-friendly: use list comprehension once)
        non_edges = [(u,v) for (u,v) in all_edges if not adj[u][v]]
        if not non_edges:
            break
        edges_present = [(u,v) for (u,v) in all_edges if adj[u][v]]
        if not edges_present:
            break
        add = rng.choice(non_edges)
        remove = rng.choice(edges_present)
        res = _try_swap(adj, deg, remove, add)
        if res is not None:
            adj, deg = res
            new_count = _edge_count(adj)
            if new_count > count:
                count = new_count
                stall_count = 0
            else:
                stall_count += 1
        else:
            stall_count += 1
        # If stalled for a while, try violation-driven removal
        if stall_count > 500:
            # Find the edge with the most conflicts
            worst_edge = None
            worst_conf = 0
            for (eu, ev) in edges_present:
                c = _conflict_count_for_edge(adj, deg, eu, ev)
                if c > worst_conf:
                    worst_conf = c
                    worst_edge = (eu, ev)
            if worst_edge is not None and worst_conf > 0:
                # Remove the worst edge
                adj[worst_edge[0]][worst_edge[1]] = 0
                adj[worst_edge[1]][worst_edge[0]] = 0
                deg[worst_edge[0]] -= 1
                deg[worst_edge[1]] -= 1
                count -= 1
                stall_count = 0
                # Try to add back a different edge
                for _ in range(10):
                    a = rng.choice(non_edges)
                    res2 = _try_add_edge(adj, deg, a[0], a[1])
                    if res2 is not None:
                        adj, deg = res2
                        count += 1
                        break
            elif worst_edge is None:
                stall_count = 0
    return adj, deg, count

def _make_seed_k78():
    """Create K7,8 bipartite seed."""
    adj = [[0]*N for _ in range(N)]
    deg = [0]*N
    for a in range(7):
        for b in range(7, 15):
            adj[a][b] = adj[b][a] = 1
    for a in range(7):
        deg[a] = 8
    for b in range(7, 15):
        deg[b] = 7
    return adj, deg

def _make_seed_random(rng, density=0.4):
    """Create a random valid seed by greedily adding edges."""
    adj = [[0]*N for _ in range(N)]
    deg = [0]*N
    # Start with a few random edges
    for _ in range(5):
        u, v = rng.sample(range(N), 2)
        if not adj[u][v]:
            adj[u][v] = adj[v][u] = 1
            deg[u] += 1
            deg[v] += 1
    # Greedily add more edges
    for u in range(N):
        for v in range(u+1, N):
            if adj[u][v]:
                continue
            res = _try_add_edge(adj, deg, u, v)
            if res is not None:
                adj, deg = res
    return adj, deg

def construct():
    start = time.time()
    deadline = start + 82  # 8s safety margin before 90s kill
    
    best_adj = None
    best_deg = None
    best_count = 0
    
    # Multi-start with different seeds
    seeds = []
    
    # Seed 1: K7,8 bipartite
    seeds.append(("k78", _make_seed_k78()))
    
    # Seed 2-5: Random seeds with different densities
    rng_init = random.Random(42)
    for i in range(4):
        adj, deg = _make_seed_random(rng_init, density=0.3 + 0.1*i)
        seeds.append((f"random_{i}", (adj, deg)))
    
    # Process each seed
    num_seeds = len(seeds)
    time_per_seed = (deadline - start) / num_seeds
    
    for idx, (name, (adj, deg)) in enumerate(seeds):
        seed_deadline = start + (idx + 1) * time_per_seed
        if time.time() > seed_deadline - 2:
            break
        
        # Greedy add phase
        adj, deg = _greedy_add(adj, deg, seed_deadline - 3)
        count = _edge_count(adj)
        
        # Local search phase
        rng = random.Random(42 + idx * 1000)
        adj, deg, count = _local_search(adj, deg, seed_deadline - 1, rng, max_iter=100000)
        
        # Final greedy add
        adj, deg = _greedy_add(adj, deg, seed_deadline - 0.5)
        count = _edge_count(adj)
        
        if count > best_count:
            best_count = count
            best_adj = [row[:] for row in adj]
            best_deg = deg[:]
    
    # If we have time left, do more local search on the best solution
    if best_adj is not None and time.time() < deadline - 3:
        remaining = deadline - 3 - time.time()
        if remaining > 1:
            rng = random.Random(999)
            adj, deg, count = _local_search(best_adj, best_deg, deadline - 1, rng, max_iter=300000)
            adj, deg = _greedy_add(adj, deg, deadline - 0.5)
            count = _edge_count(adj)
            if count > best_count:
                best_count = count
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
    
    # Final verification: ensure the result is valid
    if best_adj is not None:
        if not _is_valid_fast(best_adj, best_deg):
            # Fallback to K7,8 if somehow invalid
            adj, deg = _make_seed_k78()
            best_adj = adj
            best_deg = deg
            best_count = _edge_count(adj)
    
    # Convert to edge list
    if best_adj is None:
        # Fallback: return K7,8
        adj, deg = _make_seed_k78()
        best_adj = adj
        best_deg = deg
    
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if best_adj[u][v]:
                edges.append([int(u), int(v)])
    return edges
# EVOLVE-BLOCK-END
