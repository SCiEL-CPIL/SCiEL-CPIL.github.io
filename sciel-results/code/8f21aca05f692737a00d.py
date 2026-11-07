"""Refined dense graph construction with directed swaps and multiple restarts."""
# EVOLVE-BLOCK-START
import time
import random

N = 15

def _is_valid_fast(adj, deg):
    """Fast validity check using degree-count method per edge."""
    for u in range(N):
        for v in range(u + 1, N):
            if not adj[u][v]:
                continue
            deg_a = {}
            deg_b = {}
            for s in range(N):
                if s == u or s == v:
                    continue
                if adj[s][u]:
                    d = deg[s]
                    deg_a[d] = deg_a.get(d, 0) + 1
                if adj[s][v]:
                    d = deg[s]
                    deg_b[d] = deg_b.get(d, 0) + 1
            for d in deg_a:
                if d in deg_b:
                    ca = deg_a[d]
                    cb = deg_b[d]
                    if ca + cb >= 2:
                        if ca == 1 and cb == 1:
                            s_val = -1
                            for s in range(N):
                                if s != u and s != v and adj[s][u] and deg[s] == d:
                                    s_val = s
                                    break
                            if s_val != -1 and adj[s_val][v]:
                                continue
                        return False
    return True

def _blocking_edges_for_add(adj, deg, a, b):
    """Find edges that block adding (a,b). Returns list of (u,v) edges to remove."""
    blockers = []
    # Paths s-u-v-t where (a,b) is the middle edge
    # s~a, t~b, s!=t, deg[s]==deg[t]
    for s in range(N):
        if s == a or s == b or not adj[s][a]:
            continue
        for t in range(N):
            if t == a or t == b or t == s or not adj[t][b]:
                continue
            if deg[s] == deg[t]:
                # Edge (s,a) or (t,b) or (s,b) or (t,a) blocks this
                # Removing (s,a) breaks path s-a-b-t
                # Removing (t,b) breaks path s-a-b-t
                blockers.append((s, a))
                blockers.append((t, b))
    # Paths s-v-u-t where (a,b) is traversed b->a
    for s in range(N):
        if s == a or s == b or not adj[s][b]:
            continue
        for t in range(N):
            if t == a or t == b or t == s or not adj[t][a]:
                continue
            if deg[s] == deg[t]:
                blockers.append((s, b))
                blockers.append((t, a))
    return list(set(blockers))

def _try_add_edge(adj, deg, u, v):
    if u == v or adj[u][v]:
        return None
    new_adj = [row[:] for row in adj]
    new_adj[u][v] = new_adj[v][u] = 1
    new_deg = deg[:]
    new_deg[u] += 1
    new_deg[v] += 1
    if _is_valid_fast(new_adj, new_deg):
        return (new_adj, new_deg)
    return None

def _try_swap(adj, deg, remove, add):
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
    if _is_valid_fast(new_adj, new_deg):
        return (new_adj, new_deg)
    return None

def _edge_count(adj):
    return sum(sum(row) for row in adj) // 2

def _greedy_add(adj, deg, deadline):
    """Try to add any missing edge greedily."""
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
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

def _directed_swap_search(adj, deg, deadline, seed):
    """Swap search targeting blocking edges for potential additions."""
    random.seed(seed)
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    while time.time() < deadline:
        # Find a non-edge and its blockers
        non_edges = [(u, v) for (u, v) in all_edges if not adj[u][v]]
        if not non_edges:
            break
        
        # Try each non-edge, find blockers, try to remove one blocker
        random.shuffle(non_edges)
        found_improvement = False
        for (a, b) in non_edges:
            if time.time() > deadline:
                break
            blockers = _blocking_edges_for_add(adj, deg, a, b)
            if not blockers:
                # No blockers, just try adding
                res = _try_add_edge(adj, deg, a, b)
                if res:
                    adj, deg = res
                    found_improvement = True
                    break
                continue
            # Try removing each blocker
            random.shuffle(blockers)
            for (ru, rv) in blockers:
                if time.time() > deadline:
                    break
                # Normalize edge
                if ru > rv:
                    ru, rv = rv, ru
                res = _try_swap(adj, deg, (ru, rv), (a, b))
                if res:
                    adj, deg = res
                    found_improvement = True
                    break
            if found_improvement:
                break
        
        if not found_improvement:
            # Try random swaps to escape
            present = [(u, v) for (u, v) in all_edges if adj[u][v]]
            if not present:
                break
            for _ in range(100):
                if time.time() > deadline:
                    break
                add = random.choice(non_edges) if non_edges else None
                if not add:
                    break
                remove = random.choice(present)
                if remove == add:
                    continue
                res = _try_swap(adj, deg, remove, add)
                if res:
                    adj, deg = res
                    # Update non_edges
                    non_edges = [(u, v) for (u, v) in all_edges if not adj[u][v]]
                    break
    return adj, deg

def construct():
    start = time.time()
    deadline = start + 85  # 5s safety margin

    # Seed: K7,8 bipartite
    adj = [[0] * N for _ in range(N)]
    deg = [0] * N
    for a in range(7):
        for b in range(7, 15):
            adj[a][b] = adj[b][a] = 1
    for a in range(7):
        deg[a] = 8
    for b in range(7, 15):
        deg[b] = 7
    assert _is_valid_fast(adj, deg), "Seed K7,8 should be valid"

    best_adj = [row[:] for row in adj]
    best_deg = deg[:]
    best_count = _edge_count(best_adj)

    # Phase 1: Greedy edge addition
    adj, deg = _greedy_add(adj, deg, deadline - 40)
    cnt = _edge_count(adj)
    if cnt > best_count:
        best_adj = [row[:] for row in adj]
        best_deg = deg[:]
        best_count = cnt

    # Phase 2: Directed swap search with multiple restarts
    remaining_time = deadline - time.time()
    restarts = 3
    time_per_restart = remaining_time / restarts
    
    for r in range(restarts):
        restart_deadline = time.time() + time_per_restart
        adj, deg = _directed_swap_search(adj, deg, restart_deadline, seed=42 + r * 1000)
        cnt = _edge_count(adj)
        if cnt > best_count:
            best_adj = [row[:] for row in adj]
            best_deg = deg[:]
            best_count = cnt
        
        # Greedy add after each restart
        if time.time() < deadline - 5:
            adj, deg = _greedy_add(adj, deg, deadline - 5)
            cnt = _edge_count(adj)
            if cnt > best_count:
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
                best_count = cnt

    # Final polish: try all single-edge additions
    if time.time() < deadline - 1:
        for u in range(N):
            for v in range(u + 1, N):
                if best_adj[u][v]:
                    continue
                # Try adding to best_adj
                test_adj = [row[:] for row in best_adj]
                test_deg = best_deg[:]
                test_adj[u][v] = test_adj[v][u] = 1
                test_deg[u] += 1
                test_deg[v] += 1
                if _is_valid_fast(test_adj, test_deg):
                    best_adj = test_adj
                    best_deg = test_deg
                    best_count = _edge_count(best_adj)

    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u][v]:
                edges.append([int(u), int(v)])
    return edges
# EVOLVE-BLOCK-END
