import time
import random

N = 20
C = 20
DEADLINE = 145.0


def _validate(edges):
    """Return True iff every 4-cycle is rainbow."""
    if not edges:
        return True
    adj = {}
    neigh = [set() for _ in range(N)]
    for u, v, c in edges:
        adj[(u, v)] = c
        adj[(v, u)] = c
        neigh[u].add(v)
        neigh[v].add(u)

    for u in range(N):
        for v in neigh[u]:
            if u >= v:
                continue
            common = neigh[u] & neigh[v]
            common.discard(u)
            common.discard(v)
            common_list = list(common)
            for i in range(len(common_list)):
                x = common_list[i]
                for j in range(i + 1, len(common_list)):
                    y = common_list[j]
                    cols = {adj[(u, x)], adj[(x, v)], adj[(v, y)], adj[(y, u)]}
                    if len(cols) != 4:
                        return False
    return True


def _check_add(adj, neigh, u, v, c):
    """Check if adding edge (u,v) with color c maintains rainbow 4-cycles.
    Assumes proper edge coloring is maintained separately."""
    # 4-cycles: u-v-x-y-u where x~v, y~u, x!=y, x~y
    for x in neigh[v]:
        if x == u:
            continue
        for y in neigh[u]:
            if y == v or y == x:
                continue
            if y not in neigh[x]:
                continue
            # cycle u-v-x-y-u
            # colors: c, adj[(v,x)], adj[(x,y)], adj[(y,u)]
            if c == adj[(v, x)] or c == adj[(x, y)] or c == adj[(y, u)]:
                return False
            if adj[(v, x)] == adj[(x, y)] or adj[(v, x)] == adj[(y, u)]:
                return False
            if adj[(x, y)] == adj[(y, u)]:
                return False
    return True


def _greedy_build(order, color_order_fn, rng):
    """Build graph greedily. order is list of (u,v) pairs.
    color_order_fn(u,v,used_u,used_v) returns list of colors to try."""
    adj = {}
    neigh = [set() for _ in range(N)]
    color_at = [[None]*N for _ in range(N)]  # color_at[u][v] = color of edge u-v
    edges = []

    for (u, v) in order:
        # Determine available colors (proper edge coloring)
        used_u = set(color_at[u][w] for w in neigh[u] if color_at[u][w] is not None)
        used_v = set(color_at[v][w] for w in neigh[v] if color_at[v][w] is not None)
        available = [c for c in range(C) if c not in used_u and c not in used_v]
        
        if not available:
            continue
        
        # Try colors in order
        for c in available:
            if _check_add(adj, neigh, u, v, c):
                # Add edge
                adj[(u, v)] = c
                adj[(v, u)] = c
                neigh[u].add(v)
                neigh[v].add(u)
                color_at[u][v] = c
                color_at[v][u] = c
                edges.append([u, v, c])
                break
    
    return edges


def _greedy_with_shuffled_colors(order, rng):
    """Greedy with random color ordering among available colors."""
    adj = {}
    neigh = [set() for _ in range(N)]
    color_at = [[None]*N for _ in range(N)]
    edges = []

    for (u, v) in order:
        used_u = set(color_at[u][w] for w in neigh[u] if color_at[u][w] is not None)
        used_v = set(color_at[v][w] for w in neigh[v] if color_at[v][w] is not None)
        available = [c for c in range(C) if c not in used_u and c not in used_v]
        
        if not available:
            continue
        
        rng.shuffle(available)
        for c in available:
            if _check_add(adj, neigh, u, v, c):
                adj[(u, v)] = c
                adj[(v, u)] = c
                neigh[u].add(v)
                neigh[v].add(u)
                color_at[u][v] = c
                color_at[v][u] = c
                edges.append([u, v, c])
                break
    
    return edges


def _try_add_edges(edges, adj, neigh, color_at, rng, time_budget):
    """Try to add more edges to an existing valid graph."""
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    existing = set()
    for u, v, c in edges:
        existing.add((min(u,v), max(u,v)))
    
    candidates = [(u,v) for (u,v) in all_pairs if (u,v) not in existing]
    rng.shuffle(candidates)
    
    for (u, v) in candidates:
        if time.time() > time_budget:
            break
        used_u = set(color_at[u][w] for w in neigh[u] if color_at[u][w] is not None)
        used_v = set(color_at[v][w] for w in neigh[v] if color_at[v][w] is not None)
        available = [c for c in range(C) if c not in used_u and c not in used_v]
        
        if not available:
            continue
        
        rng.shuffle(available)
        for c in available:
            if _check_add(adj, neigh, u, v, c):
                adj[(u, v)] = c
                adj[(v, u)] = c
                neigh[u].add(v)
                neigh[v].add(u)
                color_at[u][v] = c
                color_at[v][u] = c
                edges.append([u, v, c])
                break
    
    return edges


def _remove_edge_and_retry(edges, adj, neigh, color_at, u, v, rng):
    """Remove edge (u,v), then try to re-add it with a different color or skip it
    to allow other edges to be added."""
    # Find and remove edge (u,v)
    idx = None
    for i, (a, b, c) in enumerate(edges):
        if (a == u and b == v) or (a == v and b == u):
            idx = i
            break
    if idx is None:
        return edges, adj, neigh, color_at
    
    a, b, c = edges[idx]
    edges.pop(idx)
    del adj[(a, b)]
    del adj[(b, a)]
    neigh[a].discard(b)
    neigh[b].discard(a)
    color_at[a][b] = None
    color_at[b][a] = None
    
    return edges, adj, neigh, color_at


def _local_search(edges, adj, neigh, color_at, rng, time_budget):
    """Local search: try removing edges that block others, then re-add."""
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    
    for _ in range(500):
        if time.time() > time_budget:
            break
        
        existing = set()
        for u, v, c in edges:
            existing.add((min(u,v), max(u,v)))
        
        candidates = [(u,v) for (u,v) in all_pairs if (u,v) not in existing]
        if not candidates:
            break
        
        # Try each candidate a few times
        rng.shuffle(candidates)
        added_any = False
        for (u, v) in candidates[:50]:
            if time.time() > time_budget:
                break
            used_u = set(color_at[u][w] for w in neigh[u] if color_at[u][w] is not None)
            used_v = set(color_at[v][w] for w in neigh[v] if color_at[v][w] is not None)
            available = [c for c in range(C) if c not in used_u and c not in used_v]
            
            if not available:
                continue
            
            rng.shuffle(available)
            for c in available:
                if _check_add(adj, neigh, u, v, c):
                    adj[(u, v)] = c
                    adj[(v, u)] = c
                    neigh[u].add(v)
                    neigh[v].add(u)
                    color_at[u][v] = c
                    color_at[v][u] = c
                    edges.append([u, v, c])
                    added_any = True
                    break
        
        if not added_any:
            # Try removing a random edge and see if we can add 2+ back
            if edges:
                idx = rng.randrange(len(edges))
                a, b, c = edges[idx]
                edges, adj, neigh, color_at = _remove_edge_and_retry(
                    edges, adj, neigh, color_at, a, b, rng)
                # Now try to add edges again
                existing2 = set()
                for u, v, cc in edges:
                    existing2.add((min(u,v), max(u,v)))
                candidates2 = [(u,v) for (u,v) in all_pairs if (u,v) not in existing2]
                rng.shuffle(candidates2)
                for (u, v) in candidates2[:30]:
                    if time.time() > time_budget:
                        break
                    used_u = set(color_at[u][w] for w in neigh[u] if color_at[u][w] is not None)
                    used_v = set(color_at[v][w] for w in neigh[v] if color_at[v][w] is not None)
                    available2 = [c for c in range(C) if c not in used_u and c not in used_v]
                    if not available2:
                        continue
                    rng.shuffle(available2)
                    for c in available2:
                        if _check_add(adj, neigh, u, v, c):
                            adj[(u, v)] = c
                            adj[(v, u)] = c
                            neigh[u].add(v)
                            neigh[v].add(u)
                            color_at[u][v] = c
                            color_at[v][u] = c
                            edges.append([u, v, c])
                            break
    
    return edges


def _structured_build(rng, time_budget):
    """Build using a structured approach: try to make a near-complete graph."""
    # Strategy: build in layers. First add a spanning structure, then fill in.
    # Use vertex ordering that helps with 4-cycle constraints.
    
    # Random vertex permutation
    perm = list(range(N))
    rng.shuffle(perm)
    
    # Build edges in order: for each pair (i,j) in perm order
    order = []
    for i in range(N):
        for j in range(i+1, N):
            order.append((perm[i], perm[j]))
    
    rng.shuffle(order)
    
    return _greedy_with_shuffled_colors(order, rng)


def construct():
    start = time.time()
    best_edges = []
    best_len = 0
    
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    
    # Multiple random greedy attempts
    for trial in range(30):
        if time.time() > DEADLINE - 20:
            break
        
        rng = random.Random(trial * 137 + 42)
        
        # Different strategies per trial
        if trial < 10:
            # Random edge order, random color order
            order = all_pairs[:]
            rng.shuffle(order)
            edges = _greedy_with_shuffled_colors(order, rng)
        elif trial < 20:
            # Structured: try to build dense subgraph
            edges = _structured_build(rng, DEADLINE - 20)
        else:
            # Try with a fixed color preference (low colors first)
            order = all_pairs[:]
            rng.shuffle(order)
            adj = {}
            neigh = [set() for _ in range(N)]
            color_at = [[None]*N for _ in range(N)]
            edges = []
            for (u, v) in order:
                used_u = set(color_at[u][w] for w in neigh[u] if color_at[u][w] is not None)
                used_v = set(color_at[v][w] for w in neigh[v] if color_at[v][w] is not None)
                available = [c for c in range(C) if c not in used_u and c not in used_v]
                if not available:
                    continue
                # Try low colors first
                for c in sorted(available):
                    if _check_add(adj, neigh, u, v, c):
                        adj[(u, v)] = c
                        adj[(v, u)] = c
                        neigh[u].add(v)
                        neigh[v].add(u)
                        color_at[u][v] = c
                        color_at[v][u] = c
                        edges.append([u, v, c])
                        break
        
        if len(edges) > best_len:
            best_len = len(edges)
            best_edges = edges[:]
    
    # Local search on best
    if best_edges and time.time() < DEADLINE - 10:
        adj = {}
        neigh = [set() for _ in range(N)]
        color_at = [[None]*N for _ in range(N)]
        for u, v, c in best_edges:
            adj[(u, v)] = c
            adj[(v, u)] = c
            neigh[u].add(v)
            neigh[v].add(u)
            color_at[u][v] = c
            color_at[v][u] = c
        
        rng = random.Random(12345)
        budget = DEADLINE - 5
        improved = _local_search(best_edges[:], adj, neigh, color_at, rng, budget)
        
        if len(improved) > best_len:
            best_len = len(improved)
            best_edges = improved
    
    # Final attempt: try to add edges to best
    if best_edges and time.time() < DEADLINE - 3:
        adj = {}
        neigh = [set() for _ in range(N)]
        color_at = [[None]*N for _ in range(N)]
        for u, v, c in best_edges:
            adj[(u, v)] = c
            adj[(v, u)] = c
            neigh[u].add(v)
            neigh[v].add(u)
            color_at[u][v] = c
            color_at[v][u] = c
        
        rng = random.Random(99999)
        final = _try_add_edges(best_edges[:], adj, neigh, color_at, rng, DEADLINE - 1)
        if len(final) > best_len:
            best_edges = final
    
    if not best_edges:
        # Fallback: star
        return [[0, v, v] for v in range(1, N)]
    
    return best_edges


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res))
    print("valid:", _validate(res))
