import time
import random
import math

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


def _count_c4s(adj_matrix):
    """Count number of 4-cycles in a graph given as adjacency matrix (0/1)."""
    count = 0
    for u in range(N):
        for v in range(u + 1, N):
            if adj_matrix[u][v]:
                common = 0
                for w in range(N):
                    if w != u and w != v and adj_matrix[u][w] and adj_matrix[v][w]:
                        common += 1
                count += common
    return count // 2  # each C4 counted twice (once per pair of opposite edges)


def _check_add(adj, neigh, u, v, c):
    """Check if adding edge (u,v) with color c maintains rainbow 4-cycles.
    Assumes proper edge coloring is maintained separately."""
    for x in neigh[v]:
        if x == u:
            continue
        for y in neigh[u]:
            if y == v or y == x:
                continue
            if y not in neigh[x]:
                continue
            if c == adj[(v, x)] or c == adj[(x, y)] or c == adj[(y, u)]:
                return False
            if adj[(v, x)] == adj[(x, y)] or adj[(v, x)] == adj[(y, u)]:
                return False
            if adj[(x, y)] == adj[(y, u)]:
                return False
    return True


def _check_add_c4_free(adj_matrix, u, v):
    """Check if adding edge (u,v) keeps the graph C4-free."""
    for w in range(N):
        if w != u and w != v and adj_matrix[u][w] and adj_matrix[v][w]:
            return False
    return True


def _greedy_build(order, color_order_fn, rng):
    """Build graph greedily. order is list of (u,v) pairs.
    color_order_fn(u,v,used_u,used_v) returns list of colors to try."""
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


def _greedy_c4_free(order, rng):
    """Build a C4-free graph greedily."""
    adj_matrix = [[0]*N for _ in range(N)]
    edges = []
    
    for (u, v) in order:
        if _check_add_c4_free(adj_matrix, u, v):
            adj_matrix[u][v] = 1
            adj_matrix[v][u] = 1
            edges.append((u, v))
    
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


def _color_c4_free(edges, rng):
    """Color a C4-free graph. Since there are no 4-cycles, any proper edge coloring works."""
    adj = {}
    neigh = [set() for _ in range(N)]
    color_at = [[None]*N for _ in range(N)]
    colored_edges = []
    
    for (u, v) in edges:
        used_u = set(color_at[u][w] for w in neigh[u] if color_at[u][w] is not None)
        used_v = set(color_at[v][w] for w in neigh[v] if color_at[v][w] is not None)
        available = [c for c in range(C) if c not in used_u and c not in used_v]
        
        if not available:
            continue
        
        rng.shuffle(available)
        c = available[0]
        adj[(u, v)] = c
        adj[(v, u)] = c
        neigh[u].add(v)
        neigh[v].add(u)
        color_at[u][v] = c
        color_at[v][u] = c
        colored_edges.append([u, v, c])
    
    return colored_edges


def _remove_edge_and_retry(edges, adj, neigh, color_at, u, v, rng):
    """Remove edge (u,v), then try to re-add it with a different color or skip it."""
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


def _local_search_c4_free(edge_set, adj_matrix, rng, time_budget):
    """Local search on C4-free graph: try to add edges, remove if blocked."""
    edges = list(edge_set)
    
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    
    while time.time() < time_budget:
        existing = set(edge_set)
        candidates = [(u,v) for (u,v) in all_pairs if (u,v) not in existing]
        if not candidates:
            break
        
        rng.shuffle(candidates)
        added_any = False
        for (u, v) in candidates:
            if time.time() > time_budget:
                break
            if _check_add_c4_free(adj_matrix, u, v):
                adj_matrix[u][v] = 1
                adj_matrix[v][u] = 1
                edge_set.add((u, v))
                added_any = True
        
        if not added_any:
            # Try removing a random edge and see if we can add more back
            if edges:
                e = rng.choice(edges)
                u, v = e
                adj_matrix[u][v] = 0
                adj_matrix[v][u] = 0
                edge_set.discard(e)
                edges.remove(e)
                
                # Try to add edges again
                existing2 = set(edge_set)
                candidates2 = [(u2,v2) for (u2,v2) in all_pairs if (u2,v2) not in existing2]
                rng.shuffle(candidates2)
                for (u2, v2) in candidates2:
                    if time.time() > time_budget:
                        break
                    if _check_add_c4_free(adj_matrix, u2, v2):
                        adj_matrix[u2][v2] = 1
                        adj_matrix[v2][u2] = 1
                        edge_set.add((u2, v2))
    
    return edge_set


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
            if edges:
                idx = rng.randrange(len(edges))
                a, b, c = edges[idx]
                edges, adj, neigh, color_at = _remove_edge_and_retry(
                    edges, adj, neigh, color_at, a, b, rng)
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


def _simulated_annealing_c4_free(edge_set, adj_matrix, rng, time_budget):
    """Simulated annealing on C4-free graph structure."""
    edges = list(edge_set)
    current_energy = -len(edges)  # maximize edges
    best_edges = set(edges)
    best_energy = current_energy
    
    start_temp = 1.0
    end_temp = 0.01
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    
    n_steps = 0
    max_steps = 100000
    
    while time.time() < time_budget and n_steps < max_steps:
        n_steps += 1
        temp = start_temp * (end_temp / start_temp) ** (n_steps / max_steps)
        
        existing = set(edge_set)
        candidates = [(u,v) for (u,v) in all_pairs if (u,v) not in existing]
        
        if candidates:
            u, v = rng.choice(candidates)
            if _check_add_c4_free(adj_matrix, u, v):
                adj_matrix[u][v] = 1
                adj_matrix[v][u] = 1
                edge_set.add((u, v))
                current_energy = -len(edge_set)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_edges = set(edge_set)
            else:
                # Can't add, try removing a random edge
                if edges:
                    e = rng.choice(edges)
                    eu, ev = e
                    adj_matrix[eu][ev] = 0
                    adj_matrix[ev][eu] = 0
                    edge_set.discard(e)
                    current_energy = -len(edge_set)
                    # Accept or reject
                    delta = current_energy - best_energy
                    if delta < 0 or rng.random() < math.exp(-delta / max(temp, 0.001)):
                        pass  # accept
                    else:
                        # Reject: put edge back
                        adj_matrix[eu][ev] = 1
                        adj_matrix[ev][eu] = 1
                        edge_set.add(e)
                        current_energy = -len(edge_set)
        else:
            break
    
    # Restore best
    edge_set.clear()
    edge_set.update(best_edges)
    adj_matrix = [[0]*N for _ in range(N)]
    for u, v in best_edges:
        adj_matrix[u][v] = 1
        adj_matrix[v][u] = 1
    
    return best_edges


def _structured_build(rng, time_budget):
    """Build using a structured approach: try to make a near-complete graph."""
    perm = list(range(N))
    rng.shuffle(perm)
    
    order = []
    for i in range(N):
        for j in range(i+1, N):
            order.append((perm[i], perm[j]))
    
    rng.shuffle(order)
    
    return _greedy_with_shuffled_colors(order, rng)


def _c4_free_construction(rng, time_budget):
    """Construct a dense C4-free graph, then color it."""
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    
    best_edges = set()
    best_len = 0
    
    # Multiple random greedy C4-free attempts
    for trial in range(20):
        if time.time() > time_budget - 30:
            break
        
        order = all_pairs[:]
        rng.shuffle(order)
        edges = _greedy_c4_free(order, rng)
        
        if len(edges) > best_len:
            best_len = len(edges)
            best_edges = set(edges)
    
    # Local search on C4-free graph
    if best_edges and time.time() < time_budget - 20:
        adj_matrix = [[0]*N for _ in range(N)]
        for u, v in best_edges:
            adj_matrix[u][v] = 1
            adj_matrix[v][u] = 1
        
        rng2 = random.Random(42)
        improved = _local_search_c4_free(best_edges, adj_matrix, rng2, time_budget - 10)
        if len(improved) > best_len:
            best_edges = improved
            best_len = len(improved)
    
    # Simulated annealing
    if best_edges and time.time() < time_budget - 5:
        adj_matrix = [[0]*N for _ in range(N)]
        for u, v in best_edges:
            adj_matrix[u][v] = 1
            adj_matrix[v][u] = 1
        
        rng3 = random.Random(123)
        sa_result = _simulated_annealing_c4_free(best_edges, adj_matrix, rng3, time_budget - 1)
        if len(sa_result) > best_len:
            best_edges = sa_result
            best_len = len(sa_result)
    
    # Color the C4-free graph
    colored = _color_c4_free(list(best_edges), rng)
    return colored


def construct():
    start = time.time()
    best_edges = []
    best_len = 0
    
    # Strategy 1: C4-free construction (most promising)
    if time.time() < DEADLINE - 30:
        rng_c4 = random.Random(777)
        c4_result = _c4_free_construction(rng_c4, DEADLINE - 30)
        if len(c4_result) > best_len:
            best_len = len(c4_result)
            best_edges = c4_result[:]
    
    # Strategy 2: Standard greedy with rainbow constraint
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    for trial in range(10):
        if time.time() > DEADLINE - 15:
            break
        
        rng = random.Random(trial * 137 + 42)
        order = all_pairs[:]
        rng.shuffle(order)
        edges = _greedy_with_shuffled_colors(order, rng)
        
        if len(edges) > best_len:
            best_len = len(edges)
            best_edges = edges[:]
    
    # Local search on best
    if best_edges and time.time() < DEADLINE - 5:
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
        budget = DEADLINE - 2
        improved = _local_search(best_edges[:], adj, neigh, color_at, rng, budget)
        
        if len(improved) > best_len:
            best_len = len(improved)
            best_edges = improved
    
    # Final attempt: try to add edges to best
    if best_edges and time.time() < DEADLINE - 1:
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
        final = _try_add_edges(best_edges[:], adj, neigh, color_at, rng, DEADLINE - 0.5)
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
