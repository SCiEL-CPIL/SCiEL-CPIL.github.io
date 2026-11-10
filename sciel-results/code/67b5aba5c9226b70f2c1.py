import time
import random
import heapq
from collections import defaultdict


def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _edge_set(edges):
    return {tuple(sorted(e)) for e in edges}


def _is_simple(path):
    return len(set(path)) == len(path)


def _try_merge(paths, i, j):
    P1 = list(paths[i])
    P2 = list(paths[j])
    if P1[-1] == P2[0]:
        merged = P1 + P2[1:]
        if _is_simple(merged):
            return merged
    if P1[-1] == P2[-1]:
        P2r = P2[::-1]
        merged = P1 + P2r[1:]
        if _is_simple(merged):
            return merged
    if P1[0] == P2[0]:
        P1r = P1[::-1]
        merged = P1r + P2[1:]
        if _is_simple(merged):
            return merged
    if P1[0] == P2[-1]:
        P1r = P1[::-1]
        merged = P1r + P2[1:]
        if _is_simple(merged):
            return merged
    return None


def _improve_by_merge(paths, max_iters=1000):
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        n = len(paths)
        for i in range(n):
            for j in range(i + 1, n):
                merged = _try_merge(paths, i, j)
                if merged is not None:
                    paths[i] = merged
                    paths.pop(j)
                    improved = True
                    break
            if improved:
                break
    return paths


def _pairing_path_cover(edges, adj, rng):
    """
    Structural approach: model path decomposition as edge-pairing.
    At each vertex, pair edges to minimize breaks. Use greedy with
    priority on extending paths that have the most remaining options.
    """
    remaining = set(_edge_set(edges))
    if not remaining:
        return []
    
    # Track degree
    deg = defaultdict(int)
    for u, v in remaining:
        deg[u] += 1
        deg[v] += 1
    
    # Active paths: list of [path_list, set_of_edges_used, current_end_vertex]
    # We'll manage a pool of open path-ends
    # Each path-end is (vertex, path_id)
    
    paths = []  # list of vertex lists
    path_edges = []  # set of edges per path
    open_ends = []  # (vertex, path_index) for each open end
    
    # Start: pick vertices with odd degree first (they must be endpoints)
    odd_vertices = [v for v in deg if deg[v] % 2 == 1]
    rng.shuffle(odd_vertices)
    
    # Process: repeatedly pick an open end and extend it
    # Priority: extend ends at vertices with high remaining degree (more options)
    # to avoid getting stuck
    
    while remaining:
        # If no open ends, start a new path
        if not open_ends:
            # Pick start vertex: prefer odd degree, then high degree
            candidates = [v for v in deg if deg[v] > 0]
            if not candidates:
                break
            odd_c = [v for v in candidates if deg[v] % 2 == 1]
            if odd_c and rng.random() < 0.8:
                start = rng.choice(odd_c)
            else:
                # Pick highest degree
                max_d = max(deg[v] for v in candidates)
                high = [v for v in candidates if deg[v] == max_d]
                start = rng.choice(high)
            
            path = [start]
            p_edges = set()
            pid = len(paths)
            paths.append(path)
            path_edges.append(p_edges)
            open_ends.append((start, pid))
        
        # Pick which open end to extend: prefer the one at vertex with most remaining edges
        # This avoids creating dead-ends
        best_idx = 0
        best_score = -1
        # Sample a few for speed
        check_count = min(len(open_ends), 20)
        indices = list(range(len(open_ends)))
        rng.shuffle(indices)
        for idx in indices[:check_count]:
            v, pid = open_ends[idx]
            # Score: number of available edges from v
            avail = 0
            for u in adj[v]:
                e = tuple(sorted((v, u)))
                if e in remaining:
                    avail += 1
            # Also consider: if this is the only edge at v, it's urgent
            if avail == 0:
                score = -1  # Can't extend, skip
            else:
                score = avail * 10 + (1 if deg.get(v, 0) % 2 == 1 else 0)
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_score < 0:
            # No open end can be extended; start new path
            candidates = [v for v in deg if deg[v] > 0]
            if not candidates:
                break
            odd_c = [v for v in candidates if deg[v] % 2 == 1]
            if odd_c:
                start = rng.choice(odd_c)
            else:
                max_d = max(deg[v] for v in candidates)
                high = [v for v in candidates if deg[v] == max_d]
                start = rng.choice(high)
            path = [start]
            p_edges = set()
            pid = len(paths)
            paths.append(path)
            path_edges.append(p_edges)
            open_ends.append((start, pid))
            continue
        
        v, pid = open_ends[best_idx]
        path = paths[pid]
        p_edges = path_edges[pid]
        
        # Find best next vertex: prefer unvisited, then low degree (avoid trapping)
        visited = set(path)
        nbrs = []
        for u in adj[v]:
            e = tuple(sorted((v, u)))
            if e in remaining:
                nbrs.append((u, e))
        
        if not nbrs:
            # This end is stuck; remove it
            open_ends.pop(best_idx)
            continue
        
        # Choose next: prefer unvisited, then min degree
        unvis = [(u, e) for u, e in nbrs if u not in visited]
        if unvis:
            min_d = min(deg[u] for u, e in unvis)
            best_cands = [(u, e) for u, e in unvis if deg[u] == min_d]
            nxt, e = rng.choice(best_cands)
        else:
            # All neighbors visited; pick min degree to minimize damage
            min_d = min(deg[u] for u, e in nbrs)
            best_cands = [(u, e) for u, e in nbrs if deg[u] == min_d]
            nxt, e = rng.choice(best_cands)
        
        # Extend
        remaining.discard(e)
        deg[v] -= 1
        deg[nxt] -= 1
        if deg[v] <= 0:
            del deg[v]
        if deg[nxt] <= 0:
            del deg[nxt]
        
        path.append(nxt)
        p_edges.add(e)
        
        # Update open ends: remove old end, add new end
        open_ends[best_idx] = (nxt, pid)
        
        # If nxt has no more edges, this end is done
        if deg.get(nxt, 0) == 0:
            # Check if any remaining edges incident to nxt
            has_edge = False
            for u in adj[nxt]:
                ee = tuple(sorted((nxt, u)))
                if ee in remaining:
                    has_edge = True
                    break
            if not has_edge:
                open_ends.pop(best_idx)
    
    # Filter out single-vertex paths
    result = []
    for p in paths:
        if len(p) >= 2:
            result.append(p)
    
    # Add any remaining edges as single-edge paths
    for e in remaining:
        result.append([e[0], e[1]])
    
    return result


def _eulerian_decompose(edges, adj, seed=12345):
    """Eulerian decomposition with multiple pairings."""
    remaining = set(_edge_set(edges))
    if not remaining:
        return []
    
    deg = defaultdict(int)
    for u, v in remaining:
        deg[u] += 1
        deg[v] += 1
    
    odd = [v for v, d in deg.items() if d % 2 == 1]
    n_odd = len(odd)
    
    dummy_edges = []
    rng = random.Random(seed)
    if n_odd > 0:
        rng.shuffle(odd)
        for i in range(0, n_odd, 2):
            if i + 1 < n_odd:
                dummy_edges.append((odd[i], odd[i+1]))
    
    aug_edges = list(remaining) + [(a, b) for a, b in dummy_edges]
    
    edge_index = defaultdict(list)
    for idx, (u, v) in enumerate(aug_edges):
        edge_index[u].append((v, idx))
        edge_index[v].append((u, idx))
    
    circuit = []
    stack = []
    start_v = aug_edges[0][0]
    stack.append(start_v)
    used_edges = set()
    
    while stack:
        v = stack[-1]
        found = False
        while edge_index.get(v):
            nb, idx = edge_index[v].pop()
            if idx not in used_edges:
                used_edges.add(idx)
                stack.append(nb)
                found = True
                break
        if not found:
            circuit.append(stack.pop())
    
    circuit.reverse()
    
    dummy_set = {tuple(sorted((a, b))) for a, b in dummy_edges}
    
    paths = []
    current_path = [circuit[0]]
    for i in range(len(circuit) - 1):
        u, v = circuit[i], circuit[i+1]
        e = tuple(sorted((u, v)))
        if e in dummy_set:
            if len(current_path) >= 2:
                paths.append(current_path)
            current_path = [v]
        else:
            current_path.append(v)
    
    if len(current_path) >= 2:
        paths.append(current_path)
    
    # Fix simplicity
    fixed_paths = []
    for trail in paths:
        if _is_simple(trail):
            fixed_paths.append(trail)
        else:
            i = 0
            n = len(trail)
            while i < n - 1:
                cur = [trail[i]]
                in_cur = {trail[i]}
                j = i + 1
                while j < n and trail[j] not in in_cur:
                    cur.append(trail[j])
                    in_cur.add(trail[j])
                    j += 1
                if len(cur) >= 2:
                    fixed_paths.append(cur)
                i = j
    
    return fixed_paths


def _greedy_path_cover(edges, adj, rng):
    """Original greedy as fallback."""
    remaining = set(_edge_set(edges))
    deg = defaultdict(int)
    for u, v in remaining:
        deg[u] += 1
        deg[v] += 1
    
    paths = []
    while remaining:
        candidates = [v for v, d in deg.items() if d > 0]
        if not candidates:
            break
        odd = [v for v in candidates if deg[v] % 2 == 1]
        leaves = [v for v in candidates if deg[v] == 1]
        if odd and rng.random() < 0.7:
            start = rng.choice(odd)
        elif leaves and rng.random() < 0.5:
            start = rng.choice(leaves)
        else:
            min_d = min(deg[v] for v in candidates)
            low = [v for v in candidates if deg[v] == min_d]
            start = rng.choice(low)
        
        path = [start]
        in_path = {start}
        while True:
            v = path[-1]
            nbrs = []
            for u in adj[v]:
                e = tuple(sorted((v, u)))
                if e in remaining:
                    nbrs.append(u)
            if not nbrs:
                break
            unvis = [u for u in nbrs if u not in in_path]
            if unvis:
                min_d = min(deg[u] for u in unvis)
                best = [u for u in unvis if deg[u] == min_d]
                nxt = rng.choice(best)
            else:
                break
            e = tuple(sorted((v, nxt)))
            remaining.discard(e)
            deg[v] -= 1
            deg[nxt] -= 1
            if deg[v] <= 0:
                del deg[v]
            if deg[nxt] <= 0:
                del deg[nxt]
            path.append(nxt)
            in_path.add(nxt)
        if len(path) >= 2:
            paths.append(path)
    
    for e in remaining:
        paths.append([e[0], e[1]])
    return paths


def _solve_one(edges, time_limit=40.0):
    adj = _build_adj(edges)
    best = None
    best_count = float('inf')
    t0 = time.time()
    
    # Strategy 1: Pairing-based path cover (structural)
    for trial in range(500):
        if time.time() - t0 > time_limit - 2.0:
            break
        seed = trial * 7919 + 13
        rng = random.Random(seed)
        paths = _pairing_path_cover(edges, adj, rng)
        paths = _improve_by_merge(paths)
        cnt = len(paths)
        if cnt < best_count:
            best_count = cnt
            best = [list(p) for p in paths]
            if best_count <= 1:
                break
    
    # Strategy 2: Eulerian decomposition
    if time.time() - t0 < time_limit - 2.0:
        for trial in range(150):
            if time.time() - t0 > time_limit - 2.0:
                break
            paths = _eulerian_decompose(edges, adj, seed=1000 + trial)
            if paths:
                paths = _improve_by_merge(paths)
                cnt = len(paths)
                if cnt < best_count:
                    best_count = cnt
                    best = [list(p) for p in paths]
                    if best_count <= 1:
                        break
    
    # Strategy 3: Original greedy
    if time.time() - t0 < time_limit - 1.0:
        for trial in range(300):
            if time.time() - t0 > time_limit - 1.0:
                break
            seed = 200000 + trial * 4931
            rng = random.Random(seed)
            paths = _greedy_path_cover(edges, adj, rng)
            paths = _improve_by_merge(paths)
            cnt = len(paths)
            if cnt < best_count:
                best_count = cnt
                best = [list(p) for p in paths]
                if best_count <= 1:
                    break
    
    if best is None:
        best = _greedy_path_cover(edges, adj, random.Random(42))
    
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    n_graphs = len(specs)
    
    for i, (n, p, seed) in enumerate(specs):
        remaining = deadline - time.time()
        if remaining < 3:
            edges = _instance(n, p, seed)
            adj = _build_adj(edges)
            results.append(_greedy_path_cover(edges, adj, random.Random(1)))
            continue
        alloc = max(2.0, (remaining - 1.0) / (n_graphs - i))
        edges = _instance(n, p, seed)
        paths = _solve_one(edges, time_limit=alloc)
        paths = _improve_by_merge(paths)
        results.append(paths)
    
    return results


if __name__ == "__main__":
    res = construct()
    for i, paths in enumerate(res):
        print(f"Graph {i}: {len(paths)} paths")
    print(f"Total: {sum(len(p) for p in res)}")
