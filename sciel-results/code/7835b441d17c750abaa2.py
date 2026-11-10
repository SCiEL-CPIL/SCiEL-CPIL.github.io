import time
import random
from collections import defaultdict, deque


def _instance(n, p, seed):
    """Exact copy of the verifier's instance generator."""
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


def _path_edges_set(path):
    return {tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)}


def _try_merge(paths, i, j):
    """Try to merge paths[i] and paths[j] by connecting endpoints."""
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


def _improve_by_merge(paths, max_iters=500):
    """Repeatedly try to merge any two paths until no more merges possible."""
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


def _greedy_path_cover(edges, adj, rng):
    """
    Greedy path cover with degree-based heuristics.
    """
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


def _eulerian_decompose(edges, adj):
    """
    Eulerian decomposition: pair odd vertices, find Eulerian circuit,
    split at dummy edges to get trails, then fix simplicity.
    """
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
    rng = random.Random(12345)
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
    
    # Fix simplicity by splitting at repeated vertices
    fixed_paths = []
    for trail in paths:
        if _is_simple(trail):
            fixed_paths.append(trail)
        else:
            seen = {}
            split_idx = None
            for i, v in enumerate(trail):
                if v in seen:
                    split_idx = i
                    break
                seen[v] = i
            
            if split_idx is not None:
                part1 = trail[:split_idx+1]
                part2 = trail[split_idx:]
                for p in [part1, part2]:
                    fixed_paths.append(p)
            else:
                fixed_paths.append(trail)
    
    return fixed_paths


def _local_search(paths, adj, remaining_edges, max_iters=200):
    """
    Local search: try 2-opt style swaps to reduce number of paths.
    For each pair of paths, try to rearrange edges to create a valid merge.
    """
    # Try to find edge swaps that enable merges
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        n = len(paths)
        for i in range(n):
            for j in range(i + 1, n):
                # Try all endpoint connections
                merged = _try_merge(paths, i, j)
                if merged is not None:
                    paths[i] = merged
                    paths.pop(j)
                    improved = True
                    break
            if improved:
                break
    
    return paths


def _solve_one(edges, time_limit=40.0):
    """Solve one graph instance, trying multiple strategies."""
    adj = _build_adj(edges)
    best = None
    best_count = float('inf')
    t0 = time.time()
    
    # Strategy 1: Greedy with multiple seeds
    for trial in range(500):
        if time.time() - t0 > time_limit - 1.0:
            break
        seed = trial * 7919 + 13
        rng = random.Random(seed)
        paths = _greedy_path_cover(edges, adj, rng)
        paths = _improve_by_merge(paths)
        cnt = len(paths)
        if cnt < best_count:
            best_count = cnt
            best = [list(p) for p in paths]
            if best_count <= 1:
                break
    
    # Strategy 2: Eulerian decomposition
    if time.time() - t0 < time_limit - 1.0:
        paths = _eulerian_decompose(edges, adj)
        if paths:
            paths = _improve_by_merge(paths)
            cnt = len(paths)
            if cnt < best_count:
                best_count = cnt
                best = [list(p) for p in paths]
    
    # Strategy 3: Greedy with leaf-first selection
    for trial in range(300):
        if time.time() - t0 > time_limit - 0.5:
            break
        seed = 100000 + trial * 6151
        rng = random.Random(seed)
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
            
            leaves = [v for v in candidates if deg[v] == 1]
            odd = [v for v in candidates if deg[v] % 2 == 1]
            
            if leaves and rng.random() < 0.8:
                start = rng.choice(leaves)
            elif odd and rng.random() < 0.6:
                start = rng.choice(odd)
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
                    best_n = [u for u in unvis if deg[u] == min_d]
                    nxt = rng.choice(best_n)
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
    deadline = t0 + 148.0
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
        results.append(paths)
    
    return results


if __name__ == "__main__":
    res = construct()
    for i, paths in enumerate(res):
        print(f"Graph {i}: {len(paths)} paths")
    print(f"Total: {sum(len(p) for p in res)}")
