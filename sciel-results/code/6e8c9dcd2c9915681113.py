import time
import random
import sys
from collections import defaultdict, deque
import heapq

def _instance(n, p, seed):
    """Exact copy of the verifier's instance generator."""
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_graph(edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_shortest_paths(adj, source):
    """BFS from source to all reachable vertices. Returns (dist, parent) dicts."""
    dist = {source: 0}
    parent = {source: None}
    queue = deque([source])
    while queue:
        v = queue.popleft()
        for u in adj[v]:
            if u not in dist:
                dist[u] = dist[v] + 1
                parent[u] = v
                queue.append(u)
    return dist, parent

def _reconstruct_path(parent, source, target):
    """Reconstruct path from source to target using parent pointers."""
    if target not in parent:
        return None
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        if cur == source:
            break
        cur = parent[cur]
    path.reverse()
    if path[0] != source:
        return None
    return path

def _min_cost_matching_odd(adj, odd_vertices, edges_set, deadline):
    """
    Find minimum-cost perfect matching on odd-degree vertices.
    Cost of matching (u,v) = shortest path distance between u and v.
    Uses Hungarian algorithm for small cases, greedy for larger.
    """
    k = len(odd_vertices)
    if k == 0:
        return []
    if k % 2 != 0:
        # Shouldn't happen in a valid graph, but handle gracefully
        return []
    
    # Compute all-pairs shortest paths among odd vertices
    # For each odd vertex, BFS to all other odd vertices
    dist_matrix = [[0] * k for _ in range(k)]
    parent_map = {}
    
    for i, src in enumerate(odd_vertices):
        dist, parent = _bfs_shortest_paths(adj, src)
        parent_map[src] = (dist, parent)
        for j, tgt in enumerate(odd_vertices):
            if i != j and tgt in dist:
                dist_matrix[i][j] = dist[tgt]
            elif i != j:
                dist_matrix[i][j] = 10000  # Large penalty if unreachable
    
    if k <= 12:
        # Use Hungarian algorithm (exact)
        try:
            return _hungarian_matching(dist_matrix, odd_vertices, parent_map)
        except:
            pass
    
    # Greedy fallback: repeatedly match closest pair
    matched = [False] * k
    pairs = []
    remaining = list(range(k))
    
    while remaining:
        best_cost = float('inf')
        best_i, best_j = -1, -1
        for idx_i in range(len(remaining)):
            for idx_j in range(idx_i + 1, len(remaining)):
                i, j = remaining[idx_i], remaining[idx_j]
                if not matched[i] and not matched[j]:
                    cost = dist_matrix[i][j]
                    if cost < best_cost:
                        best_cost = cost
                        best_i, best_j = i, j
        
        if best_i == -1:
            break
        
        matched[best_i] = True
        matched[best_j] = True
        pairs.append((odd_vertices[best_i], odd_vertices[best_j]))
        remaining = [x for x in remaining if x != best_i and x != best_j]
    
    return pairs

def _hungarian_matching(cost_matrix, odd_vertices, parent_map):
    """
    Hungarian algorithm for minimum-cost perfect matching.
    Returns list of (vertex_i, vertex_j) pairs.
    """
    n = len(odd_vertices)
    if n == 0:
        return []
    if n % 2 != 0:
        return []
    
    # Convert to assignment problem: match each vertex to exactly one other
    # Use the standard Hungarian algorithm on the cost matrix
    # We need to handle the symmetry constraint: if i matches j, j matches i
    
    # For even n, we can use a standard approach:
    # Find minimum weight perfect matching in complete graph
    # Use DP over subsets for small n
    if n <= 14:
        return _dp_matching(cost_matrix, odd_vertices)
    
    # For larger n, use greedy
    return _greedy_matching(cost_matrix, odd_vertices)

def _dp_matching(cost_matrix, odd_vertices):
    """DP over subsets for exact minimum matching (works for n <= ~14)."""
    n = len(odd_vertices)
    # State: bitmask of matched vertices
    # dp[mask] = min cost to match all vertices in mask
    INF = float('inf')
    full_mask = (1 << n) - 1
    
    dp = [INF] * (1 << n)
    dp[0] = 0
    parent = [None] * (1 << n)  # For reconstruction
    
    for mask in range(1 << n):
        if dp[mask] == INF:
            continue
        # Find first unmatched vertex in complement
        comp = full_mask ^ mask
        if comp == 0:
            continue
        # First set bit in comp
        i = (comp & (-comp)).bit_length() - 1
        if i >= n:
            continue
        # Try matching i with every other unmatched vertex j
        for j in range(i + 1, n):
            if comp & (1 << j):
                new_mask = mask | (1 << i) | (1 << j)
                cost = cost_matrix[i][j]
                if dp[mask] + cost < dp[new_mask]:
                    dp[new_mask] = dp[mask] + cost
                    parent[new_mask] = (i, j)
    
    if dp[full_mask] == INF:
        return []
    
    # Reconstruct
    pairs = []
    mask = full_mask
    while mask != 0:
        i, j = parent[mask]
        pairs.append((odd_vertices[i], odd_vertices[j]))
        mask ^= (1 << i) | (1 << j)
    
    return pairs

def _greedy_matching(cost_matrix, odd_vertices):
    """Greedy minimum-cost matching."""
    n = len(odd_vertices)
    matched = [False] * n
    pairs = []
    
    for i in range(n):
        if matched[i]:
            continue
        best_j = -1
        best_cost = float('inf')
        for j in range(n):
            if not matched[j] and j != i:
                if cost_matrix[i][j] < best_cost:
                    best_cost = cost_matrix[i][j]
                    best_j = j
        if best_j != -1:
            matched[i] = True
            matched[best_j] = True
            pairs.append((odd_vertices[i], odd_vertices[best_j]))
    
    return pairs

def _construct_paths_via_matching(edges, deadline):
    """
    Construct edge-disjoint path cover using matching on odd-degree vertices.
    For each connected component:
    1. Find all odd-degree vertices
    2. Compute minimum-cost perfect matching on them
    3. For each matched pair (u,v), find a path from u to v in remaining edges
    4. For each unmatched vertex (if any), start a path from it
    """
    adj = _build_graph(edges)
    remaining = set(map(tuple, edges))
    deg = defaultdict(int)
    for e in remaining:
        deg[e[0]] += 1
        deg[e[1]] += 1
    
    # Find connected components
    visited = set()
    components = []
    all_vertices = set()
    for e in edges:
        all_vertices.add(e[0])
        all_vertices.add(e[1])
    
    for v in all_vertices:
        if v not in visited:
            # BFS to find component
            comp = []
            queue = deque([v])
            visited.add(v)
            while queue:
                cur = queue.popleft()
                comp.append(cur)
                for u in adj[cur]:
                    if u not in visited:
                        visited.add(u)
                        queue.append(u)
            components.append(comp)
    
    paths = []
    
    for comp in components:
        comp_set = set(comp)
        # Get edges within this component
        comp_edges = [e for e in remaining if e[0] in comp_set and e[1] in comp_set]
        if not comp_edges:
            continue
        
        # Build subgraph adjacency
        sub_adj = defaultdict(list)
        for u, v in comp_edges:
            sub_adj[u].append(v)
            sub_adj[v].append(u)
        
        # Find odd-degree vertices in this component
        sub_deg = defaultdict(int)
        for u, v in comp_edges:
            sub_deg[u] += 1
            sub_deg[v] += 1
        
        odd_verts = [v for v in comp_set if sub_deg[v] % 2 == 1]
        
        if len(odd_verts) == 0:
            # Eulerian circuit exists - find one
            circuit = _find_eulerian_circuit(sub_adj, comp_edges)
            if circuit:
                paths.append(circuit)
        else:
            # Find matching on odd vertices
            pairs = _min_cost_matching_odd(sub_adj, odd_verts, set(comp_edges), deadline)
            
            # For each matched pair, find a path
            used_edges = set()
            sub_remaining = set(comp_edges)
            
            for u, v in pairs:
                if time.time() > deadline:
                    break
                # Find path from u to v using remaining edges
                path = _find_path_in_remaining(sub_adj, sub_remaining, u, v)
                if path:
                    paths.append(path)
                    for i in range(len(path) - 1):
                        e = (min(path[i], path[i+1]), max(path[i], path[i+1]))
                        sub_remaining.discard(e)
            
            # Handle any remaining edges (shouldn't be many)
            while sub_remaining:
                if time.time() > deadline:
                    break
                # Start a path from some vertex with remaining edges
                start_v = None
                for e in list(sub_remaining):
                    if sub_deg.get(e[0], 0) > 0:
                        start_v = e[0]
                        break
                    if sub_deg.get(e[1], 0) > 0:
                        start_v = e[1]
                        break
                if start_v is None:
                    # Just add remaining edges as single-edge paths
                    for e in sub_remaining:
                        paths.append(list(e))
                        sub_remaining.clear()
                    break
                
                path = _greedy_walk_sub(sub_adj, sub_remaining, start_v)
                if len(path) >= 2:
                    paths.append(path)
                else:
                    # Single vertex, add remaining edges individually
                    for e in sub_remaining:
                        paths.append(list(e))
                    sub_remaining.clear()
                    break
    
    return paths

def _find_eulerian_circuit(adj, edges):
    """Find Eulerian circuit using Hierholzer's algorithm."""
    if not edges:
        return []
    
    # Build edge list with indices
    edge_list = list(edges)
    edge_idx = {tuple(sorted(e)): i for i, e in enumerate(edge_list)}
    
    # Adjacency with edge indices
    adj_edges = defaultdict(list)
    for i, (u, v) in enumerate(edge_list):
        adj_edges[u].append((v, i))
        adj_edges[v].append((u, i))
    
    used = [False] * len(edge_list)
    circuit = []
    
    def dfs(v):
        while adj_edges[v]:
            u, idx = adj_edges[v].pop()
            if not used[idx]:
                used[idx] = True
                dfs(u)
                circuit.append(v)
    
    # Start from any vertex with edges
    start = edge_list[0][0]
    dfs(start)
    circuit.reverse()
    
    return circuit

def _find_path_in_remaining(adj, remaining, source, target):
    """Find a simple path from source to target using only edges in remaining."""
    if source == target:
        return [source]
    
    # BFS
    parent = {source: None}
    queue = deque([source])
    
    while queue:
        v = queue.popleft()
        if v == target:
            # Reconstruct path
            path = []
            cur = target
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        
        for u in adj[v]:
            if u not in parent:
                e = (min(v, u), max(v, u))
                if e in remaining:
                    parent[u] = v
                    queue.append(u)
    
    return None

def _greedy_walk_sub(adj, remaining, start_v):
    """Greedy walk in subgraph."""
    path = [start_v]
    visited = {start_v}
    
    while True:
        v = path[-1]
        valid_neighbors = []
        for u in adj[v]:
            if u not in visited:
                e = (min(v, u), max(v, u))
                if e in remaining:
                    valid_neighbors.append(u)
        
        if not valid_neighbors:
            break
        
        # Pick neighbor with lowest degree
        next_v = min(valid_neighbors, key=lambda u: sum(1 for x in adj[u] if (min(u,x), max(u,x)) in remaining))
        
        e = (min(v, next_v), max(v, next_v))
        remaining.discard(e)
        path.append(next_v)
        visited.add(next_v)
    
    return path

def _greedy_walk(adj, remaining, deg, start_v, rng=None, randomize=False, mode="low_deg"):
    """
    Greedy walk that builds a simple path by extending from start_v.
    """
    path = [start_v]
    visited = {start_v}
    
    while True:
        v = path[-1]
        valid_neighbors = []
        for u in adj[v]:
            if u not in visited:
                e = (min(v, u), max(v, u))
                if e in remaining:
                    valid_neighbors.append(u)
        
        if not valid_neighbors:
            break
        
        if randomize and rng is not None:
            if mode == "high_deg":
                weights = [float(deg.get(u, 0) + 1) for u in valid_neighbors]
            elif mode == "low_deg":
                weights = [1.0 / (deg.get(u, 0) + 1) for u in valid_neighbors]
            else:
                weights = [1.0 + deg.get(u, 0) * 0.5 for u in valid_neighbors]
            
            total_w = sum(weights)
            if total_w == 0:
                next_v = valid_neighbors[0]
            else:
                r = rng.random() * total_w
                cum = 0
                next_v = valid_neighbors[-1]
                for i, u in enumerate(valid_neighbors):
                    cum += weights[i]
                    if r <= cum:
                        next_v = u
                        break
        else:
            if mode == "high_deg":
                next_v = max(valid_neighbors, key=lambda u: deg.get(u, 0))
            elif mode == "low_deg":
                next_v = min(valid_neighbors, key=lambda u: deg.get(u, 0))
            else:
                next_v = max(valid_neighbors, key=lambda u: (deg.get(u, 0), -u))
        
        e = (min(v, next_v), max(v, next_v))
        remaining.discard(e)
        deg[v] -= 1
        deg[next_v] -= 1
        path.append(next_v)
        visited.add(next_v)
        
    return path

def _path_cover_single(edges, deadline, rng=None, randomize=False, mode="low_deg", start_strategy="leaf", shuffle_adj=False):
    """
    Single pass greedy path cover.
    """
    adj = _build_graph(edges)
    
    if shuffle_adj and rng is not None:
        for v in adj:
            rng.shuffle(adj[v])

    remaining = set(map(tuple, edges))
    paths = []
    
    deg = defaultdict(int)
    for e in remaining:
        deg[e[0]] += 1
        deg[e[1]] += 1
    
    while remaining and time.time() < deadline:
        start_candidates = set()
        for e in remaining:
            start_candidates.add(e[0])
            start_candidates.add(e[1])
        
        if start_strategy == "leaf":
            leaves = [v for v in start_candidates if deg[v] == 1]
            if leaves:
                if randomize and rng is not None:
                    start_v = rng.choice(leaves)
                else:
                    start_v = leaves[0]
            else:
                min_deg = min(deg[v] for v in start_candidates)
                candidates = [v for v in start_candidates if deg[v] == min_deg]
                if randomize and rng is not None:
                    start_v = rng.choice(candidates)
                else:
                    start_v = min(candidates)
        elif start_strategy == "max_deg":
            max_deg = max(deg[v] for v in start_candidates)
            candidates = [v for v in start_candidates if deg[v] == max_deg]
            if randomize and rng is not None:
                start_v = rng.choice(candidates)
            else:
                start_v = min(candidates)
        elif start_strategy == "min_deg":
            min_deg = min(deg[v] for v in start_candidates)
            candidates = [v for v in start_candidates if deg[v] == min_deg]
            if randomize and rng is not None:
                start_v = rng.choice(candidates)
            else:
                start_v = min(candidates)
        else:
            if randomize and rng is not None:
                start_v = rng.choice(list(start_candidates))
            else:
                start_v = min(start_candidates)
        
        path = _greedy_walk(adj, remaining, deg, start_v, rng, randomize, mode)
        paths.append(path)
    
    for e in remaining:
        paths.append(list(e))
        
    return paths

def _path_cover_with_restarts(edges, deadline):
    """
    Try matching-based approach first, then fall back to greedy if needed.
    """
    start = time.time()
    
    # 1. Try matching-based approach
    try:
        paths_match = _construct_paths_via_matching(edges, deadline)
        # Validate: check that all edges are covered and paths are valid
        if _validate_path_cover(paths_match, edges):
            return paths_match
    except Exception:
        pass
    
    # 2. Deterministic baseline: leaf-start, low-degree greedy
    paths = _path_cover_single(edges, deadline, randomize=False, mode="low_deg", start_strategy="leaf")
    best_paths = paths
    best_count = len(paths)
    
    # 3. Deterministic variant: min-degree start
    if time.time() < deadline - 5:
        paths2 = _path_cover_single(edges, deadline, randomize=False, mode="low_deg", start_strategy="min_deg")
        if len(paths2) < best_count:
            best_count = len(paths2)
            best_paths = paths2
    
    # 4. Deterministic variant: high-degree greedy
    if time.time() < deadline - 5:
        paths3 = _path_cover_single(edges, deadline, randomize=False, mode="high_deg", start_strategy="leaf")
        if len(paths3) < best_count:
            best_count = len(paths3)
            best_paths = paths3
    
    # 5. Randomized restarts
    rng = random.Random(42)
    modes = ["low_deg", "low_deg", "balanced", "high_deg"]
    start_strats = ["leaf", "min_deg", "random"]
    
    trial = 0
    while time.time() < deadline - 2:
        mode = modes[trial % len(modes)]
        strat = start_strats[trial % len(start_strats)]
        
        paths = _path_cover_single(edges, deadline - 2, rng, randomize=True, mode=mode, start_strategy=strat, shuffle_adj=True)
        
        if len(paths) < best_count:
            best_count = len(paths)
            best_paths = paths
            if best_count <= 8:
                break
        
        trial += 1
    
    return best_paths

def _validate_path_cover(paths, edges):
    """Validate that paths are edge-disjoint, cover all edges, and are simple."""
    if not paths:
        return len(edges) == 0
    
    edge_set = set(map(tuple, edges))
    used_edges = set()
    
    for path in paths:
        if len(path) < 2:
            # Single vertex path - only valid if no edges
            continue
        # Check simplicity
        if len(set(path)) != len(path):
            return False
        # Check edges exist and are not used
        for i in range(len(path) - 1):
            e = (min(path[i], path[i+1]), max(path[i], path[i+1]))
            if e not in edge_set:
                return False
            if e in used_edges:
                return False
            used_edges.add(e)
    
    # Check all edges are covered
    if used_edges != edge_set:
        return False
    
    return True

def construct():
    total_deadline = time.time() + 145
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    all_edges = [_instance(n, p, seed) for n, p, seed in specs]
    total_edges = sum(len(e) for e in all_edges)
    
    for i, (n, p, seed) in enumerate(specs):
        edges = all_edges[i]
        frac = len(edges) / total_edges
        time_alloc = max(15.0, frac * 140.0)
        remaining_time = total_deadline - time.time()
        time_alloc = min(time_alloc, remaining_time - 2.0)
        if time_alloc < 5.0:
            time_alloc = remaining_time - 1.0
        
        graph_deadline = time.time() + time_alloc
        paths = _path_cover_with_restarts(edges, graph_deadline)
        results.append(paths)
    
    return results

if __name__ == "__main__":
    result = construct()
    for i, paths in enumerate(result):
        print(f"Graph {i}: {len(paths)} paths")
    print(f"Total: {sum(len(p) for p in result)}")
