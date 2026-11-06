"""Rewritten program combining flow-based path decomposition with multi-strategy search."""
# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

import time
import random
import sys
from collections import deque, defaultdict

def _build_adj():
    a = [set() for _ in range(P['n'])]
    for u, v in P['edges']:
        a[u].add(v)
        a[v].add(u)
    return a

def _find_largest_component(adj):
    visited = set()
    components = []
    for start in range(P['n']):
        if start not in visited:
            comp = []
            stack = [start]
            visited.add(start)
            while stack:
                v = stack.pop()
                comp.append(v)
                for u in adj[v]:
                    if u not in visited:
                        visited.add(u)
                        stack.append(u)
            components.append(comp)
    return max(components, key=len) if components else []

def _max_flow_vertex_disjoint_paths(adj, comp_set, deadline):
    """
    Find maximum set of vertex-disjoint paths using max-flow on a vertex-split graph.
    Each vertex v is split into v_in -> v_out with capacity 1.
    Each edge (u,v) becomes u_out -> v_in and v_out -> u_in with capacity 1.
    We add a super source connected to all v_in and super sink from all v_out.
    The max flow gives the number of vertex-disjoint paths (each unit of flow is one path).
    We then extract the actual paths from the flow.
    Returns: list of paths (each path is a list of vertices), and set of used vertices.
    """
    n = P['n']
    vertices = sorted(comp_set)
    vset = set(vertices)
    
    # Build flow network
    # Node indices: v_in = 2*v, v_out = 2*v+1
    # Super source = 2*n, super sink = 2*n+1
    SRC = 2 * n
    SNK = 2 * n + 1
    total_nodes = 2 * n + 2
    
    # Adjacency for flow network (residual)
    flow_adj = defaultdict(list)
    flow_cap = defaultdict(int)
    
    def add_flow_edge(u, v, cap):
        flow_adj[u].append(v)
        flow_adj[v].append(u)
        flow_cap[(u, v)] = flow_cap.get((u, v), 0) + cap
        flow_cap[(v, u)] = flow_cap.get((v, u), 0) + 0  # reverse edge
    
    # Vertex capacity edges: v_in -> v_out with cap 1
    for v in vertices:
        vin = 2 * v
        vout = 2 * v + 1
        add_flow_edge(vin, vout, 1)
    
    # Graph edges: u_out -> v_in with cap 1
    for u in vertices:
        uout = 2 * u + 1
        for v in adj[u]:
            if v in vset and v != u:
                vin = 2 * v
                add_flow_edge(uout, vin, 1)
    
    # Super source to all v_in
    for v in vertices:
        vin = 2 * v
        add_flow_edge(SRC, vin, 1)
    
    # All v_out to super sink
    for v in vertices:
        vout = 2 * v + 1
        add_flow_edge(vout, SNK, 1)
    
    # Edmonds-Karp (BFS-based max flow)
    def bfs_max_flow():
        total_flow = 0
        while True:
            if time.time() > deadline:
                break
            # BFS to find augmenting path
            parent = [-1] * total_nodes
            parent[SRC] = SRC
            queue = deque([SRC])
            found = False
            while queue:
                if time.time() > deadline:
                    break
                u = queue.popleft()
                for v in flow_adj[u]:
                    if parent[v] == -1 and flow_cap.get((u, v), 0) > 0:
                        parent[v] = u
                        if v == SNK:
                            found = True
                            break
                        queue.append(v)
                if found:
                    break
            if not found:
                break
            # Find bottleneck
            min_cap = float('inf')
            v = SNK
            while v != SRC:
                u = parent[v]
                min_cap = min(min_cap, flow_cap.get((u, v), 0))
                v = u
            # Augment
            v = SNK
            while v != SRC:
                u = parent[v]
                flow_cap[(u, v)] -= min_cap
                flow_cap[(v, u)] = flow_cap.get((v, u), 0) + min_cap
                v = u
            total_flow += min_cap
        return total_flow
    
    bfs_max_flow()
    
    # Extract paths from the flow
    # A vertex v is "used" if flow goes through v_in -> v_out
    # For each used vertex, find which v_out -> v_in edges carry flow
    paths = []
    used_vertices = set()
    # Build successor map from flow
    succ = {}  # v_out -> v_in (the next vertex in path)
    pred = {}  # v_in -> v_out (the previous vertex in path)
    
    for u in vertices:
        uout = 2 * u + 1
        for v in adj[u]:
            if v in vset and v != u:
                vin = 2 * v
                # Check if there's flow on u_out -> v_in
                if flow_cap.get((vin, uout), 0) > 0:  # reverse cap > 0 means forward flow exists
                    # Actually, let's check: original cap was 1, if flow_cap[(uout,vin)] < 1, flow went through
                    # Better: track flow separately
                    pass
    
    # Simpler: re-derive paths by tracing flow
    # For each vertex, check if flow passed through v_in -> v_out
    vertex_flow = {}
    for v in vertices:
        vin = 2 * v
        vout = 2 * v + 1
        # Flow on vin->vout is 1 if flow_cap[(vout,vin)] > 0 (reverse edge has capacity)
        if flow_cap.get((vout, vin), 0) > 0:
            vertex_flow[v] = True
            used_vertices.add(v)
    
    # Build path successor: for each v with flow, find w such that flow went v_out -> w_in
    path_succ = {}
    for v in used_vertices:
        vout = 2 * v + 1
        for w in adj[v]:
            if w in used_vertices and w != v:
                win = 2 * w
                # Check if flow went from v_out to w_in
                if flow_cap.get((win, vout), 0) > 0:
                    path_succ[v] = w
                    break
    
    # Find path starts (vertices with no predecessor in path)
    path_starts = []
    for v in used_vertices:
        has_pred = False
        for u in adj[v]:
            if u in used_vertices and u != v:
                if path_succ.get(u) == v:
                    has_pred = True
                    break
        if not has_pred:
            path_starts.append(v)
    
    # Trace paths
    for start in path_starts:
        path = [start]
        cur = start
        while cur in path_succ:
            cur = path_succ[cur]
            path.append(cur)
        if len(path) > 1:
            paths.append(path)
    
    # Also check for single-vertex "paths" that might be isolated
    # Sort paths by length descending
    paths.sort(key=len, reverse=True)
    
    return paths, used_vertices

def _stitch_paths(adj, paths, unused_vertices, deadline):
    """
    Try to stitch vertex-disjoint paths into a single longer path by
    inserting unused vertices as connectors between path endpoints.
    """
    if not paths:
        return []
    
    # Start with the longest path
    best_path = list(paths[0])
    best_set = set(best_path)
    
    remaining_paths = paths[1:]
    unused = list(unused_vertices - best_set)
    
    # Try to extend best_path by connecting to other paths via unused vertices
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try extending from both ends of best_path
        for end in ['front', 'back']:
            if end == 'front':
                endpoint = best_path[0]
            else:
                endpoint = best_path[-1]
            
            # Find unused vertices adjacent to endpoint
            cands = [v for v in unused if v in adj[endpoint]]
            if not cands:
                continue
            
            # For each candidate, try to connect to a remaining path
            for v in sorted(cands, key=lambda x: len(adj[x])):
                # Check if v can connect to any remaining path endpoint
                for pi, p in enumerate(remaining_paths):
                    # Check if v is adjacent to either end of path p
                    connected = False
                    new_segment = []
                    if v in adj[p[0]]:
                        new_segment = [v] + p
                        connected = True
                    elif v in adj[p[-1]]:
                        new_segment = p + [v]
                        connected = True
                    
                    if not connected:
                        continue
                    
                    # Check that all vertices in new_segment are unused
                    seg_set = set(new_segment)
                    if seg_set & best_set:
                        continue
                    
                    # Build new path
                    if end == 'front':
                        new_path = new_segment + best_path
                    else:
                        new_path = best_path + new_segment
                    
                    new_set = set(new_path)
                    
                    # Verify it's a valid path
                    valid = True
                    for i in range(len(new_path) - 1):
                        if new_path[i+1] not in adj[new_path[i]]:
                            valid = False
                            break
                    
                    if valid and len(new_path) > len(best_path):
                        best_path = new_path
                        best_set = new_set
                        remaining_paths.pop(pi)
                        unused = [v for v in unused if v not in best_set]
                        improved = True
                        break
                if improved:
                    break
        
        # Also try inserting single unused vertices between consecutive path vertices
        for i in range(len(best_path) - 1):
            a, b = best_path[i], best_path[i+1]
            common = (adj[a] & adj[b]) - best_set
            for w in sorted(common, key=lambda x: len(adj[x])):
                new_path = best_path[:i+1] + [w] + best_path[i+1:]
                new_set = set(new_path)
                valid = True
                for j in range(len(new_path) - 1):
                    if new_path[j+1] not in adj[new_path[j]]:
                        valid = False
                        break
                if valid:
                    best_path = new_path
                    best_set = new_set
                    unused = [v for v in unused if v not in best_set]
                    improved = True
                    break
            if improved:
                break
    
    return best_path

def _greedy_path(adj, start, strategy, rng=None, deadline=None):
    path = [start]
    seen = {start}
    while True:
        if deadline and time.time() > deadline:
            break
        cands = adj[path[-1]] - seen
        if not cands:
            break
        if strategy == 'min_deg':
            v = min(cands, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'max_deg':
            v = max(cands, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'random':
            v = rng.choice(sorted(cands))
        elif strategy == 'slack_aware':
            v = min(cands, key=lambda w: (len(adj[w] - seen), len(adj[w]), w))
        else:
            v = min(cands)
        path.append(v)
        seen.add(v)
    return path

def _extend_from_both_ends(adj, path, deadline):
    if not path:
        return path
    result = list(path)
    seen = set(result)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cands = adj[result[-1]] - seen
        if cands:
            v = min(cands, key=lambda w: len(adj[w] - seen))
            result.append(v)
            seen.add(v)
            improved = True
        cands = adj[result[0]] - seen
        if cands:
            v = min(cands, key=lambda w: len(adj[w] - seen))
            result.insert(0, v)
            seen.add(v)
            improved = True
    return result

def _backtrack_search(adj, start, deadline, best_path, best_len):
    """DFS with backtracking, Warnsdorff-like heuristic."""
    path = [start]
    seen = {start}
    sys.setrecursionlimit(10000)
    
    def dfs():
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        cur = path[-1]
        cands = [v for v in adj[cur] if v not in seen]
        if not cands:
            return
        
        # Warnsdorff: prefer vertices with fewer unvisited neighbors
        cands.sort(key=lambda v: (len(adj[v] - seen), v))
        
        for v in cands:
            if time.time() > deadline:
                return
            path.append(v)
            seen.add(v)
            dfs()
            path.pop()
            seen.discard(v)
    
    dfs()
    return best_path, best_len

def _randomized_backtrack(adj, start, deadline, best_path, best_len, rng):
    """Randomized DFS with backtracking."""
    path = [start]
    seen = {start}
    
    def dfs():
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        cur = path[-1]
        cands = [v for v in adj[cur] if v not in seen]
        if not cands:
            return
        
        # Mix of Warnsdorff and random
        cands.sort(key=lambda v: (len(adj[v] - seen), rng.random()))
        
        # Limit branching
        max_branch = 4
        if len(cands) > max_branch:
            chosen = cands[:max(1, max_branch // 2)]
            remaining = cands[max(1, max_branch // 2):]
            if remaining:
                chosen.extend(rng.sample(remaining, min(max_branch - len(chosen), len(remaining))))
            cands = chosen
        
        for v in cands:
            if time.time() > deadline:
                return
            path.append(v)
            seen.add(v)
            dfs()
            path.pop()
            seen.discard(v)
    
    dfs()
    return best_path, best_len

def _local_search_improve(adj, path, deadline, rng=None):
    """Try to improve a path by local insertions and extensions."""
    if len(path) < 2:
        return path
    best = list(path)
    best_len = len(best)
    seen = set(best)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try inserting vertices between consecutive path vertices
        for i in range(len(best) - 1):
            if time.time() > deadline:
                break
            a, b = best[i], best[i+1]
            common = (adj[a] & adj[b]) - seen
            for w in sorted(common, key=lambda x: len(adj[x])):
                new_path = best[:i+1] + [w] + best[i+1:]
                new_seen = set(new_path)
                valid = True
                for j in range(len(new_path) - 1):
                    if new_path[j+1] not in adj[new_path[j]]:
                        valid = False
                        break
                if valid and len(new_path) > best_len:
                    best = new_path
                    best_len = len(best)
                    seen = new_seen
                    improved = True
                    break
        # Extend ends
        cands = adj[best[-1]] - seen
        if cands:
            v = min(cands, key=lambda w: len(adj[w] - seen))
            best.append(v)
            seen.add(v)
            best_len = len(best)
            improved = True
        cands = adj[best[0]] - seen
        if cands:
            v = min(cands, key=lambda w: len(adj[w] - seen))
            best.insert(0, v)
            seen.add(v)
            best_len = len(best)
            improved = True
    return best

def construct():
    deadline = time.time() + 85
    adj = _build_adj()
    best = []
    
    largest_comp = _find_largest_component(adj)
    if not largest_comp:
        return [0]
    comp_set = set(largest_comp)
    
    # Step 1: Flow-based decomposition to find vertex-disjoint paths
    flow_paths, flow_used = _max_flow_vertex_disjoint_paths(adj, comp_set, deadline)
    
    # Step 2: Stitch flow paths together
    unused = comp_set - flow_used
    stitched = _stitch_paths(adj, flow_paths, unused, deadline)
    if len(stitched) > len(best):
        best = stitched
    
    # Step 3: Greedy from every start
    for start in largest_comp:
        if time.time() > deadline:
            break
        path = _greedy_path(adj, start, 'min_deg')
        if len(path) > len(best):
            best = path
    
    # Step 4: Randomized backtracking from various starts
    rng = random.Random(42)
    best_path = best[:]
    best_len = len(best)
    
    # Try backtracking from low-degree vertices (likely endpoints)
    degs = [(len(adj[i]), i) for i in largest_comp]
    degs.sort()
    low_deg_starts = [v for _, v in degs[:20]]
    
    for start in low_deg_starts:
        if time.time() > deadline:
            break
        best_path, best_len = _backtrack_search(adj, start, deadline, best_path, best_len)
        if len(best_path) > len(best):
            best = best_path
    
    # Step 5: Randomized backtracking with multiple restarts
    for _ in range(30):
        if time.time() > deadline:
            break
        start = rng.choice(largest_comp)
        best_path, best_len = _randomized_backtrack(adj, start, deadline, best_path, best_len, rng)
        if len(best_path) > len(best):
            best = best_path
    
    # Step 6: Local search improvement
    if best and time.time() < deadline:
        improved = _local_search_improve(adj, best[:], deadline, rng)
        if len(improved) > len(best):
            best = improved
    
    # Step 7: More greedy strategies
    strategies = ['min_deg', 'max_deg', 'random', 'slack_aware']
    for strat in strategies:
        for start in largest_comp:
            if time.time() > deadline:
                break
            path = _greedy_path(adj, start, strat, rng)
            if len(path) > len(best):
                best = path
            if len(path) >= 3:
                extended = _extend_from_both_ends(adj, path[:], deadline)
                if len(extended) > len(best):
                    best = extended
    
    # Step 8: Final local search polish
    if best and time.time() < deadline:
        improved = _local_search_improve(adj, best[:], deadline, rng)
        if len(improved) > len(best):
            best = improved
    
    # Step 9: Final extension
    if best and time.time() < deadline:
        extended = _extend_from_both_ends(adj, best[:], deadline)
        if len(extended) > len(best):
            best = extended
    
    if not best:
        best = [largest_comp[0]]
    
    return best
# EVOLVE-BLOCK-END
