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


def _build_graph(edges, n):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _max_2_matching(edges, n):
    """
    Find a maximum 2-matching using reduction to bipartite matching.
    Each vertex v is split into v_out (left) and v_in (right).
    Each undirected edge (u,v) creates bipartite edges:
      u_out -> v_in  and  v_out -> u_in
    A matching of size M in the bipartite graph corresponds to a 2-matching
    with M edges in the original graph.
    """
    # Build bipartite graph: left nodes are (v, 'out'), right nodes are (v, 'in')
    # We use node indices: left v -> v, right v -> v + n
    # Adjacency from left to right
    left_adj = defaultdict(list)  # left node -> list of right nodes
    for u, v in edges:
        left_adj[u].append(v + n)   # u_out -> v_in
        left_adj[v].append(u + n)   # v_out -> u_in

    # Hopcroft-Karp for maximum bipartite matching
    # Left nodes: 0..n-1, Right nodes: n..2n-1
    pair_u = [-1] * n      # match for left node u
    pair_v = [-1] * n      # match for right node v (stored as v - n)
    dist = [0] * n

    from collections import deque

    def bfs():
        q = deque()
        found = False
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        while q:
            u = q.popleft()
            for rv in left_adj[u]:
                v = rv - n
                if pair_v[v] == -1:
                    found = True
                elif dist[pair_v[v]] == -1:
                    dist[pair_v[v]] = dist[u] + 1
                    q.append(pair_v[v])
        return found

    def dfs(u):
        for rv in left_adj[u]:
            v = rv - n
            if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False

    matching_size = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1:
                if dfs(u):
                    matching_size += 1

    # Extract the 2-matching edges
    # pair_u[u] = v means u_out matched to v_in, which corresponds to edge (u,v)
    matched_edges = set()
    for u in range(n):
        if pair_u[u] != -1:
            v = pair_u[u]
            e = (min(u, v), max(u, v))
            matched_edges.add(e)

    return matched_edges, matching_size


def _extract_paths_from_2matching(matched_edges, n):
    """
    From a 2-matching (subgraph with max degree 2), extract all paths and cycles.
    Returns: list of paths (each path is a list of vertices), and list of cycles.
    """
    adj = defaultdict(list)
    for u, v in matched_edges:
        adj[u].append(v)
        adj[v].append(u)

    paths = []
    cycles = []
    visited_edges = set()

    for start in range(n):
        if not adj[start]:
            continue
        # Find an unvisited edge incident to start
        start_edge = None
        for nb in adj[start]:
            e = (min(start, nb), max(start, nb))
            if e not in visited_edges:
                start_edge = (start, nb)
                break
        if start_edge is None:
            continue

        # Walk the path/cycle starting from start along start_edge
        path = [start]
        visited_edges.add((min(start, start_edge[1]), max(start, start_edge[1])))
        prev = start
        curr = start_edge[1]
        path.append(curr)

        while True:
            # Find next unvisited edge from curr
            next_node = None
            for nb in adj[curr]:
                if nb == prev:
                    continue
                e = (min(curr, nb), max(curr, nb))
                if e not in visited_edges:
                    next_node = nb
                    break
            if next_node is None:
                break
            visited_edges.add((min(curr, next_node), max(curr, next_node)))
            prev, curr = curr, next_node
            path.append(curr)

        # Check if it's a cycle: first and last are same? No, we track edges.
        # It's a cycle if the last node connects back to the first via an unvisited edge
        # Actually, if we've used all edges in this component, check if it's a cycle
        first = path[0]
        last = path[-1]
        if first == last:
            cycles.append(path)
        else:
            # Check if there's an edge from last to first not yet visited
            e = (min(last, first), max(last, first))
            if e in matched_edges and e not in visited_edges:
                # It's a cycle
                cycles.append(path)
                visited_edges.add(e)
            else:
                paths.append(path)

    return paths, cycles


def _path_cover_via_2matching(edges, n, deadline):
    """
    Use maximum 2-matching to get a near-optimal path cover.
    """
    matched_edges, msize = _max_2_matching(edges, n)
    all_edge_set = set(map(tuple, edges))
    remaining_edges = all_edge_set - matched_edges

    # Extract paths and cycles from the 2-matching
    paths, cycles = _extract_paths_from_2matching(matched_edges, n)

    # Each cycle needs to be "broken" into a path if we have remaining edges,
    # or we can keep it as a cycle... but the task requires paths (simple paths).
    # A cycle is not a simple path (it's closed). We need to break it.
    # Actually, re-reading: "Paths must be simple" - a simple path has distinct vertices.
    # A cycle like [1,2,3,1] is NOT a simple path because vertex 1 appears twice.
    # So we must break cycles into paths by removing one edge.
    
    # But wait - if we break a cycle, we create one path from it.
    # The broken edge becomes a remaining edge.
    
    broken_cycles_paths = []
    for cycle in cycles:
        # Break the cycle: remove the last edge (last -> first)
        # cycle = [v0, v1, ..., vk] where edge (vk, v0) closes it
        # The path is [v0, v1, ..., vk]
        path = cycle[:-1] if len(cycle) > 1 else cycle
        if len(path) >= 2:
            broken_cycles_paths.append(path)
        # The broken edge (last, first) goes to remaining
        if len(cycle) >= 2:
            e = (min(cycle[-1], cycle[0]), max(cycle[-1], cycle[0]))
            remaining_edges.add(e)

    paths.extend(broken_cycles_paths)

    # Now cover remaining_edges. Each remaining edge is a single-edge path,
    # but we might be able to merge them with existing path endpoints.
    # Simple approach: add each remaining edge as a separate path,
    # then try to merge.
    
    for u, v in remaining_edges:
        paths.append([u, v])

    # Now try to merge paths: if endpoint of path A is adjacent (via a used edge? No,
    # all edges are used) to endpoint of path B... 
    # Actually, all edges are already assigned. Merging would require reassigning edges.
    # Let's try a different improvement: look for two paths where an endpoint of one
    # is connected to an endpoint of another by an edge that's in one of the paths? No.
    
    # Better improvement: try to find augmenting structures.
    # For now, return what we have.
    
    return paths


def _greedy_walk(adj, remaining, deg, start_v, rng=None, randomize=False, mode="low_deg"):
    """Greedy walk that builds a simple path."""
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


def _path_cover_single(edges, n, deadline, rng=None, randomize=False, mode="low_deg", start_strategy="leaf", shuffle_adj=False):
    """Single pass greedy path cover."""
    adj = _build_graph(edges, n)
    
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
        
        if not start_candidates:
            break
            
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


def _try_merge_paths(paths, edges, n, deadline):
    """
    Try to merge paths by finding two paths P1=[a,...,b] and P2=[c,...,d]
    where b==c or b==d or a==c or a==d (sharing an endpoint) and the edge
    connecting them can be used. Actually, for edge-disjoint paths, merging
    requires that an endpoint of P1 is adjacent to an endpoint of P2 via
    an edge that belongs to one of the paths... this is complex.
    
    Simpler: if two paths share a vertex at their endpoints, we can concatenate
    them (removing the duplicate vertex). But this changes edge usage.
    
    Actually, the correct merge: if path1 ends at vertex x and path2 starts at vertex x,
    and the last edge of path1 and first edge of path2 are different edges,
    we can merge into one path: path1[:-1] + path2 (if x appears only once in the merge).
    But we need to ensure simplicity.
    
    For now, skip complex merging.
    """
    return paths


def _path_cover_with_restarts(edges, n, deadline):
    """Try multiple strategies and return the best path cover."""
    best_paths = None
    best_count = len(edges)
    
    # 1. 2-matching based approach
    if time.time() < deadline - 10:
        paths = _path_cover_via_2matching(edges, n, deadline - 10)
        if paths:
            best_paths = paths
            best_count = len(paths)
    
    # 2. Deterministic baseline: leaf-start, low-degree greedy
    if time.time() < deadline - 5:
        paths2 = _path_cover_single(edges, n, deadline - 5, randomize=False, mode="low_deg", start_strategy="leaf")
        if len(paths2) < best_count:
            best_count = len(paths2)
            best_paths = paths2
    
    # 3. Deterministic variant: min-degree start
    if time.time() < deadline - 5:
        paths3 = _path_cover_single(edges, n, deadline - 5, randomize=False, mode="low_deg", start_strategy="min_deg")
        if len(paths3) < best_count:
            best_count = len(paths3)
            best_paths = paths3
    
    # 4. Deterministic variant: high-degree greedy
    if time.time() < deadline - 5:
        paths4 = _path_cover_single(edges, n, deadline - 5, randomize=False, mode="high_deg", start_strategy="leaf")
        if len(paths4) < best_count:
            best_count = len(paths4)
            best_paths = paths4
    
    # 5. Randomized restarts
    rng = random.Random(42)
    modes = ["low_deg", "low_deg", "balanced", "high_deg"]
    start_strats = ["leaf", "min_deg", "random"]
    
    trial = 0
    while time.time() < deadline - 2:
        mode = modes[trial % len(modes)]
        strat = start_strats[trial % len(start_strats)]
        
        paths = _path_cover_single(edges, n, deadline - 2, rng, randomize=True, mode=mode, start_strategy=strat, shuffle_adj=True)
        
        if len(paths) < best_count:
            best_count = len(paths)
            best_paths = paths
            if best_count <= 8:
                break
        
        trial += 1
    
    if best_paths is None:
        best_paths = _path_cover_single(edges, n, deadline, randomize=False, mode="low_deg", start_strategy="leaf")
    
    return best_paths


def _validate_and_fix(paths, edges, n):
    """
    Validate that paths are simple, use graph edges, are edge-disjoint, and cover all edges.
    If invalid, fix by reconstructing.
    """
    edge_set = set(map(tuple, edges))
    used_edges = set()
    valid_paths = []
    
    for path in paths:
        if len(path) < 2:
            continue
        # Check simplicity
        if len(set(path)) != len(path):
            continue
        # Check edges
        valid = True
        path_edges = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            e = (min(u, v), max(u, v))
            if e not in edge_set:
                valid = False
                break
            if e in used_edges:
                valid = False
                break
            path_edges.append(e)
        if not valid:
            continue
        for e in path_edges:
            used_edges.add(e)
        valid_paths.append(path)
    
    # Check coverage
    if used_edges != edge_set:
        # Add missing edges as single-edge paths
        for e in edge_set - used_edges:
            valid_paths.append([e[0], e[1]])
    
    return valid_paths


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
        paths = _path_cover_with_restarts(edges, n, graph_deadline)
        paths = _validate_and_fix(paths, edges, n)
        results.append(paths)
    
    return results

if __name__ == "__main__":
    result = construct()
    for i, paths in enumerate(result):
        print(f"Graph {i}: {len(paths)} paths")
    print(f"Total: {sum(len(p) for p in result)}")
