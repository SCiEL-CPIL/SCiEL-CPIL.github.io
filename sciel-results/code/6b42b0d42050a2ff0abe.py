import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges):
        if not paths:
            return len(edges) == 0
        edge_set = set(edges)
        used_edges = set()
        for path in paths:
            if len(path) < 2:
                continue
            if len(set(path)) != len(path):
                return False
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        return used_edges == edge_set

    def eulerian_trails(edges, seed):
        """
        Decompose graph into edge-disjoint trails using Hierholzer's algorithm.
        Returns list of trails (each a list of vertices).
        """
        rng = random.Random(seed)
        # Build adjacency with edge tracking
        adj = defaultdict(list)  # vertex -> list of (neighbor, edge_id)
        edge_map = {}  # edge_id -> (u, v)
        for eid, (u, v) in enumerate(edges):
            adj[u].append((v, eid))
            adj[v].append((u, eid))
            edge_map[eid] = (u, v)
        
        # Shuffle adjacency for randomness
        for v in adj:
            rng.shuffle(adj[v])
        
        used_edges = set()
        trails = []
        
        # Find start vertices: prefer odd-degree vertices
        degrees = defaultdict(int)
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
        odd_verts = [v for v in degrees if degrees[v] % 2 == 1]
        start_pool = odd_verts if odd_verts else list(range(max(degrees.keys()) + 1 if degrees else 1))
        rng.shuffle(start_pool)
        
        remaining_edges = set(range(len(edges)))
        
        while remaining_edges:
            # Find a start vertex that has unused edges
            start = None
            for v in start_pool:
                if any(eid in remaining_edges for nb, eid in adj[v]):
                    start = v
                    break
            if start is None:
                # Try any vertex
                for v in adj:
                    if any(eid in remaining_edges for nb, eid in adj[v]):
                        start = v
                        break
            if start is None:
                break
            
            # Hierholzer's algorithm
            trail = [start]
            stack = [(start, None)]  # (vertex, edge_id_used_to_get_here)
            while stack:
                v, _ = stack[-1]
                # Find next unused edge from v
                next_edge = None
                next_nb = None
                for nb, eid in adj[v]:
                    if eid in remaining_edges:
                        next_edge = eid
                        next_nb = nb
                        break
                if next_edge is None:
                    # Backtrack
                    stack.pop()
                    trail.append(v)
                else:
                    remaining_edges.discard(next_edge)
                    stack.append((next_nb, next_edge))
            
            trail = list(reversed(trail))
            if len(trail) >= 2:
                trails.append(trail)
        
        return trails

    def split_trail_to_paths(trail):
        """
        Split a trail (which may revisit vertices) into simple paths.
        A trail v0, v1, ..., vk is simple if all vi are distinct.
        We split at the first repeated vertex.
        """
        paths = []
        if len(trail) < 2:
            return paths
        
        seen = {}
        current_start = 0
        for i in range(1, len(trail)):
            if trail[i] in seen:
                # Split: [current_start..i-1] is a path, restart from i-1
                # Actually, the repeated vertex is trail[i] == trail[seen[trail[i]]]
                # The segment from current_start to i-1 forms a path
                # But we need to be careful: the edge (trail[i-1], trail[i]) connects
                # trail[i-1] to the repeated vertex.
                
                # Path 1: trail[current_start : i]  (includes the repeated vertex at position i)
                # Wait, let me think again.
                # Trail: v0, v1, ..., vj, ..., vi where vi == vj (j < i)
                # The sub-trail v0..vj is a simple path (if no other repeats)
                # Then vj..vi is another trail that starts and ends at... no, vj..vi starts at vj and goes to vi=vj
                # Actually vj, v(j+1), ..., vi=vj forms a cycle-like structure
                # 
                # Simpler approach: greedy split
                # When we encounter a repeat at position i (trail[i] == trail[j] for some j >= current_start),
                # we emit trail[current_start..j] as a path, and continue from j (the repeated vertex)
                # But we need to handle the edge (trail[i-1], trail[i])
                
                # Let's use a different approach:
                # Find the earliest position j >= current_start such that trail[j] == trail[i]
                # Emit trail[current_start..j] as a path
                # Set current_start = j, and continue (the vertex trail[i] = trail[j] is the new start)
                # But we've "used" the edge (trail[i-1], trail[i]) which connects to trail[j]
                # This edge needs to be assigned to some path.
                
                # Actually, let me reconsider. The trail uses each edge exactly once.
                # When we split, we need to partition the edges.
                
                # Better approach: find all positions where a vertex is revisited,
                # and split the trail into maximal simple subpaths.
                
                # For now, let's just emit trail[current_start:i] and start fresh from i-1
                # No wait, that doesn't work either because we'd lose edges.
                
                # Correct approach:
                # The trail is v0, v1, ..., vk
                # If vi appears again at position j > i, then:
                #   - v_i, v_{i+1}, ..., v_j is a closed walk (cycle in terms of vertices, but may have repeats)
                #   - We can extract the simple path from v_i to v_{j-1}... 
                
                # Let me just do a simple greedy: when we hit a repeat, cut there.
                # trail[current_start .. i] where trail[i] == trail[j] for j in [current_start, i-1]
                # The path is trail[current_start .. j] (simple, ends at the repeated vertex)
                # Then we continue the trail from position j (the repeated vertex), but we need
                # to account for the edges from j to i.
                
                # Actually, the cleanest way:
                # Identify all "break points" where a vertex is revisited.
                # Between consecutive break points, we have simple paths.
                
                pass
            seen[trail[i]] = i
        
        # Let me redo this with a cleaner algorithm
        paths = []
        n = len(trail)
        if n < 2:
            return paths
        
        # Find all positions where the current vertex was seen before
        # and determine split points
        first_seen = {}
        split_points = []  # positions where we should split (end of a path)
        
        for i in range(n):
            v = trail[i]
            if v in first_seen:
                # v was first seen at first_seen[v]
                # The segment from first_seen[v] to i contains v twice
                # We should split at first_seen[v] (end the path there)
                # But only if first_seen[v] is not already a split point
                j = first_seen[v]
                if j not in split_points:
                    split_points.append(j)
            else:
                first_seen[v] = i
        
        split_points.sort()
        
        # Now split the trail at these points
        # Path 1: trail[0 .. split_points[0]]
        # Path 2: trail[split_points[0] .. split_points[1]]
        # etc.
        
        if not split_points:
            paths.append(list(trail))
        else:
            prev = 0
            for sp in split_points:
                if sp > prev:
                    paths.append(list(trail[prev:sp+1]))
                prev = sp
            if prev < n - 1:
                paths.append(list(trail[prev:n]))
        
        # Filter out trivial paths (single vertex)
        paths = [p for p in paths if len(p) >= 2]
        
        return paths

    def greedy_random(edges, seed):
        rng = random.Random(seed)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        paths = []
        while remaining:
            edge = rng.choice(list(remaining))
            remaining.remove(edge)
            u, v = edge
            path = [u, v]
            changed = True
            while changed:
                changed = False
                for end in [0, 1]:
                    if end == 0:
                        current = path[0]
                        cands = [nb for nb in adj[current]
                                  if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                        if cands:
                            nb = rng.choice(cands)
                            if (current, nb) in remaining:
                                remaining.remove((current, nb))
                            else:
                                remaining.remove((nb, current))
                            path.insert(0, nb)
                            changed = True
                    else:
                        current = path[-1]
                        cands = [nb for nb in adj[current]
                                  if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                        if cands:
                            nb = rng.choice(cands)
                            if (current, nb) in remaining:
                                remaining.remove((current, nb))
                            else:
                                remaining.remove((nb, current))
                            path.append(nb)
                            changed = True
                if not changed:
                    break
            paths.append(path)
        return paths

    def greedy_target(edges, k, seed_val):
        """Try to cover all edges with exactly k simple paths."""
        if k <= 0 and edges:
            return None
        if len(edges) < k:
            return None
        rng = random.Random(seed_val)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        paths = [[] for _ in range(k)]
        edge_list = list(remaining)
        rng.shuffle(edge_list)
        start_edges = edge_list[:k]
        for i in range(k):
            u, v = start_edges[i]
            if (u, v) in remaining:
                remaining.remove((u, v))
            else:
                remaining.remove((v, u))
            paths[i] = [u, v]
        changed = True
        while changed:
            changed = False
            idxs = list(range(k))
            rng.shuffle(idxs)
            for i in idxs:
                if not paths[i]:
                    continue
                back = paths[i][-1]
                cands = [nb for nb in adj[back] if nb not in paths[i]
                         and ((back, nb) in remaining or (nb, back) in remaining)]
                if cands:
                    nb = rng.choice(cands)
                    if (back, nb) in remaining:
                        remaining.remove((back, nb))
                    else:
                        remaining.remove((nb, back))
                    paths[i].append(nb)
                    changed = True
                    continue
                front = paths[i][0]
                cands = [nb for nb in adj[front] if nb not in paths[i]
                         and ((front, nb) in remaining or (nb, front) in remaining)]
                if cands:
                    nb = rng.choice(cands)
                    if (front, nb) in remaining:
                        remaining.remove((front, nb))
                    else:
                        remaining.remove((nb, front))
                    paths[i].insert(0, nb)
                    changed = True
                    continue
        if remaining:
            return None
        return paths

    def max_matching_bipartite(left_nodes, right_nodes, adjacency):
        if not left_nodes or not right_nodes:
            return []
        left_index = {node: i for i, node in enumerate(left_nodes)}
        right_index = {node: i for i, node in enumerate(right_nodes)}
        n_left = len(left_nodes)
        n_right = len(right_nodes)
        adj = [[] for _ in range(n_left)]
        for u in left_nodes:
            ui = left_index[u]
            for v in adjacency.get(u, set()):
                if v in right_index:
                    adj[ui].append(right_index[v])
        match_left = [-1] * n_left
        match_right = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            queue = deque()
            for u in range(n_left):
                if match_left[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if match_right[v] == -1:
                        found = True
                    elif dist[match_right[v]] == float('inf'):
                        dist[match_right[v]] = dist[u] + 1
                        queue.append(match_right[v])
            return found

        def dfs(u):
            for v in adj[u]:
                if match_right[v] == -1 or (dist[match_right[v]] == dist[u] + 1 and dfs(match_right[v])):
                    match_left[u] = v
                    match_right[v] = u
                    return True
            dist[u] = float('inf')
            return False

        while bfs():
            for u in range(n_left):
                if match_left[u] == -1:
                    dfs(u)
        matching = []
        for u in range(n_left):
            if match_left[u] != -1:
                matching.append((left_nodes[u], right_nodes[match_left[u]]))
        return matching

    def merge_endpoints(paths, edges, adj, used_edges, max_iter=50):
        """Merge paths at their endpoints using bipartite matching."""
        if not paths:
            return []
        current = [list(p) for p in paths]
        current_used = set(used_edges)
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if len(current) < 2:
                break
            left_nodes = []
            right_nodes = []
            for i, path in enumerate(current):
                if len(path) < 2:
                    continue
                for pos in [0, 1]:
                    if i % 2 == 0:
                        left_nodes.append((i, pos))
                    else:
                        right_nodes.append((i, pos))
            if not left_nodes or not right_nodes:
                break
            adjacency = defaultdict(set)
            for (i1, pos1) in left_nodes:
                v1 = current[i1][0] if pos1 == 0 else current[i1][-1]
                for (i2, pos2) in right_nodes:
                    if i1 == i2:
                        continue
                    v2 = current[i2][0] if pos2 == 0 else current[i2][-1]
                    e = tuple(sorted((v1, v2)))
                    if e not in edges or e in current_used:
                        continue
                    if set(current[i1]) & set(current[i2]):
                        continue
                    adjacency[(i1, pos1)].add((i2, pos2))
            if not adjacency:
                break
            matching = max_matching_bipartite(left_nodes, right_nodes, adjacency)
            if not matching:
                break
            new_paths = {i: list(p) for i, p in enumerate(current)}
            removed = set()
            applied = 0
            for (lep, rep) in matching:
                i1, pos1 = lep
                i2, pos2 = rep
                if i1 in removed or i2 in removed:
                    continue
                p1 = new_paths[i1]
                p2 = new_paths[i2]
                v1 = p1[0] if pos1 == 0 else p1[-1]
                v2 = p2[0] if pos2 == 0 else p2[-1]
                e = tuple(sorted((v1, v2)))
                if e not in edges or e in current_used:
                    continue
                q1 = list(reversed(p1)) if pos1 == 1 else list(p1)
                q2 = list(reversed(p2)) if pos2 == 1 else list(p2)
                if q1[-1] != v1 or q2[0] != v2:
                    continue
                merged = q1 + q2[1:]
                if len(set(merged)) != len(merged):
                    continue
                ok = True
                for j in range(len(merged) - 1):
                    ej = tuple(sorted((merged[j], merged[j+1])))
                    if ej not in edges:
                        ok = False
                        break
                    if ej in current_used and ej != e:
                        ok = False
                        break
                if not ok:
                    continue
                new_paths[i1] = merged
                del new_paths[i2]
                removed.add(i2)
                current_used.add(e)
                applied += 1
            if applied == 0:
                break
            current = [new_paths[i] for i in sorted(new_paths.keys())]
        return current

    def local_search(paths, edges, adj, used_edges, max_iter=20):
        """
        Local search: try to reduce path count by:
        1. Merging two paths at endpoints if connected by an unused edge
        2. Splitting a path and reattaching pieces to other paths
        """
        if not paths or len(paths) <= 1:
            return [list(p) for p in paths]
        current = [list(p) for p in paths]
        current_used = set(used_edges)
        edge_set = set(edges)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            
            # Try merging any pair of paths at their endpoints
            for pi in range(len(current)):
                for pj in range(pi + 1, len(current)):
                    p1 = current[pi]
                    p2 = current[pj]
                    if len(p1) < 2 or len(p2) < 2:
                        continue
                    if set(p1) & set(p2):
                        continue
                    for pos1 in [0, 1]:
                        for pos2 in [0, 1]:
                            v1 = p1[0] if pos1 == 0 else p1[-1]
                            v2 = p2[0] if pos2 == 0 else p2[-1]
                            e = tuple(sorted((v1, v2)))
                            if e not in edge_set or e in current_used:
                                continue
                            q1 = list(reversed(p1)) if pos1 == 1 else list(p1)
                            q2 = list(reversed(p2)) if pos2 == 1 else list(p2)
                            merged = q1 + q2[1:]
                            if len(set(merged)) != len(merged):
                                continue
                            ok = True
                            for j in range(len(merged) - 1):
                                ej = tuple(sorted((merged[j], merged[j+1])))
                                if ej not in edge_set:
                                    ok = False
                                    break
                                if ej in current_used and ej != e:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            new_current = [list(p) for p in current]
                            new_current[pi] = merged
                            del new_current[pj]
                            current_used.add(e)
                            current = new_current
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return current

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        adj = defaultdict(set)
        degrees = defaultdict(int)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
        odd_vertices = [v for v in degrees if degrees[v] % 2 != 0]
        num_odd = len(odd_vertices)
        if num_odd == 0:
            lb = 1 if len(edges) == 1 else 2
        else:
            lb = num_odd // 2

        best_paths = None
        best_count = float('inf')
        max_attempts = 200

        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            seed_val = seed * 10000 + attempt

            if attempt % 2 == 0:
                # Eulerian trail decomposition + split
                trails = eulerian_trails(edges, seed_val)
                paths = []
                for trail in trails:
                    paths.extend(split_trail_to_paths(trail))
                # Local search to merge
                used_edges = set()
                for path in paths:
                    for j in range(len(path) - 1):
                        u, v = path[j], path[j+1]
                        used_edges.add(tuple(sorted((u, v))))
                if validate_paths(paths, edges):
                    merged = merge_endpoints(paths, edges, adj, used_edges)
                    merged = local_search(merged, edges, adj, used_edges)
                    if validate_paths(merged, edges):
                        if len(merged) < best_count:
                            best_count = len(merged)
                            best_paths = [list(p) for p in merged]
                            if best_count <= lb:
                                break
            else:
                # Greedy approaches
                init = greedy_target(edges, lb, seed_val)
                if init is None:
                    init = greedy_target(edges, lb + 1, seed_val)
                if init is None:
                    init = greedy_random(edges, seed_val)
                
                if not init:
                    continue
                
                used_edges = set()
                for path in init:
                    for j in range(len(path) - 1):
                        u, v = path[j], path[j+1]
                        used_edges.add(tuple(sorted((u, v))))
                
                merged = merge_endpoints(init, edges, adj, used_edges)
                merged = local_search(merged, edges, adj, used_edges)
                
                if validate_paths(merged, edges):
                    if len(merged) < best_count:
                        best_count = len(merged)
                        best_paths = [list(p) for p in merged]
                        if best_count <= lb:
                            break

        if best_paths is None:
            for i in range(50):
                if time.time() > deadline - 5:
                    break
                paths = greedy_random(edges, seed * 10000 + 50000 + i)
                if validate_paths(paths, edges):
                    if len(paths) < best_count:
                        best_count = len(paths)
                        best_paths = [list(p) for p in paths]

        if best_paths is None:
            best_paths = [[u, v] for u, v in edges]

        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    for spec in specs:
        n, p, seed = spec
        if time.time() > deadline - 10:
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        paths = solve_instance(n, p, seed)
        results.append(paths)
    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])
    return results

if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
