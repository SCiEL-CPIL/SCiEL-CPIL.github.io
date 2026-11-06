"""Path decomposition of graphs using matching-based approach with greedy fallback."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    best_results = []
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback: each edge as its own path
            edges = _instance(*spec)
            best_results.append([list(e) for e in edges])
            continue
        edges = _instance(*spec)
        result = _decompose(edges, deadline)
        if result is not None:
            best_results.append(result)
        else:
            # Fallback: each edge as its own path
            best_results.append([list(e) for e in edges])
    
    return best_results


def _instance(n, p, seed):
    """Generate the instance graph exactly as in the verifier."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, deadline):
    """Decompose edges into minimum number of edge-disjoint simple paths."""
    if not edges:
        return []
    
    # Try matching-based decomposition
    result = _matching_decompose(edges, deadline)
    if result is not None:
        return result
    
    # Fallback: greedy approach
    best_paths = None
    best_count = len(edges)
    
    num_restarts = max(20, int((deadline - time.time()) * 10))
    num_restarts = min(num_restarts, 100)
    
    for attempt in range(num_restarts):
        if time.time() > deadline - 3:
            break
        
        rng = random.Random(attempt * 7919 + 13)
        paths = _greedy_decompose(edges, rng, deadline)
        
        if paths is not None:
            paths = _merge_paths(paths, edges)
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                lower_bound = _lower_bound(edges)
                if count <= lower_bound:
                    break
    
    if best_paths is None:
        return [list(e) for e in edges]
    
    return best_paths


def _lower_bound(edges):
    """Compute theoretical lower bound on number of paths."""
    if not edges:
        return 0
    degree = defaultdict(int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    odd_count = sum(1 for d in degree.values() if d % 2 == 1)
    return max(1, (odd_count + 1) // 2)


def _matching_decompose(edges, deadline):
    """
    Matching-based path decomposition.
    
    Key idea: Each edge (u,v) can be represented as two "half-edge" tokens: u→v and v→u.
    We want to pair up half-edges such that:
    - If we pair u→v with v→w, we extend a path through vertex v
    - Unpaired half-edges are path endpoints
    
    The number of paths = |E| - (number of pairs) = |E| - |M|
    where M is a matching in a graph whose vertices are half-edges.
    
    We build a graph where:
    - Vertices: one per half-edge (2|E| total)
    - Edge between half-edge (u→v) and half-edge (v→w) if v is common and u≠w
    - We find maximum matching to maximize connections, minimizing paths
    
    But we must ensure simplicity: no vertex appears twice in any path.
    """
    n_edges = len(edges)
    if n_edges <= 1:
        return [[list(e)] for e in edges] if n_edges == 1 else []
    
    # Create half-edge identifiers
    # half_edge[i] = (u, v) means the directed edge from u to v
    half_edges = []
    for i, (u, v) in enumerate(edges):
        half_edges.append((u, v, i))  # u→v, edge index i
        half_edges.append((v, u, i))  # v→u, edge index i
    
    num_half = len(half_edges)
    
    # Build adjacency for matching graph
    # Two half-edges can be matched if they share a middle vertex
    # half_edge a = (a_from, a_to, a_idx) can match half_edge b = (b_from, b_to, b_idx)
    # if a_to == b_from AND a_idx != b_idx (different original edges)
    # This creates a path segment: a_from → a_to(=b_from) → b_to
    
    match_adj = defaultdict(list)
    for i in range(num_half):
        a_from, a_to, a_idx = half_edges[i]
        for j in range(i + 1, num_half):
            b_from, b_to, b_idx = half_edges[j]
            if a_idx == b_idx:
                continue  # Same original edge, can't match
            if a_to == b_from:
                match_adj[i].append(j)
                match_adj[j].append(i)
    
    # Find maximum matching using greedy + local improvement
    # For small graphs, try exact; for larger, use heuristic
    matching = _find_maximum_matching(match_adj, num_half, deadline)
    
    if matching is None:
        return None
    
    # Reconstruct paths from matching
    # matched[i] = j means half-edge i is connected to half-edge j
    # This means: if half_edge[i] = (u, v, idx_i) and half_edge[j] = (v, w, idx_j)
    # then we have a connection u-v-w
    
    # Build the path structure
    # Each half-edge has at most one match
    # Start from unpaired half-edges (path starts) and follow matches
    
    # For each half-edge i, find its match
    match_of = [-1] * num_half
    for i, j in matching:
        match_of[i] = j
        match_of[j] = i
    
    # Find path starts: half-edges whose "from" vertex is not the "to" vertex of their match
    # Actually, a path start is a half-edge i where either:
    # 1. i is unmatched, OR  
    # 2. i's match j has j_from != i_to (which shouldn't happen by construction)
    
    # Let me think differently. The matching connects half-edges that share a vertex.
    # If half_edge[i] = (u, v, e1) is matched with half_edge[j] = (v, w, e2),
    # this means in the path, we go ...u, v, w...
    # So the "tail" of i (u) connects to the "head" of j (w) through v.
    
    # To reconstruct: 
    # - A half-edge i = (u, v, e) is a "start" if it's unmatched OR if its match doesn't continue from u
    # - Actually, let's trace: each original edge appears twice (as two half-edges)
    # - A path is a sequence of original edges where consecutive edges share a vertex
    
    # Better approach: build directed graph of connections
    # If half_edge[i]=(u,v,e1) matches half_edge[j]=(v,w,e2), then e1→e2 via vertex v
    # This means in some path, edge e1 is followed by edge e2
    
    edge_next = [None] * n_edges  # edge_next[e1] = (e2, shared_vertex) if e1 is followed by e2
    edge_prev = [None] * n_edges
    
    for i, j in matching:
        u1, v1, e1 = half_edges[i]
        u2, v2, e2 = half_edges[j]
        # i is (u1,v1,e1), j is (u2,v2,e2), and v1 == u2 (the shared vertex)
        # So e1 goes to e2 through vertex v1
        if edge_next[e1] is None:
            edge_next[e1] = (e2, v1)
        if edge_prev[e2] is None:
            edge_prev[e2] = (e1, v1)
    
    # Now trace paths
    # A path starts at an edge e where edge_prev[e] is None
    # A path ends at an edge e where edge_next[e] is None
    
    paths = []
    visited_edges = set()
    
    for e in range(n_edges):
        if e in visited_edges:
            continue
        # Start a new path from edge e
        path_edges = []
        current = e
        
        # Find the start: go back as far as possible
        while edge_prev[current] is not None and current not in visited_edges:
            prev_e, shared_v = edge_prev[current]
            if prev_e in visited_edges:
                break
            path_edges.append((current, shared_v))
            current = prev_e
        
        # Now go forward from current
        path_edges.append((current, None))  # current is the first edge
        current = e
        while edge_next[current] is not None and current not in visited_edges:
            next_e, shared_v = edge_next[current]
            if next_e in visited_edges:
                break
            path_edges.append((next_e, shared_v))
            current = next_e
        
        # Convert edge sequence to vertex sequence
        # path_edges is a list of (edge_idx, shared_vertex_with_next)
        # We need to order them properly
        
        visited_edges.update(e for e, _ in path_edges)
        
        if len(path_edges) == 0:
            continue
        
        # Reconstruct vertex list from edge sequence
        # The edges in path_edges should be in order from start to end
        # Let's sort them by following the chain
        
        # Actually, let me rebuild this more carefully
        # Start from the beginning edge and follow next pointers
        start_e = path_edges[0][0]
        vertex_list = []
        cur_e = start_e
        while cur_e is not None and cur_e not in visited_edges:
            u, v = edges[cur_e]
            if not vertex_list:
                vertex_list.append(u)
                vertex_list.append(v)
            else:
                # Check if the last vertex matches
                last_v = vertex_list[-1]
                if last_v == u:
                    vertex_list.append(v)
                elif last_v == v:
                    vertex_list.append(u)
                else:
                    # Mismatch - something went wrong
                    break
            visited_edges.add(cur_e)
            if edge_next[cur_e] is not None:
                cur_e, _ = edge_next[cur_e]
            else:
                cur_e = None
        
        if len(vertex_list) >= 2:
            # Verify it's a valid simple path
            if _is_valid_path(vertex_list, edges):
                paths.append(vertex_list)
    
    # Check if all edges are covered
    covered_edges = set()
    for path in paths:
        for i in range(len(path) - 1):
            e = tuple(sorted((path[i], path[i+1])))
            covered_edges.add(e)
    
    if covered_edges != set(edges):
        # Some edges not covered - fallback
        return None
    
    # Verify all paths are valid
    for path in paths:
        if not _is_valid_path(path, edges):
            return None
    
    return paths


def _is_valid_path(path, edges):
    """Check if path is simple and uses only graph edges."""
    if len(path) < 2:
        return False
    if len(set(path)) != len(path):
        return False  # Not simple
    edge_set = set(tuple(sorted(e)) for e in edges)
    for i in range(len(path) - 1):
        e = tuple(sorted((path[i], path[i+1])))
        if e not in edge_set:
            return False
    return True


def _find_maximum_matching(adj, n, deadline):
    """
    Find maximum cardinality matching in a general graph.
    Uses greedy initialization + local search (augmenting path search).
    """
    if n == 0:
        return []
    
    # Greedy initialization
    matched = [-1] * n
    matching = []
    
    # Sort vertices by degree for better greedy
    order = sorted(range(n), key=lambda v: len(adj.get(v, [])), reverse=True)
    
    for v in order:
        if matched[v] != -1:
            continue
        # Find an unmatched neighbor
        for u in adj.get(v, []):
            if matched[u] == -1:
                matched[v] = u
                matched[u] = v
                matching.append((v, u))
                break
    
    # Local search: try to find augmenting paths
    # Use randomized restarts for augmenting path search
    best_matching = list(matching)
    best_matched = list(matched)
    
    rng = random.Random(42)
    
    # Try to improve with augmenting path search
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
        
        # Pick a random unmatched vertex
        unmatched = [v for v in range(n) if best_matched[v] == -1]
        if not unmatched:
            break
        
        start = rng.choice(unmatched)
        
        # BFS/DFS to find augmenting path from start
        # An augmenting path starts and ends at unmatched vertices,
        # alternating between matched and unmatched edges
        
        parent = [-1] * n
        parent_edge = [-1] * n  # -1 for unmatched edge, otherwise the matched partner
        visited = [False] * n
        
        queue = [start]
        visited[start] = True
        found_end = -1
        
        while queue:
            if time.time() > deadline - 5:
                break
            v = queue.pop(0)
            for u in adj.get(v, []):
                if not visited[u]:
                    visited[u] = True
                    parent[u] = v
                    # Check if u is matched
                    if best_matched[u] != -1:
                        # Follow the matched edge
                        w = best_matched[u]
                        if not visited[w]:
                            visited[w] = True
                            parent[w] = u
                            parent_edge[w] = u  # came via matched edge from u
                            queue.append(w)
                    else:
                        # u is unmatched - found augmenting path!
                        found_end = u
                        break
            if found_end != -1:
                break
        
        if found_end == -1:
            continue
        
        # Augment along the path
        # Trace back from found_end to start
        path_vertices = []
        cur = found_end
        while cur != -1:
            path_vertices.append(cur)
            if parent[cur] == -1:
                break
            cur = parent[cur]
        
        # The path is start -> ... -> found_end
        # Alternate: unmatched, matched, unmatched, matched, ...
        # Flip: unmatched becomes matched, matched becomes unmatched
        
        # Reconstruct the augmenting path properly
        # parent[v] = u means we reached v from u
        # If v is matched, the edge (u,v) is unmatched, and (v, matched[v]) is matched
        
        # Let me redo this with a cleaner augmenting path algorithm
        
        # Actually, let's use a simpler local search: try swapping matches
        # Pick two matched edges and see if swapping helps
        improved = False
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            
            # Pick a random matched edge
            if not best_matching:
                break
            idx = rng.randrange(len(best_matching))
            v, u = best_matching[idx]
            
            # Try to find an alternative match for v or u
            # For simplicity, just try removing this match and re-matching greedily
            # This is a weak local search but may help
            
            # Alternative: try to find an augmenting path from a random unmatched vertex
            unmatched = [x for x in range(n) if best_matched[x] == -1]
            if not unmatched:
                break
            
            s = rng.choice(unmatched)
            
            # DFS for augmenting path
            stack = [(s, -1, [])]  # (vertex, parent, path)
            visited_local = {s}
            found = False
            
            while stack and not found:
                if time.time() > deadline - 5:
                    break
                node, par, path_so_far = stack.pop()
                
                for nb in adj.get(node, []):
                    if nb in visited_local:
                        continue
                    
                    if best_matched[nb] == -1:
                        # Found augmenting path!
                        # path: s -> ... -> node -> nb
                        # Flip along this path
                        full_path = path_so_far + [node, nb]
                        
                        # Build the new matching
                        # The path alternates: unmatched edges and matched edges
                        # s is unmatched, so first edge (s, next) is unmatched
                        # Then matched, then unmatched, etc.
                        
                        # For now, just accept if it increases matching size
                        # This is a simplification - proper augmentation needed
                        improved = True
                        found = True
                        break
                    else:
                        # nb is matched to best_matched[nb]
                        w = best_matched[nb]
                        if w not in visited_local:
                            visited_local.add(nb)
                            visited_local.add(w)
                            stack.append((w, nb, path_so_far + [node, nb]))
            
            if found:
                break
        
        if not improved:
            # Try a different strategy: random restart of greedy
            if iteration % 10 == 0:
                new_matched = [-1] * n
                new_matching = []
                order2 = sorted(range(n), key=lambda v: len(adj.get(v, [])), reverse=True)
                rng.shuffle(order2)
                for v in order2:
                    if new_matched[v] != -1:
                        continue
                    for u in adj.get(v, []):
                        if new_matched[u] == -1:
                            new_matched[v] = u
                            new_matched[u] = v
                            new_matching.append((v, u))
                            break
                if len(new_matching) > len(best_matching):
                    best_matching = new_matching
                    best_matched = new_matched
    
    return best_matching


def _greedy_decompose(edges, rng, deadline):
    """
    Greedy path decomposition fallback.
    """
    remaining = set(edges)
    if not remaining:
        return []
    
    adj = defaultdict(set)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    degree = defaultdict(int)
    for u, v in remaining:
        degree[u] += 1
        degree[v] += 1
    
    paths = []
    
    while remaining:
        if time.time() > deadline:
            return None
        
        start = _pick_start(adj, degree, rng)
        if start is None:
            candidates = [v for v in adj if adj[v]]
            if not candidates:
                break
            start = rng.choice(candidates)
        
        path = _find_long_path(start, adj, degree, rng, remaining, deadline)
        
        if path is None or len(path) < 2:
            if adj[start]:
                next_v = rng.choice(list(adj[start]))
                path = [start, next_v]
            else:
                e = next(iter(remaining))
                path = [e[0], e[1]]
        
        path_edges = set()
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge = tuple(sorted((u, v)))
            if edge in remaining:
                path_edges.add(edge)
        
        if not path_edges:
            break
        
        for e in path_edges:
            remaining.discard(e)
        
        for u, v in path_edges:
            adj[u].discard(v)
            adj[v].discard(u)
            degree[u] -= 1
            degree[v] -= 1
        
        for v in list(adj.keys()):
            if not adj[v]:
                del adj[v]
        
        paths.append(path)
    
    return paths


def _pick_start(adj, degree, rng):
    """Pick a starting vertex, preferring odd-degree vertices."""
    odd_vertices = [v for v in adj if degree.get(v, 0) % 2 == 1 and adj[v]]
    if odd_vertices:
        return rng.choice(odd_vertices)
    
    candidates = [v for v in adj if adj[v]]
    if not candidates:
        return None
    
    weights = [max(1, degree.get(v, 0)) for v in candidates]
    total = sum(weights)
    r = rng.random() * total
    cum = 0
    for v, w in zip(candidates, weights):
        cum += w
        if r <= cum:
            return v
    return candidates[-1]


def _find_long_path(start, adj, degree, rng, remaining, deadline):
    """Find a long simple path starting from start."""
    best_path = [start]
    _greedy_extend(start, adj, best_path, rng, deadline)
    return best_path if len(best_path) >= 2 else None


def _greedy_extend(start, adj, best_path, rng, deadline):
    """Greedy path extension."""
    path = [start]
    visited = {start}
    
    max_depth = 80
    for step in range(max_depth):
        if time.time() > deadline:
            break
        current = path[-1]
        neighbors = [nb for nb in adj[current] if nb not in visited]
        
        if not neighbors:
            break
        
        def score(nb):
            unvisited_count = sum(1 for x in adj[nb] if x not in visited)
            deg_bonus = 1.0 / (1.0 + adj[nb].__len__() * 0.1)
            return unvisited_count * 1.0 - deg_bonus * 0.5
        
        if rng.random() < 0.80:
            neighbors.sort(key=score)
            top_k = min(3, len(neighbors))
            choice = rng.choice(neighbors[:top_k])
        else:
            choice = rng.choice(neighbors)
        
        path.append(choice)
        visited.add(choice)
        
        if len(path) > len(best_path):
            best_path[:] = path[:]
    
    return path


def _merge_paths(paths, edges):
    """Post-process: try to merge paths that share an endpoint."""
    if not paths:
        return paths
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    changed = True
    while changed:
        changed = False
        new_paths = list(paths)
        
        for i in range(len(new_paths)):
            if time.time() > 145:
                break
            for j in range(len(new_paths)):
                if i == j:
                    continue
                merged = _try_merge(new_paths[i], new_paths[j], edge_set)
                if merged is not None:
                    new_paths.pop(j)
                    new_paths[i] = merged
                    changed = True
                    break
            if changed:
                break
        
        paths = new_paths
    
    return paths


def _try_merge(path_a, path_b, edge_set):
    """Try to merge two paths into one."""
    if not path_a or not path_b:
        return None
    
    for a in (path_a, path_a[::-1]):
        for b in (path_b, path_b[::-1]):
            if a[-1] == b[0]:
                merged = a + b[1:]
                if _is_simple_path(merged, edge_set):
                    return merged
            if a[0] == b[-1]:
                merged = a[::-1] + b[:-1][::-1]
                if _is_simple_path(merged, edge_set):
                    return merged
    
    return None


def _is_simple_path(path, edge_set):
    """Check if path is simple and all edges exist."""
    if len(path) < 2:
        return True
    
    if len(set(path)) != len(path):
        return False
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = tuple(sorted((u, v)))
        if edge not in edge_set:
            return False
    
    return True

# EVOLVE-BLOCK-END
