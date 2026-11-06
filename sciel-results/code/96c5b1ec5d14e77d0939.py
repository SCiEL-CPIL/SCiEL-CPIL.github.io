"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
import time
import random
from collections import defaultdict, deque
import numpy as np
from scipy.optimize import linear_sum_assignment


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _hopcroft_karp(n, edges):
    """
    Maximum matching in the bipartite double cover of the graph.
    Edge (u,v) in G creates (L_u,R_v) and (L_v,R_u).
    Returns matching size.
    """
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)

    pair_u = [-1] * n
    pair_v = [-1] * n
    INF = float('inf')
    dist = [0] * n

    def bfs():
        queue = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                w = pair_v[v]
                if w != -1 and dist[w] == INF:
                    dist[w] = dist[u] + 1
                    queue.append(w)
                elif w == -1:
                    found = True
        return found

    def dfs(u):
        for v in adj[u]:
            w = pair_v[v]
            if w == -1 or (dist[w] == dist[u] + 1 and dfs(w)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    matching_size = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1:
                if dfs(u):
                    matching_size += 1

    return matching_size


def _beam_longest_path(start, remaining_edges, rem_adj, visited, beam_width=20, max_depth=60):
    """
    Beam search for a long simple path starting at `start` in the remaining graph.
    Uses Warnsdorff's rule (prefer nodes with fewest onward moves) for ordering.
    Returns a list of vertices (the path) or [start] if it cannot extend.
    """
    # Each beam entry: (path_list, frozenset_visited, current_node)
    beam = [([start], frozenset({start}), start)]
    best_path = [start]
    best_len = 1

    for _ in range(max_depth):
        new_beam = []
        for path, vset, cur in beam:
            # Candidates: neighbors of cur via remaining edges, not in visited
            cands = []
            for nb in rem_adj[cur]:
                if nb in vset:
                    continue
                e = (min(cur, nb), max(cur, nb))
                if e not in remaining_edges:
                    continue
                # Warnsdorff: count onward moves from nb
                onward = 0
                for m in rem_adj[nb]:
                    if m in vset:
                        continue
                    e2 = (min(nb, m), max(nb, m))
                    if e2 in remaining_edges:
                        onward += 1
                cands.append((onward, nb))
            if not cands:
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
                continue
            # Sort by fewest onward moves (Warnsdorff), tie-break by degree desc
            cands.sort(key=lambda x: (x[0], -len(rem_adj[x[1]])))
            for _, nb in cands[:beam_width]:
                new_path = path + [nb]
                new_vset = vset | {nb}
                new_beam.append((new_path, new_vset, nb))
        if not new_beam:
            break
        # Keep only the longest paths in the beam
        new_beam.sort(key=lambda x: -len(x[0]))
        beam = new_beam[:beam_width]
        longest = len(beam[0][0])
        if longest > best_len:
            best_len = longest
            best_path = beam[0][0][:]

    return best_path


def _greedy_decomposition(n, edges, rng, use_beam=True):
    """
    Greedy path decomposition. For each path, pick a good starting edge and
    either extend greedily (both directions) or use beam search for the longest
    simple path. Edges are edge-disjoint and cover all edges.
    """
    remaining_edges = set()
    for e in edges:
        remaining_edges.add((min(e[0], e[1]), max(e[0], e[1])))

    rem_adj = defaultdict(set)
    for (u, v) in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)

    paths = []

    while remaining_edges:
        # Pick starting edge: max sum of remaining degrees, random tie-break
        best_score = -1
        candidates = []
        for e in remaining_edges:
            u, v = e
            score = len(rem_adj[u]) + len(rem_adj[v])
            if score > best_score:
                best_score = score
                candidates = [e]
            elif score == best_score:
                candidates.append(e)

        if not candidates:
            break

        start_edge = rng.choice(candidates)
        u, v = start_edge

        if use_beam:
            # Try beam search from both endpoints, take the longer
            path_u = _beam_longest_path(u, remaining_edges, rem_adj, frozenset({u}), beam_width=20, max_depth=n + 5)
            path_v = _beam_longest_path(v, remaining_edges, rem_adj, frozenset({v}), beam_width=20, max_depth=n + 5)
            if len(path_v) > len(path_u):
                path = path_v
            else:
                path = path_u
            # Ensure the start edge is in the path; if not, fall back to greedy
            edge_set = set()
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                edge_set.add((min(a, b), max(a, b)))
            if start_edge not in edge_set:
                # Beam didn't include the chosen edge; just use beam result if valid
                if len(path) < 2:
                    path = [u, v]
        else:
            # Pure greedy extension both directions
            path = [u, v]
            visited = {u, v}

            remaining_edges.discard(start_edge)
            rem_adj[u].discard(v)
            rem_adj[v].discard(u)

            current = v
            while True:
                neighbors = [nb for nb in rem_adj[current] if nb not in visited]
                if not neighbors:
                    break
                best_deg = -1
                cand_nbs = []
                for nb in neighbors:
                    deg = len(rem_adj[nb])
                    if deg > best_deg:
                        best_deg = deg
                        cand_nbs = [nb]
                    elif deg == best_deg:
                        cand_nbs.append(nb)
                next_v = rng.choice(cand_nbs)
                path.append(next_v)
                visited.add(next_v)
                e = (min(current, next_v), max(current, next_v))
                remaining_edges.discard(e)
                rem_adj[current].discard(next_v)
                rem_adj[next_v].discard(current)
                current = next_v

            current = u
            while True:
                neighbors = [nb for nb in rem_adj[current] if nb not in visited]
                if not neighbors:
                    break
                best_deg = -1
                cand_nbs = []
                for nb in neighbors:
                    deg = len(rem_adj[nb])
                    if deg > best_deg:
                        best_deg = deg
                        cand_nbs = [nb]
                    elif deg == best_deg:
                        cand_nbs.append(nb)
                next_v = rng.choice(cand_nbs)
                path.insert(0, next_v)
                visited.add(next_v)
                e = (min(current, next_v), max(current, next_v))
                remaining_edges.discard(e)
                rem_adj[current].discard(next_v)
                rem_adj[next_v].discard(current)
                current = next_v

        # Remove all edges used by the path
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            e = (min(a, b), max(a, b))
            if e in remaining_edges:
                remaining_edges.discard(e)
                rem_adj[a].discard(b)
                rem_adj[b].discard(a)

        paths.append(path)

    return paths


def _solve_graph(n, p, seed, time_budget=30.0):
    edges = _instance(n, p, seed)
    m = len(edges)
    if m == 0:
        return []

    max_matching = _hopcroft_karp(n, edges)
    lower_bound = n - 1 - max_matching
    if lower_bound < 1:
        lower_bound = 1

    best_paths = None
    best_count = float('inf')

    start_time = time.time()
    rng = random.Random(seed + 100)

    # Try exact matching-based approach first
    exact_paths = _exact_path_decomposition(n, edges)
    if exact_paths is not None:
        count = len(exact_paths)
        if count < best_count:
            best_count = count
            best_paths = exact_paths
        if best_count == lower_bound:
            return best_paths

    attempts = 0
    # Alternate between beam-based and pure greedy to diversify
    while time.time() - start_time < time_budget:
        use_beam = (attempts % 3 != 2)  # 2 out of 3 use beam
        paths = _greedy_decomposition(n, edges, rng, use_beam=use_beam)
        count = len(paths)

        if count < best_count:
            best_count = count
            best_paths = paths

        if best_count == lower_bound:
            break

        attempts += 1
        if attempts > 300:
            break

    if best_paths is None:
        best_paths = [[u, v] for (u, v) in edges]

    return best_paths


def _exact_path_decomposition(n, edges):
    """
    Exact minimum path decomposition via maximum bipartite matching on directed edges.
    
    Formulation:
    - For each undirected edge {u,v}, create two directed edges: u->v and v->u.
    - Build bipartite graph: Left = all directed edges, Right = all directed edges.
    - Connect (a->b)_L to (c->d)_R if b==c and {a,b} != {c,d}.
    - Maximum matching gives max number of edge-chains.
    - Reconstruct paths from the matching.
    
    Returns list of paths or None if infeasible.
    """
    m = len(edges)
    if m <= 1:
        return [[u, v] for (u, v) in edges]
    
    # Create directed edges
    dir_edges = []  # list of (tail, head, undirected_edge_index)
    for idx, (u, v) in enumerate(edges):
        dir_edges.append((u, v, idx))
        dir_edges.append((v, u, idx))
    
    d = len(dir_edges)  # 2*m
    
    # Build adjacency: for each directed edge i, which directed edges j can follow it?
    # dir_edges[i] = (tail_i, head_i, und_idx_i)
    # Can follow if head_i == tail_j and und_idx_i != und_idx_j
    from collections import defaultdict
    head_to_dir = defaultdict(list)
    for j, (tail, head, und_idx) in enumerate(dir_edges):
        head_to_dir[tail].append(j)
    
    # Cost matrix for Hungarian: cost[i][j] = 0 if dir_edges[i] can precede dir_edges[j], else large
    BIG = 10**9
    cost = np.full((d, d), BIG, dtype=np.int64)
    for i in range(d):
        tail_i, head_i, und_i = dir_edges[i]
        for j in head_to_dir[head_i]:
            tail_j, head_j, und_j = dir_edges[j]
            if und_i != und_j:
                cost[i][j] = 0
    
    # Solve minimum cost assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Count number of 0-cost assignments (valid connections)
    num_connections = 0
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] == 0:
            num_connections += 1
    
    # Minimum number of paths = m - num_connections
    min_paths = m - num_connections
    
    # Reconstruct paths from the matching
    # For each directed edge i, find which directed edge j follows it (if any)
    next_edge = [-1] * d
    prev_edge = [-1] * d
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] == 0:
            next_edge[r] = c
            prev_edge[c] = r
    
    # Find path starts: directed edges with no predecessor
    path_starts = [i for i in range(d) if prev_edge[i] == -1]
    
    paths = []
    used_und_edges = set()
    
    for start in path_starts:
        path_vertices = []
        cur = start
        while cur != -1:
            tail, head, und_idx = dir_edges[cur]
            if not path_vertices:
                path_vertices.append(tail)
            path_vertices.append(head)
            used_und_edges.add(und_idx)
            cur = next_edge[cur]
        
        if len(path_vertices) >= 2:
            paths.append(path_vertices)
    
    # Check if all edges are covered
    if len(used_und_edges) != m:
        # Some edges not covered - this shouldn't happen but handle gracefully
        return None
    
    # Verify paths are valid (simple, use graph edges, edge-disjoint)
    edge_set = set()
    for e in edges:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
    
    used = set()
    valid_paths = []
    for path in paths:
        # Check simplicity
        if len(path) != len(set(path)):
            return None
        # Check edges
        valid = True
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            e = (min(a, b), max(a, b))
            if e not in edge_set or e in used:
                valid = False
                break
        if not valid:
            return None
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            used.add((min(a, b), max(a, b)))
        valid_paths.append(path)
    
    if len(used) != m:
        return None
    
    return valid_paths


def _validate_and_fix(n, edges, paths):
    """Ensure paths are simple, use graph edges, edge-disjoint, cover all edges.
    Returns a corrected list of paths (may split invalid ones into edges)."""
    edge_set = set()
    for e in edges:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))

    fixed = []
    used = set()
    for path in paths:
        # Check simplicity
        if len(path) != len(set(path)):
            # Split into single edges
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                e = (min(a, b), max(a, b))
                if e in edge_set and e not in used:
                    fixed.append([a, b])
                    used.add(e)
            continue
        # Check each consecutive pair is an edge and unused
        valid = True
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            e = (min(a, b), max(a, b))
            if e not in edge_set or e in used:
                valid = False
                break
        if valid:
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                used.add((min(a, b), max(a, b)))
            fixed.append(path)
        else:
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                e = (min(a, b), max(a, b))
                if e in edge_set and e not in used:
                    fixed.append([a, b])
                    used.add(e)

    # Add any uncovered edges as single-edge paths
    for e in edge_set:
        if e not in used:
            fixed.append([e[0], e[1]])
            used.add(e)

    return fixed


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            edges = _instance(*spec)
            results.append([[u, v] for (u, v) in edges])
        else:
            remaining_time = deadline - elapsed - 10
            time_budget = max(1.0, min(45.0, remaining_time / 3))
            n, p, seed = spec
            paths = _solve_graph(n, p, seed, time_budget=time_budget)
            paths = _validate_and_fix(n, _instance(n, p, seed), paths)
            results.append(paths)

    return results
