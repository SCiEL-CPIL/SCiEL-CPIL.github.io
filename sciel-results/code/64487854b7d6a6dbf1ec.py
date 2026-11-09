"""Find longest simple path in pinned graph using matching seed + Warnsdorff + improved DFS."""
import time
from collections import deque
import heapq

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

def find_components():
    visited = set()
    comps = []
    for s in range(N):
        if s not in visited:
            comp = set()
            stack = [s]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.add(u)
                for v in ADJ[u]:
                    if v not in visited:
                        stack.append(v)
            comps.append(comp)
    return comps

def find_max_matching_bipartite(comp_set):
    """
    Find maximum cardinality matching in bipartite graph (U, V, E)
    where U = V = comp_set, and edge (u,v) in bipartite graph iff {u,v} in original graph.
    Uses Hopcroft-Karp.
    """
    vertices = sorted(comp_set)
    idx_map = {v: i for i, v in enumerate(vertices)}
    n = len(vertices)
    
    # Build adjacency list for left side
    adj_left = [[] for _ in range(n)]
    for u in vertices:
        for v in ADJ[u]:
            if v in comp_set and v != u:
                adj_left[idx_map[u]].append(idx_map[v])
    
    # Hopcroft-Karp
    pair_u = [-1] * n  # match for left node
    pair_v = [-1] * n  # match for right node
    dist = [0] * n
    
    def bfs():
        from collections import deque as dq
        q = dq()
        found = False
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = float('inf')
        while q:
            u = q.popleft()
            for v in adj_left[u]:
                if pair_v[v] == -1:
                    found = True
                elif dist[pair_v[v]] == float('inf'):
                    dist[pair_v[v]] = dist[u] + 1
                    q.append(pair_v[v])
        return found
    
    def dfs(u):
        for v in adj_left[u]:
            if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = float('inf')
        return False
    
    matching_size = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1:
                if dfs(u):
                    matching_size += 1
    
    # Extract matching edges
    matching = []
    for u in range(n):
        if pair_u[u] != -1:
            matching.append((vertices[u], vertices[pair_u[u]]))
    
    return matching, matching_size

def matching_to_path(matching, comp_set):
    """
    Convert a matching to a path by chaining matched edges.
    The matching forms a set of disjoint edges. We can try to chain them
    by finding sequences where end of one edge connects to start of another.
    """
    if not matching:
        return []
    
    # Build a directed graph from matching: each matched pair (u,v) gives edges u->v and v->u
    # Actually, matching edges are undirected. We need to find a path that uses these edges.
    # A matching is a set of disjoint edges, so the "path" from matching is just the set of edges.
    # But we want a simple path in the original graph.
    
    # Better approach: use the matching as a lower bound on path length.
    # If we have k matched edges, we can form a path of at most 2k vertices if they chain.
    # But matching edges are disjoint, so they don't chain directly.
    
    # Instead, let's build a path greedily using the matched edges as hints.
    # Start from a vertex in the matching and extend.
    
    # Actually, the key insight: a maximum matching in the bipartite double cover
    # gives a lower bound on the length of the longest path.
    # For a graph with a matching of size m, there exists a path of length at least 2m-1 vertices?
    # Not exactly. Let me think differently.
    
    # Simpler: just use greedy from endpoints of matched edges
    best_path = []
    matched_vertices = set()
    for u, v in matching:
        matched_vertices.add(u)
        matched_vertices.add(v)
    
    # Try starting from each matched vertex
    for start in matched_vertices:
        path = [start]
        seen = {start}
        while True:
            cand = ADJ[path[-1]] & comp_set - seen
            if not cand:
                break
            # Prefer vertices in matching
            in_match = cand & matched_vertices
            if in_match:
                # Pick the one with fewest unseen neighbors (Warnsdorff)
                v = min(in_match, key=lambda w: len(ADJ[w] & comp_set - seen))
            else:
                v = min(cand, key=lambda w: len(ADJ[w] & comp_set - seen))
            path.append(v)
            seen.add(v)
        
        if len(path) > len(best_path):
            best_path = path
    
    return best_path

def warnsdorff_score(v, seen, comp_set):
    """Prefer vertices with fewer unseen neighbors (Warnsdorff's rule)."""
    return (len(ADJ[v] & comp_set - seen), v)

def greedy_path_from(start, comp_set, seen=None):
    """Greedy Warnsdorff path from start."""
    if seen is None:
        seen = {start}
    else:
        seen = set(seen)
    path = [start]
    while True:
        cand = ADJ[path[-1]] & comp_set - seen
        if not cand:
            break
        v = min(cand, key=lambda w: warnsdorff_score(w, seen, comp_set))
        path.append(v)
        seen.add(v)
    return path

def extend_both_ends(path, seen, comp_set):
    """Extend path greedily from both ends."""
    path = list(path)
    seen = set(seen)
    changed = True
    while changed:
        changed = False
        # Extend front
        while True:
            cand = ADJ[path[0]] & comp_set - seen
            if not cand:
                break
            v = min(cand, key=lambda w: warnsdorff_score(w, seen, comp_set))
            path.insert(0, v)
            seen.add(v)
            changed = True
        # Extend back
        while True:
            cand = ADJ[path[-1]] & comp_set - seen
            if not cand:
                break
            v = min(cand, key=lambda w: warnsdorff_score(w, seen, comp_set))
            path.append(v)
            seen.add(v)
            changed = True
    return path

def time_bounded_dfs(start, comp_set, deadline, best_path, best_len, depth_limit=None):
    """Time-bounded DFS to find longest path starting from start within comp."""
    best = [best_path[:], best_len]
    max_depth = len(comp_set) if depth_limit is None else depth_limit
    
    def dfs(path, seen):
        if time.time() > deadline:
            return
        if len(path) > best[1]:
            best[0] = path[:]
            best[1] = len(path)
        if len(path) == len(comp_set):
            return
        if depth_limit is not None and len(path) >= depth_limit:
            return
        cand = ADJ[path[-1]] & comp_set - seen
        if not cand:
            return
        # Sort by Warnsdorff heuristic
        cand_sorted = sorted(cand, key=lambda w: warnsdorff_score(w, seen, comp_set))
        for v in cand_sorted:
            path.append(v)
            seen.add(v)
            dfs(path, seen)
            path.pop()
            seen.remove(v)
            if time.time() > deadline:
                return
    
    dfs([start], {start})
    return best[0]

def local_search_extend(path, comp_set, deadline):
    """
    Try to improve path by local modifications:
    1. Try replacing internal edges to extend
    2. Try reordering segments
    """
    path = list(path)
    seen = set(path)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to extend from both ends
        new_path = extend_both_ends(path, seen, comp_set)
        if len(new_path) > len(path):
            path = new_path
            seen = set(path)
            improved = True
        
        # Try to find a vertex that can be inserted into the path
        # For each pair of consecutive vertices (u,v) in path, check if there's
        # a vertex w not in path such that u-w and w-v are edges
        for i in range(len(path) - 1):
            if time.time() > deadline:
                return path
            u, v = path[i], path[i+1]
            # Find common neighbors of u and v not in path
            common = (ADJ[u] & ADJ[v]) & comp_set - seen
            if common:
                # Insert the one with best Warnsdorff score
                w = min(common, key=lambda x: warnsdorff_score(x, seen, comp_set))
                # Check if inserting w maintains simplicity
                # w is not in seen, so it's fine
                path.insert(i+1, w)
                seen.add(w)
                improved = True
                break  # Restart scan after insertion
    
    return path

def construct():
    t0 = time.time()
    deadline = t0 + 85  # Safety margin
    
    # Find components
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    if not comps:
        return [0]
    
    largest_comp = comps[0]
    comp_size = len(largest_comp)
    comp_list = sorted(largest_comp)
    
    best_path = []
    
    # Phase 1: Use matching to get a strong initial seed
    matching, match_size = find_max_matching_bipartite(largest_comp)
    match_path = matching_to_path(matching, largest_comp)
    if len(match_path) > len(best_path):
        best_path = list(match_path)
    
    # Phase 2: Greedy + extend from every vertex in largest component
    for start in comp_list:
        if time.time() > deadline:
            break
        p = greedy_path_from(start, largest_comp)
        p = extend_both_ends(p, set(p), largest_comp)
        if len(p) > len(best_path):
            best_path = list(p)
    
    # If we found Hamiltonian path in largest component, done
    if len(best_path) >= comp_size:
        for i in range(len(best_path) - 1):
            if best_path[i+1] not in ADJ[best_path[i]]:
                break
        else:
            return best_path
    
    # Phase 3: Local search to extend best path
    if best_path:
        local_dl = min(deadline, time.time() + 10)
        best_path = local_search_extend(best_path, largest_comp, local_dl)
    
    # If we found Hamiltonian path, done
    if len(best_path) >= comp_size:
        for i in range(len(best_path) - 1):
            if best_path[i+1] not in ADJ[best_path[i]]:
                break
        else:
            return best_path
    
    # Phase 4: Time-bounded DFS from promising starts
    promising_starts = sorted(comp_list, key=lambda x: (-len(ADJ[x] & largest_comp), x))
    
    if best_path:
        for v in [best_path[0], best_path[-1], best_path[len(best_path)//2]]:
            if v not in promising_starts:
                promising_starts.insert(0, v)
    
    for start in promising_starts:
        if time.time() > deadline:
            break
        if len(best_path) >= comp_size:
            break
        
        remaining = deadline - time.time()
        if remaining < 0.5:
            break
        budget = min(5.0, remaining * 0.3)
        dl = time.time() + budget
        
        local_best = time_bounded_dfs(start, largest_comp, dl, best_path, len(best_path))
        if len(local_best) > len(best_path):
            best_path = local_best
    
    # Phase 5: Final local search
    if best_path:
        final_dl = min(deadline, time.time() + 15)
        best_path = local_search_extend(best_path, largest_comp, final_dl)
    
    # Phase 6: Try to extend the best path found
    if best_path:
        best_path = extend_both_ends(best_path, set(best_path), largest_comp)
        best_path = [v for v in best_path if v in largest_comp]
    
    # Final validation
    if not best_path:
        best_path = [comp_list[0]]
    
    for i in range(len(best_path) - 1):
        if best_path[i+1] not in ADJ[best_path[i]]:
            return [comp_list[0]]
    if len(set(best_path)) != len(best_path):
        seen = set()
        fixed = []
        for v in best_path:
            if v not in seen:
                fixed.append(v)
                seen.add(v)
        return fixed
    
    return best_path
