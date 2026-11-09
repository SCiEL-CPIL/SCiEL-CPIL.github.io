"""Find longest simple path in the pinned graph using IDDFS with Warnsdorff + refinement."""
import time
import random
import sys
sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)


def find_components():
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in ADJ[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            comps.append(comp)
    return comps


def greedy_path_from(start, adj, seen=None):
    if seen is None:
        seen = set()
    path = [start]
    seen.add(start)
    while True:
        cur = path[-1]
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            break
        v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        path.append(v)
        seen.add(v)
    return path


def extend_both_ends(path, adj):
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        if path:
            cur = path[0]
            candidates = [nb for nb in adj[cur] if nb not in seen]
            if candidates:
                v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
                path.insert(0, v)
                seen.add(v)
                changed = True
        if path:
            cur = path[-1]
            candidates = [nb for nb in adj[cur] if nb not in seen]
            if candidates:
                v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
                path.append(v)
                seen.add(v)
                changed = True
    return path


def try_insert(path, adj):
    path = list(path)
    seen = set(path)
    unvisited = set(range(N)) - seen
    improved = True
    while improved and unvisited:
        improved = False
        for w in list(unvisited):
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                if w in adj[u] and w in adj[v]:
                    path.insert(i + 1, w)
                    seen.add(w)
                    unvisited.remove(w)
                    improved = True
                    break
            if improved:
                break
    return path


def two_opt_refine(path, adj, deadline):
    """Try to extend path by reversing segments to create new endpoint opportunities."""
    path = list(path)
    best = list(path)
    n = len(path)
    for i in range(n - 2):
        if time.time() > deadline:
            break
        for j in range(i + 2, n):
            if time.time() > deadline:
                break
            # Try reversing path[i+1:j+1]
            new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
            # Check validity: edge from path[i] to new_path[i+1] and new_path[j] to path[j+1]
            if i > 0 and new_path[i] not in adj[new_path[i-1]]:
                continue
            if j < n - 1 and new_path[j] not in adj[new_path[j+1]]:
                continue
            # Check internal edges of reversed segment
            valid = True
            for k in range(i+1, j+1):
                if new_path[k] not in adj[new_path[k-1]]:
                    valid = False
                    break
            if not valid:
                continue
            # Extend both ends
            extended = extend_both_ends(new_path, adj)
            if len(extended) > len(best):
                best = extended
    return best


def iddfs_path(start, adj, deadline):
    """Iterative deepening DFS with Warnsdorff heuristic to find long paths."""
    best_path = [start]
    best_len = 1
    
    def dfs(cur, path, seen, remaining):
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        if len(path) > best_len:
            best_path = list(path)
            best_len = len(path)
        if remaining <= 0:
            return
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            return
        # Warnsdorff: sort by fewest unvisited neighbors
        candidates.sort(key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        for v in candidates:
            path.append(v)
            seen.add(v)
            dfs(v, path, seen, remaining - 1)
            path.pop()
            seen.remove(v)
            if time.time() > deadline:
                return
    
    # Try increasing depths with time-aware allocation
    depths = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 99]
    for depth in depths:
        if time.time() > deadline:
            break
        seen = {start}
        dfs(start, [start], seen, depth)
        if best_len >= N:
            break
    return best_path


def backtracking_extend(path, adj, deadline, max_depth=15):
    """Try to extend path at both ends using limited backtracking."""
    path = list(path)
    seen = set(path)
    
    def extend_end(end, depth_limit):
        """Extend path from one end with backtracking."""
        if depth_limit <= 0 or time.time() > deadline:
            return 0
        cur = path[0] if end == 'front' else path[-1]
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            return 0
        candidates.sort(key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        best_ext = 0
        for v in candidates:
            if time.time() > deadline:
                break
            if end == 'front':
                path.insert(0, v)
            else:
                path.append(v)
            seen.add(v)
            ext = 1 + extend_end(end, depth_limit - 1)
            if ext > best_ext:
                best_ext = ext
            # Undo
            if end == 'front':
                path.pop(0)
            else:
                path.pop()
            seen.remove(v)
        return best_ext
    
    # Try extending front
    front_ext = extend_end('front', max_depth)
    # Try extending back
    back_ext = extend_end('back', max_depth)
    # Re-apply the better extension
    if front_ext > 0:
        # Re-extend front greedily after backtracking found it possible
        extend_both_ends(path, adj)
    if back_ext > 0:
        extend_both_ends(path, adj)
    return path


def forced_move_path(start, adj, rng, mode):
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            break
        if len(candidates) == 1:
            v = candidates[0]
        elif mode == 'warns':
            v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        elif mode == 'max':
            v = max(candidates, key=lambda w: (len(adj[w]), -w))
        else:
            v = rng.choice(candidates)
        path.append(v)
        seen.add(v)
    return path


def refine(path, adj, deadline):
    path = extend_both_ends(path, adj)
    path = try_insert(path, adj)
    path = extend_both_ends(path, adj)
    # 2-opt refinement to find better endpoints
    path2 = two_opt_refine(path, adj, deadline)
    if len(path2) > len(path):
        path = path2
    path = extend_both_ends(path, adj)
    path = try_insert(path, adj)
    path = extend_both_ends(path, adj)
    return path


def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    comps = find_components()
    comps.sort(key=len, reverse=True)
    if not comps:
        return [0]
    
    largest_comp = comps[0]
    best_path = list(largest_comp[:1])
    best_len = len(best_path)
    
    # Phase 1: Greedy from every vertex
    for start in largest_comp:
        if time.time() > deadline - 30:
            break
        p = greedy_path_from(start, ADJ)
        p = refine(p, ADJ, deadline)
        if len(p) > best_len:
            best_path, best_len = p, len(p)
    
    # Phase 2: IDDFS from promising starts (high-degree vertices in largest comp)
    if time.time() < deadline - 10:
        # Sort by degree descending
        deg_sorted = sorted(largest_comp, key=lambda v: len(ADJ[v]), reverse=True)
        for start in deg_sorted[:20]:
            if time.time() > deadline - 10:
                break
            p = iddfs_path(start, ADJ, deadline - 10)
            p = refine(p, ADJ, deadline)
            if len(p) > best_len:
                best_path, best_len = p, len(p)
    
    # Phase 3: Forced-move randomized restarts
    modes = ['warns', 'max', 'rand']
    rng = random.Random(748)
    for _ in range(5000):
        if time.time() > deadline - 5:
            break
        start = rng.choice(largest_comp)
        mode = modes[rng.randrange(len(modes))]
        p = forced_move_path(start, ADJ, rng, mode)
        p = refine(p, ADJ, deadline)
        if len(p) > best_len:
            best_path, best_len = p, len(p)
    
    # Phase 4: Final refinement with 2-opt
    if best_path and time.time() < deadline - 1:
        extended = extend_both_ends(best_path, ADJ)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
        extended = try_insert(best_path, ADJ)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
        extended = extend_both_ends(best_path, ADJ)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
        # 2-opt pass
        extended = two_opt_refine(best_path, ADJ, deadline)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
        extended = extend_both_ends(best_path, ADJ)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
    
    # Validate
    if not best_path:
        best_path = [0]
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in ADJ[best_path[i]]:
            best_path = [0]
            break
    
    return best_path
