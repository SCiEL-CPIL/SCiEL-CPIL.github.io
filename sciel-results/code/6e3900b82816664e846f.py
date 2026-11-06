"""Long-path search on the pinned G(100,1.8/100) graph (seed 748)."""
import time
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)
DEADLINE = time.time() + 78.0  # safety margin under the 90s wall-clock kill


def _components():
    seen = set()
    comps = []
    for s in range(N):
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        comp = []
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in ADJ[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def _bipartite_slack(comp):
    """
    Compute a 'slack' score for each vertex in the component using a bipartite
    matching relaxation. Vertices with low slack (few unmatched neighbors) are
    bottlenecks that should be visited early in a path to avoid trapping.
    
    We build a bipartite graph: left = vertices, right = vertices.
    Edge (u_left, v_right) exists if u-v is an edge in G.
    Maximum matching size M tells us how many vertices can be 'covered' as internal
    path vertices. Vertices not in a maximum matching have high slack.
    
    We use Hopcroft-Karp for efficiency.
    """
    n = len(comp)
    if n == 0:
        return {}
    idx = {v: i for i, v in enumerate(comp)}
    # Build adjacency in indexed form
    adj_idx = [[] for _ in range(n)]
    for u in comp:
        for v in ADJ[u]:
            if v in idx:
                adj_idx[idx[u]].append(idx[v])
    
    # Hopcroft-Karp
    pair_u = [-1] * n  # match for left node
    pair_v = [-1] * n  # match for right node
    dist = [0] * n
    
    def bfs():
        q = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        found = False
        while q:
            u = q.popleft()
            for v in adj_idx[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found
    
    def dfs(u):
        for v in adj_idx[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False
    
    matching = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    
    # Slack: number of neighbors NOT matched to this vertex's match partner
    # A vertex with low slack is a bottleneck
    slack = {}
    for i, v in enumerate(comp):
        matched_partner = pair_u[i]
        if matched_partner == -1:
            # Unmatched vertex - high slack (flexible)
            slack[v] = len(adj_idx[i])
        else:
            # Count neighbors that are NOT the matched partner's neighbors
            # Low slack means this vertex constrains the path
            slack[v] = len(adj_idx[i]) - 1  # simple heuristic
    return slack


def _dfs_longest(comp, slack, deadline):
    """
    Iterative deepening DFS with vertex ordering guided by slack scores.
    Vertices with lower slack are visited earlier to avoid trapping.
    """
    n = len(comp)
    if n == 0:
        return []
    cset = set(comp)
    comp_set = cset
    
    # Precompute neighbor lists within component, sorted by slack
    adj_in_comp = {}
    for v in comp:
        nbrs = [u for u in ADJ[v] if u in comp_set]
        nbrs.sort(key=lambda u: (slack.get(u, 999), u))
        adj_in_comp[v] = nbrs
    
    best_path = []
    best_len = 0
    nodes_visited = [0]
    
    # Try different starting vertices
    # Start with low-slack vertices (bottlenecks) as they're more constrained
    starts = sorted(comp, key=lambda v: (slack.get(v, 999), v))
    
    for start in starts:
        if time.time() >= deadline:
            break
        
        path = [start]
        used = {start}
        
        def dfs():
            nonlocal best_len, best_path
            nodes_visited[0] += 1
            if nodes_visited[0] % 4096 == 0 and time.time() >= deadline:
                return False
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            
            cur = path[-1]
            for nxt in adj_in_comp[cur]:
                if nxt in used:
                    continue
                path.append(nxt)
                used.add(nxt)
                
                # Pruning: if remaining unvisited vertices can't beat best, stop
                remaining = n - len(path)
                if len(path) + remaining <= best_len:
                    path.pop()
                    used.discard(nxt)
                    continue
                
                dfs()
                
                path.pop()
                used.discard(nxt)
                
                if time.time() >= deadline:
                    return False
            return True
        
        dfs()
        if time.time() >= deadline:
            break
    
    return best_path


def _try_insert(path, deadline):
    """Insert unused vertices into path where possible."""
    p = list(path)
    used = set(p)
    unused = [v for v in range(N) if v not in used]
    improved = True
    steps = 0
    while improved and steps < 30 and time.time() < deadline:
        improved = False
        steps += 1
        for i in range(len(p) - 1):
            a, b = p[i], p[i + 1]
            for v in unused:
                if v in ADJ[a] and v in ADJ[b] and v not in used:
                    p.insert(i + 1, v)
                    used.add(v)
                    unused.remove(v)
                    improved = True
                    break
            if improved:
                break
    return p


def _local_search(path, deadline):
    """
    Local search: try to extend path from both ends, and try 2-opt-like swaps.
    """
    p = list(path)
    used = set(p)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try extending from both ends
        for end_idx in (0, -1):
            cur = p[end_idx]
            cand = [v for v in ADJ[cur] if v not in used]
            if cand:
                # Pick candidate that allows most further extension
                cand.sort(key=lambda v: -len(ADJ[v] - used))
                nxt = cand[0]
                if end_idx == 0:
                    p.insert(0, nxt)
                else:
                    p.append(nxt)
                used.add(nxt)
                improved = True
                break
        if not improved:
            # Try inserting unused vertices
            ins = _try_insert(p, deadline)
            if len(ins) > len(p):
                p = ins
                used = set(p)
                improved = True
    return p


def construct():
    t0 = time.time()
    comps = _components()
    best = []
    
    for comp in comps:
        if time.time() >= DEADLINE:
            break
        if len(comp) <= len(best):
            continue
        
        # Compute slack scores via bipartite matching
        slack = _bipartite_slack(comp)
        
        # DFS guided by slack ordering
        path = _dfs_longest(comp, slack, DEADLINE)
        if len(path) > len(best):
            best = list(path)
        
        # Local search to extend
        if best and time.time() < DEADLINE:
            ls = _local_search(best, DEADLINE)
            if len(ls) > len(best):
                best = list(ls)
        
        # Final insertion pass
        if best and time.time() < DEADLINE:
            ins = _try_insert(best, DEADLINE)
            if len(ins) > len(best):
                best = list(ins)
    
    if not best:
        best = [comps[0][0]] if comps else [0]
    return best
