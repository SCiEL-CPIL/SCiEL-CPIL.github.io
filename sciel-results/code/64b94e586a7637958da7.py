"""New program: Greedy Vertex Selection with Max-Flow Disjoint Path Routing."""
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

_N = P['n']
_ADJ = [0] * _N
for _e in P['edges']:
    _ADJ[_e[0]] |= 1 << _e[1]
    _ADJ[_e[1]] |= 1 << _e[0]


def _build_flow_graph(branches):
    n_nodes = 2 + 2 * _N
    graph = [[] for _ in range(n_nodes)]
    
    def add_edge(u, v, cap):
        graph[u].append([v, cap, len(graph[v])])
        graph[v].append([u, 0, len(graph[u]) - 1])

    for v in range(_N):
        vin = 2 + 2 * v
        vout = 2 + 2 * v + 1
        if v in branches:
            add_edge(vin, vout, 100)
        else:
            add_edge(vin, vout, 1)
        
        for nb in range(_N):
            if (1 << nb) & _ADJ[v]:
                add_edge(vout, 2 + 2 * nb, 1)
    
    src = 0
    sink = 1
    for v in branches:
        vin = 2 + 2 * v
        vout = 2 + 2 * v + 1
        add_edge(src, vin, 100)
        add_edge(vout, sink, 100)
        
    return graph, src, sink


def _max_flow(graph, src, sink):
    n = len(graph)
    flow = 0
    while True:
        parent = [-1] * n
        parent_edge = [-1] * n
        parent[src] = -2
        q = deque([src])
        while q:
            v = q.popleft()
            for i, e in enumerate(graph[v]):
                to, cap, rev = e
                if cap > 0 and parent[to] == -1:
                    parent[to] = v
                    parent_edge[to] = i
                    if to == sink:
                        break
                    q.append(to)
            if parent[sink] != -1:
                break
        
        if parent[sink] == -1:
            break
            
        min_cap = 10**9
        v = sink
        while v != src:
            u = parent[v]
            i = parent_edge[v]
            min_cap = min(min_cap, graph[u][i][1])
            v = u
            
        v = sink
        while v != src:
            u = parent[v]
            i = parent_edge[v]
            graph[u][i][1] -= min_cap
            rev = graph[u][i][2]
            graph[v][rev][1] += min_cap
            v = u
            
        flow += min_cap
    return flow


def _can_route(branches):
    if len(branches) < 2:
        return True
    graph, src, sink = _build_flow_graph(branches)
    flow = _max_flow(graph, src, sink)
    required = len(branches) * (len(branches) - 1) // 2
    return flow >= required


def _route_paths(branches):
    """
    Deterministic backtracking router.
    """
    pairs = [(i, j) for i in range(len(branches)) for j in range(i + 1, len(branches))]
    
    def solve(idx, used, paths):
        if idx == len(pairs):
            return paths
        i, j = pairs[idx]
        a, b = branches[i], branches[j]
        
        # BFS for path
        path = _bfs_path(a, b, used)
        if path:
            interiors = set(path[1:-1])
            if not (interiors & used):
                res = solve(idx + 1, used | interiors, paths + [path])
                if res:
                    return res
        
        # If BFS fails, try backtracking?
        # For small graphs, we can try to force a different path by blocking vertices.
        # But let's first try to reorder pairs if we are at the top level.
        return None

    def _bfs_path(src, dst, blocked):
        if src == dst:
            return [src]
        if (1 << dst) & _ADJ[src]:
            return [src, dst]
        parent = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            for nb in range(_N):
                if (1 << nb) & _ADJ[u] and nb not in parent:
                    if nb != dst and nb in blocked:
                        continue
                    parent[nb] = u
                    if nb == dst:
                        path = []
                        cur = dst
                        while cur is not None:
                            path.append(cur)
                            cur = parent[cur]
                        path.reverse()
                        return path
                    q.append(nb)
        return None

    # Try random orderings of pairs
    for _ in range(100):
        random.shuffle(pairs)
        res = solve(0, set(), [])
        if res:
            return res
    return None


def construct():
    order = sorted(range(_N), key=lambda x: bin(_ADJ[x]).count('1'), reverse=True)
    
    start_u = None
    start_v = None
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            u = order[i]
            v = order[j]
            if (1 << v) & _ADJ[u]:
                start_u = u
                start_v = v
                break
        if start_u is not None:
            break
            
    if start_u is None:
        u, v = P['edges'][0]
        start_u, start_v = u, v
        
    branches = [start_u, start_v]
    
    for v in order:
        if v in branches:
            continue
        test_branches = branches + [v]
        if _can_route(test_branches):
            branches = test_branches
            
    if len(branches) < 2:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        
    paths = _route_paths(branches)
    
    if paths is None:
        # Fallback: remove last vertex
        while len(branches) > 2:
            branches.pop()
            paths = _route_paths(branches)
            if paths:
                break
        
        if paths is None:
            u, v = P['edges'][0]
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    out_paths = [[int(x) for x in p] for p in paths]
    return {'branches': [int(b) for b in branches], 'paths': out_paths}
