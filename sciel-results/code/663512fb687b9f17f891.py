"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit for search, 5s buffer
    
    N = 256
    max_sum = 2 * N
    
    # Precompute squares up to 2*N
    squares = set()
    i = 2
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # 1. Filter nodes that violate the diagonal condition (a=a => 2a is square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
            
    M = len(valid_nodes)
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # 2. Build conflict graph: edge between i,j if valid_nodes[i] + valid_nodes[j] is a square
    adj = [set() for _ in range(M)]
    for i in range(M):
        vi = valid_nodes[i]
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            if vi + vj in squares:
                adj[i].add(j)
                adj[j].add(i)
    
    # 3. Check if graph is bipartite and get 2-coloring
    color = [-1] * M
    is_bipartite = True
    for start in range(M):
        if color[start] == -1:
            color[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        is_bipartite = False
                        break
                if not is_bipartite:
                    break
        if not is_bipartite:
            break
    
    best_result = None
    
    if is_bipartite:
        # 4. Solve MWIS via min-cut (all weights = 1) using Dinic's algorithm
        INF = 10**9
        num_flow_nodes = M + 2
        source = 0
        sink = M + 1
        
        # Dinic's max flow implementation
        graph = [[] for _ in range(num_flow_nodes)]
        
        def add_edge(u, v, cap):
            graph[u].append([v, cap, len(graph[v])])
            graph[v].append([u, 0, len(graph[u]) - 1])
        
        # Add edges: source -> left (cap 1), right -> sink (cap 1), left -> right (cap INF)
        for i in range(M):
            if color[i] == 0:
                add_edge(source, i + 1, 1)
            else:
                add_edge(i + 1, sink, 1)
        
        for i in range(M):
            if color[i] == 0:
                for j in adj[i]:
                    if color[j] == 1:
                        add_edge(i + 1, j + 1, INF)
        
        # Dinic's algorithm
        def dinic_max_flow():
            flow = 0
            while time.time() < deadline - 3:
                # BFS to build level graph
                level = [-1] * num_flow_nodes
                level[source] = 0
                q = deque([source])
                while q:
                    u = q.popleft()
                    for edge in graph[u]:
                        v, cap, _ = edge
                        if cap > 0 and level[v] == -1:
                            level[v] = level[u] + 1
                            q.append(v)
                
                if level[sink] == -1:
                    break
                
                # DFS to find blocking flow
                iter_arr = [0] * num_flow_nodes
                
                def dfs(u, f):
                    if u == sink:
                        return f
                    for i in range(iter_arr[u], len(graph[u])):
                        iter_arr[u] = i
                        v, cap, rev = graph[u][i]
                        if cap > 0 and level[v] == level[u] + 1:
                            d = dfs(v, min(f, cap))
                            if d > 0:
                                graph[u][i][1] -= d
                                graph[v][rev][1] += d
                                return d
                    return 0
                
                while True:
                    f = dfs(source, INF)
                    if f == 0:
                        break
                    flow += f
            
            return flow
        
        max_flow = dinic_max_flow()
        
        # Find min vertex cover from residual graph
        reachable = [False] * num_flow_nodes
        reachable[source] = True
        q = deque([source])
        while q:
            u = q.popleft()
            for edge in graph[u]:
                v, cap, _ = edge
                if cap > 0 and not reachable[v]:
                    reachable[v] = True
                    q.append(v)
        
        # Independent set = (left \ S) ∪ (right ∩ S)
        independent_set = []
        for i in range(M):
            node_in_flow = i + 1
            if color[i] == 0:  # left
                if not reachable[node_in_flow]:
                    independent_set.append(valid_nodes[i])
            else:  # right
                if reachable[node_in_flow]:
                    independent_set.append(valid_nodes[i])
        
        if len(independent_set) > 0:
            best_result = independent_set
    
    # Fallback: greedy + local search if not bipartite or timeout
    if best_result is None or time.time() > deadline - 2:
        # Greedy: pick nodes with fewest conflicts first
        degrees = [len(adj[i]) for i in range(M)]
        order = sorted(range(M), key=lambda x: (degrees[x], x))
        
        result_set = set()
        for idx in order:
            if all(j not in result_set for j in adj[idx]):
                result_set.add(idx)
        
        # Local search: try adding nodes that don't conflict
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for idx in range(M):
                if idx not in result_set:
                    if all(j not in result_set for j in adj[idx]):
                        result_set.add(idx)
                        improved = True
        
        # Local search: try swaps (remove one, add another)
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for remove_idx in list(result_set):
                # Find what we could add if we remove this node
                for add_idx in range(M):
                    if add_idx not in result_set and add_idx != remove_idx:
                        # Check if add_idx conflicts with (result_set - {remove_idx})
                        if all(j not in result_set or j == remove_idx for j in adj[add_idx]):
                            # Net gain: remove 1, add 1 = 0, but might enable future adds
                            # Only swap if it doesn't reduce size
                            result_set.discard(remove_idx)
                            result_set.add(add_idx)
                            improved = True
                            break
                if improved:
                    break
        
        best_result = [valid_nodes[i] for i in sorted(result_set)]
    
    if not best_result:
        best_result = [1]
    
    return best_result
# EVOLVE-BLOCK-END
