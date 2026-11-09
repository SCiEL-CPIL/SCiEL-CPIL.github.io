import time
import math
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 256
    
    # 1. Filter nodes: 2*v must not be a perfect square
    valid_nodes = []
    for v in range(1, N + 1):
        s = 2 * v
        r = int(math.isqrt(s))
        if r * r != s:
            valid_nodes.append(v)
    
    if not valid_nodes:
        return [1]
        
    # Map values to indices for faster lookup
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    M = len(valid_nodes)
    
    # 2. Build adjacency list
    adj = [set() for _ in range(M)]
    
    for i in range(M):
        vi = valid_nodes[i]
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            s = vi + vj
            r = int(math.isqrt(s))
            if r * r == s:
                adj[i].add(j)
                adj[j].add(i)
                
    # 3. Check if graph is bipartite via BFS 2-coloring
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
    
    if is_bipartite:
        # Use max-flow to find exact Maximum Independent Set
        # Build flow network: source -> left nodes (weight 1), left -> right for conflicts, right -> sink (weight 1)
        # Max Independent Set = total nodes - Min Vertex Cover = total nodes - Max Flow
        
        left_nodes = [i for i in range(M) if color[i] == 0]
        right_nodes = [i for i in range(M) if color[i] == 1]
        
        # Build flow network using Dinic's algorithm
        # Node indexing in flow network:
        # 0: source
        # 1 to len(left): left nodes
        # len(left)+1 to len(left)+len(right): right nodes
        # len(left)+len(right)+1: sink
        
        n_left = len(left_nodes)
        n_right = len(right_nodes)
        total_flow_nodes = 2 + n_left + n_right
        source = 0
        sink = total_flow_nodes - 1
        
        # left_node_to_flow_idx: maps original index to flow network index
        left_flow_idx = {left_nodes[i]: i + 1 for i in range(n_left)}
        right_flow_idx = {right_nodes[i]: n_left + i + 1 for i in range(n_right)}
        
        # Build adjacency list for flow network with capacities
        # Using edge list representation for Dinic's
        # graph[u] = list of [v, capacity, reverse_edge_index]
        flow_graph = [[] for _ in range(total_flow_nodes)]
        
        def add_edge(u, v, cap):
            flow_graph[u].append([v, cap, len(flow_graph[v])])
            flow_graph[v].append([u, 0, len(flow_graph[u]) - 1])
        
        # Source to left nodes with capacity 1
        for i in range(n_left):
            add_edge(source, i + 1, 1)
        
        # Right nodes to sink with capacity 1
        for i in range(n_right):
            add_edge(n_left + i + 1, sink, 1)
        
        # Conflict edges: left to right with capacity INF
        INF = float('inf')
        for i in left_nodes:
            for j in adj[i]:
                if color[j] == 1:  # j is in right set
                    add_edge(left_flow_idx[i], right_flow_idx[j], INF)
        
        # Dinic's max flow
        def dinic_max_flow():
            max_flow = 0
            while True:
                # BFS to build level graph
                level = [-1] * total_flow_nodes
                level[source] = 0
                q = deque([source])
                while q:
                    u = q.popleft()
                    for e in flow_graph[u]:
                        v, cap, _ = e
                        if cap > 0 and level[v] == -1:
                            level[v] = level[u] + 1
                            q.append(v)
                
                if level[sink] == -1:
                    break
                
                # DFS to find blocking flow
                ptr = [0] * total_flow_nodes
                
                def dfs(u, pushed):
                    if pushed == 0:
                        return 0
                    if u == sink:
                        return pushed
                    
                    while ptr[u] < len(flow_graph[u]):
                        e = flow_graph[u][ptr[u]]
                        v, cap, rev_idx = e
                        if level[v] == level[u] + 1 and cap > 0:
                            tr = dfs(v, min(pushed, cap))
                            if tr > 0:
                                e[1] -= tr
                                flow_graph[v][rev_idx][1] += tr
                                return tr
                        ptr[u] += 1
                    
                    return 0
                
                while True:
                    pushed = dfs(source, float('inf'))
                    if pushed == 0:
                        break
                    max_flow += pushed
            
            return max_flow
        
        min_vertex_cover_size = dinic_max_flow()
        max_independent_set_size = M - min_vertex_cover_size
        
        # Find the actual independent set
        # After max flow, the min vertex cover consists of:
        # - Left nodes reachable from source in residual graph
        # - Right nodes NOT reachable from source in residual graph
        # Independent set = complement of min vertex cover
        
        # BFS from source in residual graph
        visited = [False] * total_flow_nodes
        visited[source] = True
        q = deque([source])
        while q:
            u = q.popleft()
            for e in flow_graph[u]:
                v, cap, _ = e
                if cap > 0 and not visited[v]:
                    visited[v] = True
                    q.append(v)
        
        # Min vertex cover:
        # Left nodes NOT visited (not reachable from source)
        # Right nodes visited (reachable from source)
        in_vertex_cover = set()
        for i in left_nodes:
            if not visited[left_flow_idx[i]]:
                in_vertex_cover.add(i)
        for i in right_nodes:
            if visited[right_flow_idx[i]]:
                in_vertex_cover.add(i)
        
        # Independent set is complement
        independent_set_indices = [i for i in range(M) if i not in in_vertex_cover]
        
        # Convert to values
        result = [valid_nodes[i] for i in independent_set_indices]
        
        if not result:
            result = [valid_nodes[0]]
        
        return result
    
    else:
        # Graph is not bipartite, fall back to greedy MIS with local search
        remaining = set(range(M))
        independent_set_indices = []
        
        while remaining:
            if time.time() > deadline:
                break
                
            # Find node with minimum degree in the remaining subgraph
            min_deg_node = None
            min_deg = float('inf')
            
            for u in remaining:
                deg = len(adj[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_deg_node = u
                    
            if min_deg_node is None:
                break
                
            # Add to independent set
            independent_set_indices.append(min_deg_node)
            
            # Remove node and its neighbors from remaining
            remaining.discard(min_deg_node)
            for v in list(adj[min_deg_node]):
                remaining.discard(v)
                
        # Local search: try to add nodes or swap
        result_set = set(independent_set_indices)
        
        # Try adding nodes
        for i in range(M):
            if i not in result_set:
                # Check if i conflicts with any node in result_set
                conflict = False
                for j in result_set:
                    if j in adj[i]:
                        conflict = True
                        break
                if not conflict:
                    result_set.add(i)
        
        # Try swaps: remove one, add another
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in list(result_set):
                for j in range(M):
                    if j not in result_set:
                        # Simulate: remove i, add j
                        new_set = result_set - {i} | {j}
                        # Check validity
                        valid = True
                        for a in new_set:
                            for b in new_set:
                                if a >= b:
                                    break
                                if b in adj[a]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid and len(new_set) > len(result_set):
                            result_set = new_set
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        result = [valid_nodes[i] for i in result_set]
        
        if not result:
            result = [valid_nodes[0]]
        
        return result
