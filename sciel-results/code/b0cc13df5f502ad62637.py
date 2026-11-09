import time
import math
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for safety
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    n = len(valid_nodes)
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build conflict adjacency as bitmasks
    conflicts = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if valid_nodes[i] + valid_nodes[j] in squares:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    # Precompute degrees
    degrees = [bin(conflicts[i]).count('1') for i in range(n)]
    
    # Check if graph is bipartite via BFS 2-coloring
    color = [-1] * n
    is_bipartite = True
    for start in range(n):
        if color[start] == -1:
            color[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                mask = conflicts[u]
                while mask:
                    lsb = mask & (-mask)
                    v = lsb.bit_length() - 1
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        is_bipartite = False
                        break
                    mask ^= lsb
                if not is_bipartite:
                    break
        if not is_bipartite:
            break
    
    if is_bipartite:
        # Use max-flow to find exact Maximum Independent Set
        left_nodes = [i for i in range(n) if color[i] == 0]
        right_nodes = [i for i in range(n) if color[i] == 1]
        
        n_left = len(left_nodes)
        n_right = len(right_nodes)
        total_flow_nodes = 2 + n_left + n_right
        source = 0
        sink = total_flow_nodes - 1
        
        left_flow_idx = {left_nodes[i]: i + 1 for i in range(n_left)}
        right_flow_idx = {right_nodes[i]: n_left + i + 1 for i in range(n_right)}
        
        flow_graph = [[] for _ in range(total_flow_nodes)]
        
        def add_edge(u, v, cap):
            flow_graph[u].append([v, cap, len(flow_graph[v])])
            flow_graph[v].append([u, 0, len(flow_graph[u]) - 1])
        
        for i in range(n_left):
            add_edge(source, i + 1, 1)
        
        for i in range(n_right):
            add_edge(n_left + i + 1, sink, 1)
        
        INF = float('inf')
        for i in left_nodes:
            mask = conflicts[i]
            while mask:
                lsb = mask & (-mask)
                j = lsb.bit_length() - 1
                if color[j] == 1:
                    add_edge(left_flow_idx[i], right_flow_idx[j], INF)
                mask ^= lsb
        
        def dinic_max_flow():
            max_flow = 0
            while True:
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
        
        in_vertex_cover = set()
        for i in left_nodes:
            if not visited[left_flow_idx[i]]:
                in_vertex_cover.add(i)
        for i in right_nodes:
            if visited[right_flow_idx[i]]:
                in_vertex_cover.add(i)
        
        independent_set_indices = [i for i in range(n) if i not in in_vertex_cover]
        
        result = [valid_nodes[i] for i in independent_set_indices]
        
        if not result:
            result = [valid_nodes[0]]
        
        return result
    
    else:
        # Graph is not bipartite: use exact Branch-and-Bound with Tomita-style coloring
        # for upper bounds to aggressively prune the search tree
        
        # Upper bound using greedy coloring (Tomita-style)
        def color_bound(vertices_mask):
            """Return an upper bound on the size of MIS in the induced subgraph of vertices_mask."""
            if vertices_mask == 0:
                return 0
            remaining = vertices_mask
            color_count = 0
            while remaining:
                color_count += 1
                # Pick a vertex from remaining (smallest index for determinism)
                lsb = remaining & (-remaining)
                v = lsb.bit_length() - 1
                # Remove v and all its neighbors from remaining
                remaining &= ~(conflicts[v] | (1 << v))
            return color_count
        
        # Greedy MIS for initial lower bound
        def greedy_mis():
            available = (1 << n) - 1
            chosen = 0
            while available:
                # Pick vertex with lowest degree in remaining subgraph
                best_v = -1
                best_deg = float('inf')
                temp = available
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    deg = bin(conflicts[v] & available).count('1')
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                    temp ^= lsb
                chosen |= (1 << best_v)
                available &= ~(1 << best_v | conflicts[best_v])
            return chosen
        
        greedy_result = greedy_mis()
        best_size = bin(greedy_result).count('1')
        best_mask = greedy_result
        
        # Branch and bound for Maximum Independent Set
        # We'll use an iterative approach with explicit stack to avoid recursion depth issues
        # and to have better control over time limits
        
        all_mask = (1 << n) - 1
        
        # Stack entries: (available_mask, current_size, chosen_mask, branch_state)
        # branch_state: 0 = not started, 1 = included vertex, 2 = excluded vertex
        stack = [(all_mask, 0, 0, 0)]
        
        # For better pruning, we'll process nodes in a specific order
        # We'll use a priority queue-like approach where we explore more promising branches first
        
        while stack and time.time() < deadline:
            available_mask, current_size, chosen_mask, branch_state = stack.pop()
            
            # Check time
            if time.time() > deadline:
                break
            
            # Prune: even if we add all remaining vertices, we can't beat best
            remaining_count = bin(available_mask).count('1')
            if current_size + remaining_count <= best_size:
                continue
            
            if available_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_mask = chosen_mask
                continue
            
            # Color-based upper bound
            ub = color_bound(available_mask)
            if current_size + ub <= best_size:
                continue
            
            # Pick a vertex to branch on
            # Heuristic: pick vertex with highest degree in the remaining subgraph
            best_v = -1
            best_deg = -1
            temp_mask = available_mask
            while temp_mask:
                lsb = temp_mask & (-temp_mask)
                v = lsb.bit_length() - 1
                deg_in_avail = bin(conflicts[v] & available_mask).count('1')
                if deg_in_avail > best_deg:
                    best_deg = deg_in_avail
                    best_v = v
                temp_mask ^= lsb
            
            if best_v == -1:
                continue
            
            v_bit = 1 << best_v
            
            if branch_state == 0:
                # First time seeing this node: push exclude branch first (LIFO, so include is explored first)
                # Actually, let's push exclude first so include is explored first (LIFO)
                new_available2 = available_mask & ~v_bit
                stack.append((new_available2, current_size, chosen_mask, 0))
                
                # Include best_v
                new_available = available_mask & ~(v_bit | conflicts[best_v])
                stack.append((new_available, current_size + 1, chosen_mask | v_bit, 0))
            
            elif branch_state == 1:
                # Already included, now exclude
                new_available2 = available_mask & ~v_bit
                stack.append((new_available2, current_size, chosen_mask, 0))
        
        # Extract the result
        result_nodes = []
        for i in range(n):
            if best_mask & (1 << i):
                result_nodes.append(valid_nodes[i])
        
        if not result_nodes:
            result_nodes = [valid_nodes[0]]
        
        return result_nodes
