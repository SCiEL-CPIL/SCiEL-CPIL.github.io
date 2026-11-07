import time
import random
import sys
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    DEADLINE = 85.0
    
    n = P['n']
    edges = P['edges']
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def time_up():
        return (time.time() - t_start) > DEADLINE

    # Min-Cost Max-Flow implementation
    # Nodes: 
    # 0: Source (Super Source)
    # 1: Sink (Super Sink)
    # 2 + v: in-node of v
    # 2 + n + v: out-node of v
    # Total nodes: 2 + 2*n
    
    def solve_mcmf(branches):
        """
        Solves the path packing problem for a fixed set of branches.
        Returns a list of paths if feasible, else None.
        Uses min-cost max-flow to find edge-disjoint (actually vertex-disjoint interiors) paths.
        """
        m = len(branches)
        if m < 2:
            return None
            
        branch_set = set(branches)
        
        # Node indexing
        def node_in(v):
            return 2 + v
        def node_out(v):
            return 2 + n + v
            
        N_nodes = 2 + 2 * n
        S = 0
        T = 1
        
        # Graph structure for MCMF
        # graph[u] = list of [v, capacity, cost, reverse_index]
        graph = [[] for _ in range(N_nodes)]
        
        def add_edge(u, v, cap, cost):
            graph[u].append([v, cap, cost, len(graph[v])])
            graph[v].append([u, 0, -cost, len(graph[u]) - 1])
            
        # 1. Connect S to branch in-nodes
        # We need to send flow out of each branch.
        # Total flow required: m * (m - 1) / 2
        # Each branch sends (m-1) units of flow.
        for a in branches:
            add_edge(S, node_in(a), m - 1, 0)
            
        # 2. Connect branch out-nodes to T
        for a in branches:
            add_edge(node_out(a), T, m - 1, 0)
            
        # 3. Vertex capacities
        # Non-branch vertices: capacity 1 (interior disjointness)
        # Branch vertices: capacity INF (can be endpoints of many paths)
        for v in range(n):
            if v in branch_set:
                add_edge(node_in(v), node_out(v), float('inf'), 0)
            else:
                add_edge(node_in(v), node_out(v), 1, 0)
                
        # 4. Graph edges
        # For every edge (u, v) in the original graph, add flow edges
        # out(u) -> in(v) and out(v) -> in(u)
        # Cost 1 for each step to minimize path lengths (helps find simpler paths)
        for u, v in edges:
            add_edge(node_out(u), node_in(v), 1, 1)
            add_edge(node_out(v), node_in(u), 1, 1)
            
        # Run MCMF
        # We want to maximize flow, but costs are positive. 
        # Standard MCMF finds min cost for a given flow amount.
        # We know the exact flow amount required: F = m * (m - 1) // 2
        # If we can push F units with finite cost, it's feasible.
        
        # Bellman-Ford / SPFA for initial potentials (since costs are non-negative, we can start with 0 or just use Dijkstra)
        # Since all costs are >= 0, Dijkstra is safe from the start.
        
        INF = float('inf')
        required_flow = m * (m - 1) // 2
        total_cost = 0
        flow = 0
        
        # dist, prev_node, prev_edge
        dist = [INF] * N_nodes
        prev_node = [-1] * N_nodes
        prev_edge = [-1] * N_nodes
        
        # We will repeatedly find shortest paths from S to T in the residual graph
        # and augment flow.
        
        # Optimization: Since capacities are small integers, we can augment 1 unit at a time or more.
        # Given n=24, max flow is small (e.g., 8 branches -> 28 flow units).
        
        while flow < required_flow:
            # Dijkstra's algorithm
            dist[S] = 0
            prev_node = [-1] * N_nodes
            prev_edge = [-1] * N_nodes
            
            # Simple O(V^2) Dijkstra or Heapq. Heapq is better.
            import heapq
            pq = [(0, S)]
            visited = [False] * N_nodes
            
            while pq:
                d, u = heapq.heappop(pq)
                if visited[u]:
                    continue
                visited[u] = True
                if u == T:
                    break
                for i, e in enumerate(graph[u]):
                    v, cap, cost, rev = e
                    if cap > 0 and not visited[v]:
                        nd = d + cost
                        if nd < dist[v]:
                            dist[v] = nd
                            prev_node[v] = u
                            prev_edge[v] = i
                            heapq.heappush(pq, (nd, v))
                            
            if dist[T] == INF:
                # No more path
                return None
                
            # Augment flow
            # Find max flow along the path
            path_flow = INF
            v = T
            while v != S:
                u = prev_node[v]
                i = prev_edge[v]
                path_flow = min(path_flow, graph[u][i][1])
                v = u
                
            # If path_flow is 0, something is wrong, but Dijkstra only finds paths with cap > 0
            if path_flow == 0:
                return None
                
            # Update residual graph
            v = T
            while v != S:
                u = prev_node[v]
                i = prev_edge[v]
                graph[u][i][1] -= path_flow
                rev = graph[u][i][3]
                graph[v][rev][1] += path_flow
                v = u
                
            flow += path_flow
            total_cost += path_flow * dist[T]
            
        # If we reached here, we have a feasible flow of required amount.
        # Now we need to extract the paths.
        
        # The flow is distributed across the edges.
        # We need to trace back from S to T to find the individual paths.
        # Since S connects to branch_in, and branch_out connects to T,
        # and each branch has capacity m-1, we can decompose the flow.
        
        # A simpler way: 
        # For each pair (a, b) in branches, we know there is 1 unit of flow from a to b.
        # However, the flow network doesn't distinguish "which" branch the flow is going to.
        # We need to reconstruct the specific paths for each pair.
        
        # Strategy for reconstruction:
        # 1. Identify the edges carrying flow.
        # 2. Since interiors are vertex-disjoint, the paths are edge-disjoint too.
        # 3. We can try to match flows to pairs.
        
        # Let's collect all edges with flow > 0 in the original direction (u->v where u < v in index or just check capacity reduction).
        # Actually, easier: 
        # We know the set of branches.
        # For each branch 'a', there are m-1 units of flow leaving 'a' (via node_out(a) -> T is the sink side, but flow enters S->node_in(a)).
        # Wait, flow goes S -> node_in(a) -> ... -> node_out(b) -> T.
        # So flow enters branches at 'in' and leaves at 'out'.
        
        # Let's trace paths by following the flow.
        # We can do this by repeatedly finding a path from S to T in the residual graph of the *flow* (not residual capacities, but the actual flow).
        # Actually, since we just pushed flow, we can look at the edges that have flow.
        
        # Better approach for extraction:
        # The flow decomposition can be done by looking at the edges with positive flow.
        # However, MCMF doesn't store the final flow matrix easily in this implementation.
        
        # Let's re-extract from the residual graph changes.
        # Original capacity was C. Current capacity is C - flow.
        # So flow = C - current_cap.
        
        # We need to reconstruct the paths.
        # Since the graph is small, we can use a simple DFS on the "flow graph" (edges with flow > 0).
        
        # Build flow graph
        flow_graph = [[] for _ in range(N_nodes)]
        for u in range(N_nodes):
            for e in graph[u]:
                v, cap, cost, rev = e
                # We need to know original capacity to determine flow.
                # This implementation overwrites capacities. 
                # To make extraction robust, we should have stored original capacities or calculated flow during augmentation.
                
        # Let's modify the augmentation step to record the paths directly.
        # If we augment 1 unit at a time (which is likely given small capacities), we can just store the path found by Dijkstra.
        
        # Let's re-run the solver but store the paths found during augmentation.
        
    def solve_mcmf_with_paths(branches):
        m = len(branches)
        if m < 2:
            return None
            
        branch_set = set(branches)
        
        def node_in(v):
            return 2 + v
        def node_out(v):
            return 2 + n + v
            
        N_nodes = 2 + 2 * n
        S = 0
        T = 1
        
        graph = [[] for _ in range(N_nodes)]
        # Store original capacities for extraction
        orig_cap = {}
        
        def add_edge(u, v, cap, cost):
            idx = len(graph[u])
            graph[u].append([v, cap, cost, len(graph[v])])
            graph[v].append([u, 0, -cost, len(graph[u]) - 1])
            orig_cap[(u, idx)] = cap
            
        for a in branches:
            add_edge(S, node_in(a), m - 1, 0)
            
        for a in branches:
            add_edge(node_out(a), T, m - 1, 0)
            
        for v in range(n):
            if v in branch_set:
                add_edge(node_in(v), node_out(v), float('inf'), 0)
            else:
                add_edge(node_in(v), node_out(v), 1, 0)
                
        for u, v in edges:
            add_edge(node_out(u), node_in(v), 1, 1)
            add_edge(node_out(v), node_in(u), 1, 1)
            
        INF = float('inf')
        required_flow = m * (m - 1) // 2
        flow = 0
        
        import heapq
        found_paths = []
        
        while flow < required_flow:
            dist = [INF] * N_nodes
            prev_node = [-1] * N_nodes
            prev_edge = [-1] * N_nodes
            dist[S] = 0
            
            pq = [(0, S)]
            visited = [False] * N_nodes
            
            while pq:
                d, u = heapq.heappop(pq)
                if visited[u]:
                    continue
                visited[u] = True
                if u == T:
                    break
                for i, e in enumerate(graph[u]):
                    v, cap, cost, rev = e
                    if cap > 0 and not visited[v]:
                        nd = d + cost
                        if nd < dist[v]:
                            dist[v] = nd
                            prev_node[v] = u
                            prev_edge[v] = i
                            heapq.heappush(pq, (nd, v))
                            
            if dist[T] == INF:
                return None
                
            # Extract path
            path_nodes = []
            v = T
            while v != S:
                path_nodes.append(v)
                v = prev_node[v]
            path_nodes.append(S)
            path_nodes.reverse()
            
            # Map node IDs back to graph vertices
            # S=0, T=1
            # 2..2+n-1 are in(v)
            # 2+n..2+2n-1 are out(v)
            
            # The path is S -> in(a) -> ... -> out(b) -> T
            # We need to extract the sequence of original vertices.
            # in(v) -> v
            # out(v) -> v
            
            path_verts = []
            for node_id in path_nodes:
                if node_id == S or node_id == T:
                    continue
                if node_id < 2 + n:
                    v_orig = node_id - 2
                    # This is an in-node
                    # If path_verts is empty or last was different, add it
                    # Actually, we traverse in->out->in->out...
                    # We want to list vertices.
                    # If we see in(v), it's the entry to v.
                    # If we see out(v), it's the exit from v.
                    # The sequence in graph is:
                    # S -> in(a) -> out(a) -> in(x) -> out(x) -> ... -> out(b) -> T
                    # So we can just take the vertex from the in-node or out-node.
                    # To avoid duplicates, we can just append v_orig if it's not the same as the last one.
                    if not path_verts or path_verts[-1] != v_orig:
                        path_verts.append(v_orig)
                else:
                    v_orig = node_id - (2 + n)
                    if not path_verts or path_verts[-1] != v_orig:
                        path_verts.append(v_orig)
                    
            # The path should start with a branch and end with a branch.
            # path_verts[0] should be a branch, path_verts[-1] should be a branch.
            # If path_verts has length 1, it means a direct edge a->b where a==b? No, simple path.
            # If a and b are adjacent, path is [a, b].
            
            if len(path_verts) < 2:
                return None
                
            # Check endpoints
            start_v = path_verts[0]
            end_v = path_verts[-1]
            
            if start_v not in branch_set or end_v not in branch_set:
                return None
                
            # Augment flow by 1 (since we found a simple path, and capacities are 1 on edges/vertices, we can only push 1)
            # Actually, Dijkstra finds shortest path. Capacities might allow >1, but vertex capacities are 1 for non-branches.
            # Branches have INF capacity, but edges out of branches have capacity 1 (from graph edges).
            # So max flow on any single path is 1.
            
            v = T
            while v != S:
                u = prev_node[v]
                i = prev_edge[v]
                graph[u][i][1] -= 1
                rev = graph[u][i][3]
                graph[v][rev][1] += 1
                v = u
                
            flow += 1
            found_paths.append(path_verts)
            
        # We have m*(m-1)/2 paths.
        # We need to organize them into pairs.
        # Each branch 'a' appears as a start in (m-1) paths and as an end in (m-1) paths.
        
        # Construct the output paths list.
        # The problem requires "one simple path for each unordered pair of branches".
        # So we need to return a list of paths.
        
        return found_paths

    def find_max_clique():
        best = []
        candidates = list(range(n))
        candidates.sort(key=lambda x: -len(adj[x]))

        def expand(clique, cand):
            nonlocal best
            if len(clique) > len(best):
                best = list(clique)
            if not cand:
                return
            for i, v in enumerate(cand):
                new_clique = clique + [v]
                new_cand = [u for u in cand[i + 1:] if u in adj[v]]
                expand(new_clique, new_cand)

        expand([], candidates)
        return best

    clique = find_max_clique()

    best_branches = None
    best_paths = None
    best_count = 0

    def update_best(branches, paths):
        nonlocal best_branches, best_paths, best_count
        if paths is not None and len(branches) > best_count:
            best_branches = list(branches)
            best_paths = list(paths)
            best_count = len(branches)

    # Try the clique
    if len(clique) >= 2:
        paths = solve_mcmf_with_paths(clique)
        if paths is not None:
            update_best(clique, paths)

    # Greedy expansion
    current = list(clique)
    changed = True
    while changed and not time_up():
        changed = False
        for v in range(n):
            if v in current:
                continue
            trial = current + [v]
            paths = solve_mcmf_with_paths(trial)
            if paths is not None:
                current = trial
                changed = True
                update_best(current, paths)
                break
        if time_up():
            break

    # Local improvement: try swapping one branch for another
    if best_branches is not None and not time_up():
        improved = True
        while improved and not time_up():
            improved = False
            cur_b = list(best_branches)
            cur_set = set(cur_b)
            for idx in range(len(cur_b)):
                if time_up():
                    break
                for v in range(n):
                    if v in cur_set:
                        continue
                    new_b = cur_b[:idx] + cur_b[idx + 1:] + [v]
                    new_b.sort()
                    paths = solve_mcmf_with_paths(new_b)
                    if paths is not None and len(new_b) > best_count:
                        update_best(new_b, paths)
                        improved = True
                        break
                if improved:
                    break

    # Random restarts
    if not time_up():
        for restart in range(100):
            if time_up():
                break
            seed_size = min(n, max(2, best_count + random.randint(-1, 3)))
            seed = random.sample(range(n), seed_size)
            current = list(seed)
            paths = solve_mcmf_with_paths(current)
            if paths is not None:
                update_best(current, paths)
            changed = True
            while changed and not time_up():
                changed = False
                for v in range(n):
                    if v in current:
                        continue
                    trial = current + [v]
                    paths = solve_mcmf_with_paths(trial)
                    if paths is not None:
                        current = trial
                        changed = True
                        update_best(current, paths)
                        break
                if time_up():
                    break

    # Fallback
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]

    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(x) for x in p] for p in best_paths]

    return {'branches': branches_out, 'paths': paths_out}
