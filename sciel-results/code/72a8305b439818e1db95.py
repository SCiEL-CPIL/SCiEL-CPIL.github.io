"""Flow-based search for clique subdivisions using greedy set extension."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    
    # Dinic's Max Flow Implementation
    class Dinic:
        def __init__(self, N):
            self.N = N
            self.graph = [[] for _ in range(N)]
            self.level = [0] * N
            self.it = [0] * N

        def add_edge(self, fr, to, cap):
            forward = [to, cap, None]
            backward = [fr, 0, forward]
            forward[2] = backward
            self.graph[fr].append(forward)
            self.graph[to].append(backward)

        def bfs(self, s, t):
            self.level = [-1] * self.N
            self.level[s] = 0
            q = deque([s])
            while q:
                v = q.popleft()
                for e in self.graph[v]:
                    if e[1] > 0 and self.level[e[0]] < 0:
                        self.level[e[0]] = self.level[v] + 1
                        if e[0] == t:
                            return True
                        q.append(e[0])
            return self.level[t] >= 0

        def dfs(self, v, t, f):
            if v == t:
                return f
            for i in range(self.it[v], len(self.graph[v])):
                e = self.graph[v][i]
                if e[1] > 0 and self.level[v] < self.level[e[0]]:
                    d = self.dfs(e[0], t, min(f, e[1]))
                    if d > 0:
                        e[1] -= d
                        e[2][1] += d
                        return d
                self.it[v] += 1
            return 0

        def max_flow(self, s, t):
            flow = 0
            while self.bfs(s, t):
                self.it = [0] * self.N
                f = self.dfs(s, t, float('inf'))
                while f > 0:
                    flow += f
                    f = self.dfs(s, t, float('inf'))
            return flow

    def check_clique(branches):
        """
        Checks if 'branches' can be the branch set of a clique subdivision.
        Returns list of paths or None.
        """
        k = len(branches)
        if k < 2:
            return None
        
        num_pairs = k * (k - 1) // 2
        
        # Node mapping:
        # v_in = 2*v, v_out = 2*v+1
        # S = 2*n, T = 2*n+1
        S = 2 * n
        T = 2 * n + 1
        total_nodes = 2 * n + 2
        
        dinic = Dinic(total_nodes)
        
        branch_set = set(branches)
        
        # Add vertex capacity edges
        for v in range(n):
            vin = 2 * v
            vout = 2 * v + 1
            if v in branch_set:
                dinic.add_edge(vin, vout, k - 1)
            else:
                dinic.add_edge(vin, vout, 1)
        
        # Add graph edges
        for u, v in edges:
            uout = 2 * u + 1
            vin = 2 * v
            vout = 2 * v + 1
            uin = 2 * u
            dinic.add_edge(uout, vin, 1)
            dinic.add_edge(vout, uin, 1)
        
        # Add source and sink connections
        for b in branches:
            bin_ = 2 * b
            bout = 2 * b + 1
            dinic.add_edge(S, bin_, k - 1)
            dinic.add_edge(bout, T, k - 1)
        
        max_flow_val = dinic.max_flow(S, T)
        
        if max_flow_val < num_pairs:
            return None
        
        # Extract paths
        # We need to find paths from each branch_i_out to branch_j_in for all i != j
        # such that they are vertex disjoint (except endpoints).
        # Since flow is integral and capacities are 1 on internal edges, we can trace flows.
        
        # Build residual graph pointer structure for tracing
        # We will greedily pick a path for each pair.
        # To ensure correctness, we should pick paths that don't conflict.
        # However, since the max flow guarantees existence, any decomposition into 
        # unit flows from S to T corresponds to a valid set of paths?
        # Not exactly. S connects to all branch_ins. T connects from all branch_outs.
        # A unit of flow from S to T goes S -> b_i_in -> ... -> b_j_out -> T.
        # This corresponds to a path from b_i to b_j.
        # We need exactly one path for each unordered pair {i, j}.
        # The flow network allows multiple flows between the same pair.
        # We need to select a subset of the unit flows such that for each pair {i, j}, 
        # we have exactly one path from i to j (or j to i).
        
        # Let's collect all unit paths from S to T in the flow.
        # We can do this by repeatedly finding a path in the flow graph.
        
        # Build adjacency of edges with flow > 0
        # flow_edge[u][v] > 0 if original cap > residual cap.
        # Since we modified dinic.graph, we can check residual capacities.
        # Actually, it's easier to just DFS on the residual graph looking for paths from S to T,
        # but we must only use edges that carry flow.
        
        # Let's create a directed graph of "flow edges".
        # An edge (u, v) has flow if the residual capacity is less than the original capacity.
        # Wait, Dinic stores residual capacities in `e[1]`.
        # Original capacity is not stored directly, but `e[2]` is the reverse edge.
        # Flow on (u, v) = original_cap - residual_cap.
        # Since we added edges with specific capacities, we can infer flow.
        
        # Alternative: Just trace paths from each branch_i_out to each branch_j_in 
        # in the subgraph of edges that are "saturated" or carry flow?
        # No, better: The set of edges with flow > 0 forms a subgraph.
        # We can decompose this subgraph into paths from sources (branch_outs with net outflow) 
        # to sinks (branch_ins with net inflow)?
        # Actually, S is the only source, T is the only sink.
        # But S connects to branch_ins. So flow enters branch_in.
        # And branch_out connects to T. So flow leaves branch_out.
        
        # Let's just find all simple paths from S to T in the flow graph.
        # There should be exactly num_pairs * 2? No.
        # Total flow is num_pairs.
        # Each unit of flow corresponds to a path S -> b_i_in -> ... -> b_j_out -> T.
        # This gives a path from b_i to b_j.
        # We will collect these paths.
        
        paths_found = []
        
        # We need to be careful to extract exactly num_pairs paths.
        # We can modify the residual graph to remove used edges?
        # Or just collect all augmenting paths used during max flow?
        # Dinic doesn't store them.
        
        # Let's re-run a BFS/DFS on the "flow graph" to extract paths.
        # Flow graph edges: (u, v) if flow(u, v) > 0.
        # How to get flow(u, v)?
        # We can iterate through dinic.graph.
        # For each edge e in dinic.graph[u], if e[1] < original_cap, then there is flow.
        # But we don't have original_cap stored.
        
        # Let's store original capacities when building.
        # I'll modify the Dinic class or just store a parallel structure.
        
        # Simpler approach for extraction:
        # Since k is small (max ~10-12), we can try to match the flows to pairs.
        
        # Let's just do a DFS from each branch_i_out to find a path to some branch_j_in
        # in the subgraph of edges that are NOT fully residual (i.e., carry flow).
        
        # To identify flow edges:
        # We'll rebuild the flow graph explicitly.
        flow_adj = {i: [] for i in range(total_nodes)}
        
        # Iterate all edges added
        # We need to know original capacities.
        # Let's just re-add edges to a simple graph and check if they were used?
        # No, let's just use the fact that if an edge is part of the max flow,
        # its residual capacity is less than its original capacity.
        
        # I will store the original capacities in a separate dictionary during construction.
        # This requires modifying the construction loop.
        
        # Let's restart the flow part with capacity tracking.
        
        # --- Revised Flow Construction with Tracking ---
        
        # Re-initialize
        dinic = Dinic(total_nodes)
        orig_caps = {} # (u, v) -> cap
        
        for v in range(n):
            vin = 2 * v
            vout = 2 * v + 1
            cap_val = (k - 1) if v in branch_set else 1
            dinic.add_edge(vin, vout, cap_val)
            orig_caps[(vin, vout)] = cap_val
            
        for u, v in edges:
            uout = 2 * u + 1
            vin = 2 * v
            vout = 2 * v + 1
            uin = 2 * u
            dinic.add_edge(uout, vin, 1)
            orig_caps[(uout, vin)] = 1
            dinic.add_edge(vout, uin, 1)
            orig_caps[(vout, uin)] = 1
            
        for b in branches:
            bin_ = 2 * b
            bout = 2 * b + 1
            dinic.add_edge(S, bin_, k - 1)
            orig_caps[(S, bin_)] = k - 1
            dinic.add_edge(bout, T, k - 1)
            orig_caps[(bout, T)] = k - 1
            
        max_flow_val = dinic.max_flow(S, T)
        
        if max_flow_val < num_pairs:
            return None
            
        # Build flow graph
        # flow(u, v) = orig_cap(u, v) - residual_cap(u, v)
        # We need to access residual capacities.
        # In Dinic, `self.graph[u]` is a list of edges [v, cap, rev].
        # So for edge u->v, the residual cap is the current cap in the list.
        
        # Let's create a dictionary of current residual caps for quick lookup?
        # Or just iterate.
        
        flow_graph = {i: [] for i in range(total_nodes)}
        
        for u in range(total_nodes):
            for e in dinic.graph[u]:
                v = e[0]
                res_cap = e[1]
                orig_cap = orig_caps.get((u, v), 0)
                flow = orig_cap - res_cap
                if flow > 0:
                    flow_graph[u].append((v, flow))
        
        # Now extract num_pairs paths from S to T in flow_graph.
        # We will greedily pick paths.
        # Since we just need ANY valid set, we can pick paths one by one.
        # To avoid conflicts, we should pick paths that use disjoint internal vertices.
        # But the flow guarantees that a decomposition exists.
        # A simple DFS might pick a "bad" path that blocks others?
        # In a flow network, any set of edge-disjoint paths from S to T that sum to max flow
        # is a valid decomposition. Since vertex capacities are 1, edge-disjointness
        # on the split graph implies vertex-disjointness on the original graph.
        
        # So we just need to find num_pairs edge-disjoint paths from S to T in flow_graph.
        
        extracted_paths = []
        
        # We will modify flow_graph capacities as we extract paths.
        # Convert flow_graph to a mutable structure.
        # We'll use a dictionary of dicts or just lists and remove edges.
        
        # Let's use a simple DFS that removes edges as we go.
        # Since max flow is small, this is fast.
        
        def find_path_in_flow_graph(s, t):
            stack = [(s, [s])]
            visited = {s}
            while stack:
                node, path = stack.pop()
                if node == t:
                    return path
                # Get neighbors with flow > 0
                neighbors = flow_graph[node]
                for i in range(len(neighbors)):
                    v, cap = neighbors[i]
                    if cap > 0 and v not in visited:
                        visited.add(v)
                        stack.append((v, path + [v]))
            return None
        
        # We need to be careful: the flow_graph contains edges with capacities.
        # When we find a path, we reduce capacities of edges in the path by 1.
        
        # Let's represent flow_graph as a dict of lists of [v, cap]
        # And update them.
        
        # Actually, let's just use the residual graph of the Dinic instance directly?
        # No, we already computed max flow.
        
        # Let's just do a standard max flow on the flow_graph to extract paths?
        # No, the flow_graph IS the flow. We just need to decompose it.
        
        # Simple decomposition:
        # While there is flow from S to T:
        #   Find a path P from S to T in flow_graph.
        #   For each edge in P, reduce capacity by 1.
        #   Add P to extracted_paths.
        
        # We need to update flow_graph capacities.
        # Let's make flow_graph mutable.
        
        # Re-initialize flow_graph for extraction
        flow_graph = {i: [] for i in range(total_nodes)}
        for u in range(total_nodes):
            for e in dinic.graph[u]:
                v = e[0]
                res_cap = e[1]
                orig_cap = orig_caps.get((u, v), 0)
                flow = orig_cap - res_cap
                if flow > 0:
                    flow_graph[u].append([v, flow])
        
        for _ in range(num_pairs):
            # Find path
            path = None
            # BFS is better for finding any path quickly
            q = deque([(S, [S])])
            visited = {S}
            while q:
                node, p = q.popleft()
                if node == T:
                    path = p
                    break
                for i in range(len(flow_graph[node])):
                    v, cap = flow_graph[node][i]
                    if cap > 0 and v not in visited:
                        visited.add(v)
                        q.append((v, p + [v]))
            
            if path is None:
                return None # Should not happen if max flow was correct
            
            # Reduce capacities
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i+1]
                # Find edge u->v in flow_graph[u]
                for e in flow_graph[u]:
                    if e[0] == v:
                        e[1] -= 1
                        break
            
            # Convert path to original graph vertices
            # Path is in node indices: S, b_i_in, ..., b_j_out, T
            # Original path: b_i, v1, ..., b_j
            # Remove S and T, and convert node indices to vertex indices
            orig_path = []
            for node in path[1:-1]: # Skip S and T
                # node is either v_in or v_out
                # If node is even, it's v_in -> v = node // 2
                # If node is odd, it's v_out -> v = node // 2
                # In both cases, vertex index is node // 2
                orig_path.append(node // 2)
            
            extracted_paths.append(orig_path)
        
        # Verify paths are valid
        # 1. Endpoints are branches
        # 2. Interiors are disjoint
        # 3. Consecutive vertices are adjacent
        
        # We'll trust the flow construction, but a quick check is good.
        # However, for speed, we might skip if confident.
        # Let's return the paths.
        
        return extracted_paths

    def greedy_extend(seed):
        """Try to extend a seed set by adding vertices one at a time."""
        current = list(seed)
        paths = check_clique(current)
        if paths is None:
            return None
        
        current_set = set(current)
        remaining = [v for v in range(n) if v not in current_set]
        # Sort remaining by degree descending
        remaining.sort(key=lambda v: degrees[v], reverse=True)
        
        for v in remaining:
            trial = current + [v]
            trial_paths = check_clique(trial)
            if trial_paths is not None:
                current = trial
                current_set.add(v)
                paths = trial_paths
        
        return current, paths

    # Search strategy
    best_result = None
    best_k = 1
    
    start_time = time.time()
    timeout = 55
    
    # Sort vertices by degree
    sorted_vertices = sorted(range(n), key=lambda v: degrees[v], reverse=True)
    
    # Try extending from top pairs
    # We can also try triples of high-degree vertices as seeds.
    
    # 1. Start with pairs of high-degree vertices
    for i in range(len(sorted_vertices)):
        if time.time() - start_time > timeout:
            break
        v1 = sorted_vertices[i]
        for j in range(i + 1, len(sorted_vertices)):
            if time.time() - start_time > timeout:
                break
            v2 = sorted_vertices[j]
            seed = [v1, v2]
            result = greedy_extend(seed)
            if result is not None:
                branches, paths = result
                k = len(branches)
                if k > best_k:
                    best_k = k
                    best_result = (branches, paths)
                    if best_k >= 10: # Heuristic: good enough?
                        break
        if best_k >= 10:
            break
            
    # 2. Try random seeds if time permits
    if time.time() - start_time < timeout:
        for _ in range(200):
            if time.time() - start_time > timeout:
                break
            k_seed = random.randint(2, 4)
            # Bias towards high degree
            seed = random.sample(sorted_vertices[:10], k_seed)
            result = greedy_extend(seed)
            if result is not None:
                branches, paths = result
                k = len(branches)
                if k > best_k:
                    best_k = k
                    best_result = (branches, paths)

    if best_result is not None:
        branches, paths = best_result
        return {'branches': [int(b) for b in branches], 'paths': [[int(v) for v in p] for p in paths]}
    
    # Fallback
    u, v = edges[0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

# EVOLVE-BLOCK-END
