import sys
from itertools import combinations

# Increase recursion limit just in case
sys.setrecursionlimit(10000)

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # 1. Find Max Clique using Bron-Kerbosch
    max_clique = []
    
    def bron_kerbosch(R, P_set, X):
        nonlocal max_clique
        if not P_set and not X:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        
        if P_set:
            u = list(P_set)[0]
        else:
            u = list(X)[0]
            
        candidates = [v for v in P_set if v not in adj[u]]
        
        for v in candidates:
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
            
    bron_kerbosch(set(), set(range(n)), set())
    
    if not max_clique:
        max_clique = [0]
        
    if len(max_clique) < 2:
        for u, v in edges:
            max_clique = [u, v]
            break
            
    branches = list(max_clique)
    branch_set = set(branches)
    
    # 2. Greedy addition of vertices using Max Flow
    # We will try to add vertices from the remaining set
    remaining = [v for v in range(n) if v not in branch_set]
    
    # Store paths: dict mapping (u, v) to list of vertices
    # Initialize with clique edges
    paths_dict = {}
    for u, v in combinations(branches, 2):
        # Ensure consistent key ordering (min, max)
        key = (min(u, v), max(u, v))
        paths_dict[key] = [u, v]
        
    # Max Flow Helper
    # Nodes: 0 to 2n-1. v_in = 2*v, v_out = 2*v + 1
    # Super Sink: 2*n
    
    def can_add_vertex(v, current_branches):
        """
        Returns True if v can be added, and also returns the list of paths from v to each branch in current_branches.
        """
        k = len(current_branches)
        if k == 0:
            return True, []
            
        N_nodes = 2 * n + 1
        super_sink = 2 * n
        
        # Adjacency list for flow graph: list of [neighbor, capacity]
        # We'll use a dict of dicts or list of lists for residual graph
        # Let's use a standard adjacency matrix or list for simplicity since N is small
        # Residual graph: res[u][v] = capacity
        res = [[0] * N_nodes for _ in range(N_nodes)]
        
        # Add edges
        # 1. Vertex splitting edges
        for x in range(n):
            x_in = 2 * x
            x_out = 2 * x + 1
            
            if x == v:
                # Source, can send k units
                res[x_in][x_out] = k
            elif x in branch_set:
                # Other branches: cannot pass through, but can be destination
                # We handle destination by connecting x_out to super_sink
                # So x_in -> x_out should be 0? 
                # If flow enters x_in, it must go to x_out to reach sink.
                # But we don't want flow to pass THROUGH x to another node.
                # The edge x_out -> y_in is added separately.
                # If we set res[x_in][x_out] = 0, flow cannot enter x_in and leave x_out.
                # But we want flow to END at x.
                # So we add edge x_out -> super_sink with cap 1.
                # And we must ensure flow doesn't leave x_out to other nodes.
                # So we just don't add edges from x_out to other nodes?
                # Or set their capacity to 0.
                res[x_in][x_out] = 0 
            else:
                # Internal vertex
                res[x_in][x_out] = 1
                
        # 2. Graph edges
        for u, w in edges:
            u_out = 2 * u + 1
            w_in = 2 * w
            
            # Only add if we can go through
            # We can always traverse edges in the flow graph, 
            # but the vertex capacities will restrict it.
            # However, if u is a branch (and u != v), u_out -> w_in should be 0?
            # If u is a branch, flow ends at u_out -> super_sink.
            # It shouldn't go u_out -> w_in.
            # So we should check if u is a branch.
            
            # Actually, simpler: Add all graph edges with INF capacity.
            # The vertex splitting capacities will handle the rest.
            # If x is a branch, res[x_in][x_out] = 0.
            # So flow cannot go x_in -> x_out.
            # Thus, it cannot leave x_out to anywhere else?
            # Wait, the edge is x_out -> y_in.
            # If flow is at x_out, it can go to y_in.
            # But how did it get to x_out?
            # It must have come from x_in -> x_out.
            # But that edge has cap 0 for branches.
            # So flow cannot reach x_out for branches (except v).
            # So it cannot leave x_out.
            # This is correct.
            
            res[u_out][w_in] = float('inf')
            res[2 * w + 1][2 * u] = float('inf')
            
        # 3. Connect branch outs to super sink
        for u in current_branches:
            u_out = 2 * u + 1
            res[u_out][super_sink] = 1
            
        # Run Max Flow from v_in to super_sink
        source = 2 * v
        sink = super_sink
        
        flow = 0
        while True:
            # BFS to find augmenting path
            parent = [-1] * N_nodes
            parent[source] = -2
            queue = [source]
            
            while queue:
                u = queue.pop(0)
                for w in range(N_nodes):
                    if parent[w] == -1 and res[u][w] > 0:
                        parent[w] = u
                        if w == sink:
                            break
                        queue.append(w)
                        
            if parent[sink] == -1:
                break
                
            # Find min capacity on path
            path_flow = float('inf')
            s = sink
            while s != source:
                u = parent[s]
                path_flow = min(path_flow, res[u][s])
                s = u
                
            # Update residual graph
            s = sink
            while s != source:
                u = parent[s]
                res[u][s] -= path_flow
                res[s][u] += path_flow
                s = u
                
            flow += path_flow
            
        if flow == k:
            # Extract paths
            paths = []
            # We need to trace the flow in the ORIGINAL residual graph? 
            # No, we need to trace the flow that was sent.
            # We can look at the edges with positive flow in the direction of the flow.
            # Since we updated res, the "used" capacity is res[v][u] in the reverse direction?
            # Or we can store the flow separately.
            # Let's re-run or store the flow amounts.
            # For simplicity, we can check which edges have flow.
            # An edge (u, v) has flow if res[v][u] > 0 (since we added to reverse).
            # But we need to decompose into k paths.
            
            # Let's reconstruct the flow graph edges that carry flow.
            # flow_edges = {}
            # for u in range(N_nodes):
            #   for w in range(N_nodes):
            #     if res[w][u] > 0: # This means flow u->w was sent
            #        flow_edges[(u,w)] = res[w][u]
            
            # This is getting complicated. 
            # Alternative: Just return True/False for now to test the logic.
            # But we need the paths.
            
            # Let's implement a simple path tracer.
            # We know the flow is k.
            # We can find k disjoint paths in the subgraph of edges with positive flow.
            # Edges with positive flow: 
            # For each u, w: if res[w][u] > 0, then there is flow from u to w of amount res[w][u].
            
            used_edges = set()
            # Build list of (u, w, flow)
            flow_list = []
            for u in range(N_nodes):
                for w in range(N_nodes):
                    if res[w][u] > 0:
                        flow_list.append((u, w, res[w][u]))
                        
            # Now decompose flow_list into k paths from source to sink.
            # This is a standard flow decomposition problem.
            # Since k is small, we can do it greedily.
            
            # Create a dict of flow capacities remaining
            cap = {}
            for u, w, f in flow_list:
                cap[(u, w)] = f
                
            found_paths = []
            for _ in range(k):
                # Find a path from source to sink in the residual flow graph
                # Using DFS
                visited = set()
                path_nodes = []
                
                def dfs(node):
                    if node == sink:
                        return True
                    visited.add(node)
                    for w in range(N_nodes):
                        if (node, w) in cap and cap[(node, w)] > 0 and w not in visited:
                            path_nodes.append(w)
                            if dfs(w):
                                return True
                            path_nodes.pop()
                    return False
                    
                # Reset path_nodes for each path
                path_nodes = [source]
                if dfs(source):
                    # path_nodes contains [source, ..., sink]
                    # Convert to original vertices
                    orig_path = []
                    for node in path_nodes:
                        if node == super_sink:
                            break
                        # node is 2*x or 2*x+1
                        # If node is even, it's x_in. If odd, x_out.
                        # The path goes x_in -> x_out -> y_in...
                        # We want the sequence of vertices x, y, ...
                        # If node is x_in, add x.
                        # If node is x_out, skip (already added x).
                        # Actually, the path in flow graph is v_in -> v_out -> ... -> u_in -> u_out -> sink.
                        # Vertices are v, ..., u.
                        if node % 2 == 0:
                            orig_path.append(node // 2)
                            
                    # Check if last vertex is in current_branches
                    # It should be.
                    found_paths.append(orig_path)
                    
                    # Decrease capacities
                    for i in range(len(path_nodes) - 1):
                        u = path_nodes[i]
                        w = path_nodes[i+1]
                        cap[(u, w)] -= 1
                        if cap[(u, w)] == 0:
                            del cap[(u, w)]
                else:
                    return False, [] # Should not happen if flow == k
                    
            return True, found_paths
        else:
            return False, []

    # Greedy loop
    added_any = True
    while added_any and remaining:
        added_any = False
        for i, v in enumerate(remaining):
            can_add, new_paths = can_add_vertex(v, branches)
            if can_add:
                # Add v to branches
                branches.append(v)
                branch_set.add(v)
                remaining.pop(i)
                added_any = True
                
                # Add paths to paths_dict
                for p in new_paths:
                    # p is [v, ..., u]
                    start = p[0]
                    end = p[-1]
                    key = (min(start, end), max(start, end))
                    paths_dict[key] = p
                    
                # Restart loop to check other vertices with new branch set
                break
                
    # Construct final paths list
    final_paths = []
    for u, v in combinations(branches, 2):
        key = (min(u, v), max(u, v))
        if key in paths_dict:
            final_paths.append(paths_dict[key])
        else:
            # Should not happen
            final_paths.append([u, v])
            
    return {'branches': branches, 'paths': final_paths}

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}
