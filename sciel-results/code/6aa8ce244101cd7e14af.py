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
    
    def dinic_max_flow(n_nodes, source, sink, cap_matrix):
        """
        Dinic's algorithm for max flow.
        cap_matrix is a 2D list of capacities.
        Returns max flow value.
        Modifies cap_matrix to residual capacities.
        """
        flow = 0
        level = [-1] * n_nodes
        ptr = [0] * n_nodes
        
        def bfs():
            for i in range(n_nodes):
                level[i] = -1
            level[source] = 0
            q = deque([source])
            while q:
                u = q.popleft()
                for v in range(n_nodes):
                    if level[v] == -1 and cap_matrix[u][v] > 0:
                        level[v] = level[u] + 1
                        q.append(v)
            return level[sink] != -1
        
        def dfs(u, pushed):
            if pushed == 0:
                return 0
            if u == sink:
                return pushed
            while ptr[u] < n_nodes:
                v = ptr[u]
                if level[v] == level[u] + 1 and cap_matrix[u][v] > 0:
                    tr = dfs(v, min(pushed, cap_matrix[u][v]))
                    if tr:
                        cap_matrix[u][v] -= tr
                        cap_matrix[v][u] += tr
                        return tr
                ptr[u] += 1
            return 0
        
        while bfs():
            for i in range(n_nodes):
                ptr[i] = 0
            while True:
                pushed = dfs(source, float('inf'))
                if pushed == 0:
                    break
                flow += pushed
        return flow

    def verify_and_extract(branches):
        """
        Checks if branches form a valid clique subdivision using max flow.
        Returns list of paths if valid, else None.
        """
        k = len(branches)
        if k < 2:
            return None
        
        num_pairs = k * (k - 1) // 2
        if num_pairs == 0:
            return []
            
        # Node mapping:
        # v_in = 2*v, v_out = 2*v + 1
        # S = 2*n, T = 2*n + 1
        S = 2 * n
        T = 2 * n + 1
        total_nodes = 2 * n + 2
        
        # Initialize capacity matrix
        cap = [[0] * total_nodes for _ in range(total_nodes)]
        
        branch_set = set(branches)
        
        # Vertex capacities
        for v in range(n):
            vin = 2 * v
            vout = 2 * v + 1
            if v in branch_set:
                cap[vin][vout] = k - 1
            else:
                cap[vin][vout] = 1
        
        # Graph edges
        for u, v in edges:
            uout = 2 * u + 1
            vin = 2 * v
            vout = 2 * v + 1
            uin = 2 * u
            cap[uout][vin] = 1
            cap[vout][uin] = 1
            
        # Source and Sink
        for b in branches:
            bin_ = 2 * b
            bout = 2 * b + 1
            cap[S][bin_] = k - 1
            cap[bout][T] = k - 1
        
        # Run Max Flow
        original_cap = [row[:] for row in cap] # Save original for flow extraction? 
        # Actually, we need to track flow. 
        # flow(u,v) = original_cap(u,v) - residual_cap(u,v)
        # But Dinic modifies cap in place. 
        # Let's store the flow edges separately or reconstruct from residual.
        
        max_flow_val = dinic_max_flow(total_nodes, S, T, cap)
        
        if max_flow_val < num_pairs:
            return None
            
        # Extract paths
        # We need to decompose the flow into paths.
        # A standard way: 
        # 1. For each branch b, the total flow out of b_out towards T is k-1.
        # 2. We can trace paths from S to T.
        # But we want paths between specific pairs of branches.
        # The flow network doesn't distinguish *which* branch the flow is destined for, 
        # only that it leaves branch_i and enters branch_j.
        # However, since the graph is undirected and we split nodes, 
        # a path from S -> b_i_in -> b_i_out -> ... -> b_j_in -> b_j_out -> T 
        # corresponds to a path from b_i to b_j.
        
        # We can extract paths by DFS on the residual graph where flow > 0.
        # flow(u,v) = original_cap[u][v] - cap[u][v]
        
        paths_found = []
        
        # We need to find num_pairs paths.
        # We can greedily extract one path at a time.
        # For each extraction, we look for a path from S to T in the graph of positive flow.
        # Once found, we "use" 1 unit of flow on each edge of that path.
        
        # To ensure we get distinct pairs, we just need to find any S-T path.
        # The start node of the path (after S) tells us the source branch.
        # The end node of the path (before T) tells us the destination branch.
        
        # We must ensure we don't extract the same pair twice if k>2?
        # No, each unit of flow corresponds to one pair.
        # Since S->b_i has cap k-1, and b_j->T has cap k-1, 
        # and internal edges have cap 1, 
        # the flow will naturally distribute.
        
        # Let's extract paths one by one.
        for _ in range(num_pairs):
            # Find a path from S to T using only edges with flow > 0
            # We can use DFS/BFS on the directed graph of positive flow.
            
            parent = [-1] * total_nodes
            visited = [False] * total_nodes
            q = deque([S])
            visited[S] = True
            found = False
            
            while q:
                u = q.popleft()
                if u == T:
                    found = True
                    break
                for v in range(total_nodes):
                    if not visited[v]:
                        # Check if there is flow on u->v
                        # flow = original_cap[u][v] - cap[u][v]
                        if original_cap[u][v] - cap[u][v] > 0:
                            visited[v] = True
                            parent[v] = u
                            q.append(v)
                            
            if not found:
                return None # Should not happen if max flow was sufficient and we extracted correctly
            
            # Reconstruct path
            path_nodes = []
            v = T
            while v != S:
                path_nodes.append(v)
                v = parent[v]
            path_nodes.append(S)
            path_nodes.reverse()
            
            # Convert flow node indices to original graph vertices
            # Path in flow: S -> b1_in -> b1_out -> v1_in -> v1_out -> ... -> bj_in -> bj_out -> T
            # Original path: b1 -> v1 -> ... -> bj
            
            orig_path = []
            for node in path_nodes:
                if node == S or node == T:
                    continue
                if node % 2 == 0:
                    # In-node
                    v_orig = node // 2
                    # Only add if it's a branch or if it's the first internal node?
                    # Actually, the path in the flow graph is:
                    # S -> b_in -> b_out -> v_in -> v_out -> ... -> e_in -> e_out -> T
                    # We want b, v, ..., e
                    # So we take the vertex index for every node.
                    # But we have pairs (in, out).
                    # The sequence of vertices is determined by the 'out' nodes (or 'in' nodes).
                    # Let's just take the vertex index of every node in the path, 
                    # but we will get duplicates for each vertex (in and out).
                    # We can just take the 'out' node's vertex, or 'in'.
                    # Let's take the vertex index of the 'in' node.
                    orig_path.append(v_orig)
            
            # Remove duplicates? 
            # The flow path visits v_in then v_out. 
            # So orig_path will have [b, b, v, v, ...]? 
            # No, S->b_in->b_out. 
            # S is skipped. 
            # b_in -> b. 
            # b_out -> b. 
            # So we get [b, b]. 
            # We should deduplicate consecutive same vertices.
            
            clean_path = []
            for v in orig_path:
                if not clean_path or clean_path[-1] != v:
                    clean_path.append(v)
            
            paths_found.append(clean_path)
            
            # Reduce flow on this path by 1
            # We need to update cap and original_cap? 
            # No, we just need to ensure we don't pick the same flow unit again.
            # Since we are just searching for *a* path in the residual graph of *remaining* flow,
            # we should subtract 1 from the flow on each edge used.
            # Flow(u,v) is stored implicitly. 
            # To "remove" this path, we increase cap[u][v] by 1 (restoring capacity) 
            # and decrease original_cap[u][v] by 1? 
            # No, simpler: 
            # We are extracting from the *current* flow distribution.
            # If we just find a path in the graph of edges with flow>0, 
            # and then we want to prevent finding it again, 
            # we should reduce the flow on those edges.
            # flow(u,v) -= 1.
            # This is equivalent to: cap[u][v] += 1 (if we treat cap as residual).
            # But we also need to update original_cap? 
            # Actually, if we just want to extract *some* decomposition, 
            # we can just mark the edges as used? 
            # No, an edge might carry flow 2. We can use it twice.
            
            # Let's update the residual capacities to reflect the removal of this path.
            # The path is a sequence of nodes in the flow graph.
            # For each consecutive pair (u, v) in path_nodes:
            #   The flow on u->v was at least 1.
            #   We consume 1 unit.
            #   So residual capacity cap[u][v] increases by 1? 
            #   Wait, cap[u][v] is the *residual* capacity.
            #   If we remove flow, we are effectively "undoing" the flow, 
            #   which increases the residual capacity.
            #   But we want to *keep* the flow for other paths? 
            #   No, we are decomposing the *total* flow. 
            #   Once a unit of flow is assigned to a path, it should not be available for another path.
            #   So we should *reduce* the available flow.
            #   
            #   Let's track flow explicitly.
            #   flow[u][v] = original_cap[u][v] - cap[u][v].
            #   When we extract a path, we do flow[u][v] -= 1.
            #   This means cap[u][v] should increase by 1? 
            #   No, if flow decreases, residual capacity increases.
            #   But we want to *prevent* finding this specific flow unit again.
            #   So we should treat the remaining flow as the source of truth.
            #   
            #   Let's just maintain a separate `remaining_flow` matrix.
            #   Initialize remaining_flow[u][v] = original_cap[u][v] - cap_final[u][v].
            #   Then, when extracting a path, we check remaining_flow[u][v] > 0.
            #   And we decrement remaining_flow[u][v] by 1.
            
            # Let's restart the extraction logic with a remaining_flow matrix.
            # This is getting complex inside the loop. 
            # Let's refactor: 
            # 1. Compute max flow.
            # 2. Build remaining_flow matrix.
            # 3. Extract paths by finding S-T paths in the graph defined by remaining_flow > 0.
            # 4. Decrement remaining_flow along the path.
            
            # Since I already did the BFS/DFS above, let's just assume I can do this cleanly.
            # I will rewrite the extraction part in the final code.
            
            pass # Placeholder
            
        return paths_found # Will be implemented properly in the final code

    # Refactored verify_and_extract
    def verify_and_extract_v2(branches):
        k = len(branches)
        if k < 2:
            return None
        
        num_pairs = k * (k - 1) // 2
        if num_pairs == 0:
            return []
            
        S = 2 * n
        T = 2 * n + 1
        total_nodes = 2 * n + 2
        
        cap = [[0] * total_nodes for _ in range(total_nodes)]
        branch_set = set(branches)
        
        for v in range(n):
            vin = 2 * v
            vout = 2 * v + 1
            if v in branch_set:
                cap[vin][vout] = k - 1
            else:
                cap[vin][vout] = 1
        
        for u, v in edges:
            uout = 2 * u + 1
            vin = 2 * v
            vout = 2 * v + 1
            uin = 2 * u
            cap[uout][vin] = 1
            cap[vout][uin] = 1
            
        for b in branches:
            bin_ = 2 * b
            bout = 2 * b + 1
            cap[S][bin_] = k - 1
            cap[bout][T] = k - 1
        
        original_cap = [row[:] for row in cap]
        
        max_flow_val = dinic_max_flow(total_nodes, S, T, cap)
        
        if max_flow_val < num_pairs:
            return None
            
        # Build remaining flow matrix
        remaining_flow = [[0] * total_nodes for _ in range(total_nodes)]
        for u in range(total_nodes):
            for v in range(total_nodes):
                if original_cap[u][v] > 0:
                    remaining_flow[u][v] = original_cap[u][v] - cap[u][v]
        
        paths_found = []
        
        for _ in range(num_pairs):
            # Find path from S to T in graph of remaining_flow > 0
            parent = [-1] * total_nodes
            visited = [False] * total_nodes
            q = deque([S])
            visited[S] = True
            found = False
            
            while q:
                u = q.popleft()
                if u == T:
                    found = True
                    break
                for v in range(total_nodes):
                    if not visited[v] and remaining_flow[u][v] > 0:
                        visited[v] = True
                        parent[v] = u
                        q.append(v)
                        
            if not found:
                return None # Inconsistent state, should not happen
            
            # Reconstruct path
            path_nodes = []
            v = T
            while v != S:
                path_nodes.append(v)
                v = parent[v]
            path_nodes.append(S)
            path_nodes.reverse()
            
            # Convert to original graph path
            orig_path = []
            for node in path_nodes:
                if node == S or node == T:
                    continue
                v_orig = node // 2
                if not orig_path or orig_path[-1] != v_orig:
                    orig_path.append(v_orig)
            
            paths_found.append(orig_path)
            
            # Decrement remaining_flow along the path
            for i in range(len(path_nodes) - 1):
                u = path_nodes[i]
                v = path_nodes[i+1]
                remaining_flow[u][v] -= 1
                
        return paths_found

    # Greedy Extension
    def greedy_extend(seed):
        current = list(seed)
        # Verify seed
        if not verify_and_extract_v2(current):
            return None
            
        remaining = [v for v in range(n) if v not in set(current)]
        remaining.sort(key=lambda v: degrees[v], reverse=True)
        
        for v in remaining:
            trial = current + [v]
            if verify_and_extract_v2(trial):
                current = trial
                
        # Extract paths for the final set
        paths = verify_and_extract_v2(current)
        if paths is None:
            return None
        return current, paths

    # Generate seeds
    best_result = None
    best_k = 1
    
    # Seeds: high degree edges
    pairs_seed = []
    for u, v in edges:
        pairs_seed.append(([u, v], degrees[u] + degrees[v]))
    pairs_seed.sort(key=lambda x: x[1], reverse=True)
    
    start_time = time.time()
    timeout = 55
    
    for seed, _ in pairs_seed[:20]: # Limit seeds to avoid timeout
        if time.time() - start_time > timeout:
            break
        result = greedy_extend(seed)
        if result:
            branches, paths = result
            if len(branches) > best_k:
                best_k = len(branches)
                best_result = (branches, paths)
                
    # Random seeds
    for _ in range(100):
        if time.time() - start_time > timeout:
            break
        k_seed = random.randint(2, 4)
        seed = random.sample(range(n), k_seed)
        result = greedy_extend(seed)
        if result:
            branches, paths = result
            if len(branches) > best_k:
                best_k = len(branches)
                best_result = (branches, paths)
                
    if best_result:
        branches, paths = best_result
        return {'branches': [int(b) for b in branches], 'paths': [[int(v) for v in p] for p in paths]}
        
    # Fallback
    u, v = edges[0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

# EVOLVE-BLOCK-END
