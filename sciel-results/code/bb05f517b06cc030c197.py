import time
from itertools import combinations
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
import sys

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    degrees = [len(adj[i]) for i in range(n)]
    
    # Helper to check if a set of branches forms a clique subdivision
    def check_clique_subdivision(branches):
        k = len(branches)
        if k < 2:
            return None
            
        # Map original vertices to flow network vertices
        # v_in = 2*v, v_out = 2*v+1
        # S = 2*n, T = 2*n+1
        num_nodes = 2 * n + 2
        S = 2 * n
        T = 2 * n + 1
        
        # Capacities matrix
        # Using a dense matrix for simplicity since N is small (24) -> 50x50
        cap = np.zeros((num_nodes, num_nodes), dtype=int)
        
        # 1. Internal vertex capacity 1
        for v in range(n):
            cap[2*v][2*v+1] = 1
            
        # 2. Edge capacities (infinity)
        INF = 100
        for u, v in edges:
            cap[2*u+1][2*v] = INF
            cap[2*v+1][2*u] = INF
            
        # 3. Source and Sink connections based on branch indices
        # Sort branches to define direction: flow from lower index to higher index
        sorted_branches = sorted(branches)
        branch_indices = {b: i for i, b in enumerate(sorted_branches)}
        
        for b in sorted_branches:
            idx = branch_indices[b]
            # Flow leaving b (to higher indices)
            out_flow = k - 1 - idx
            # Flow entering b (from lower indices)
            in_flow = idx
            
            if out_flow > 0:
                cap[S][2*b] = out_flow
            if in_flow > 0:
                cap[2*b+1][T] = in_flow
                
        # Convert to CSR for scipy
        graph = csr_matrix(cap)
        
        # Run max flow
        flow_value, _, _, node_flow = maximum_flow(graph, S, T)
        
        # Check if max flow is sufficient
        required_flow = k * (k - 1) // 2
        if flow_value < required_flow:
            return None
            
        # Extract paths
        # We need to find paths for each pair (b_i, b_j) with i < j
        # The flow is decomposed into paths.
        # We can trace paths in the residual graph where flow > 0.
        
        # Reconstruct the flow graph edges
        # node_flow gives the net flow into each node? 
        # Actually, scipy's maximum_flow returns:
        # flow_value, residual_graph, pred, node_flow
        # node_flow: array of net flow into each node.
        
        # To extract paths, we look at the original graph edges where flow was sent.
        # We can compute flow on each edge (u, v) as:
        # flow(u, v) = cap[u, v] - residual[u, v]
        
        residual = graph - csr_matrix(cap) # This is not quite right, residual is returned
        # The function returns residual_graph which is the residual capacity.
        # So flow(u,v) = cap[u,v] - residual_graph[u,v]
        
        # Let's get the residual graph properly.
        # maximum_flow returns: flow_value, residual_graph, pred, node_flow
        # I called it as: flow_value, _, _, node_flow = maximum_flow(...)
        # I need the residual graph to calculate flow on edges.
        # Let's re-run or modify to get residual.
        
        # Re-run to get all outputs
        flow_value, residual_graph, pred, node_flow = maximum_flow(graph, S, T)
        
        if flow_value < required_flow:
            return None
            
        # Calculate flow on each directed edge (u, v)
        # flow(u, v) = cap[u, v] - residual_graph[u, v]
        
        paths = []
        
        # For each pair (b_i, b_j) with i < j, find a path
        # Since flows are integral and capacities on internal edges are 1,
        # the flow decomposes into unit paths.
        
        # We can find paths by following flow from b_i to b_j.
        # However, multiple paths might share edges? No, internal edges have cap 1, so flow is 0 or 1.
        # So the subgraph of edges with flow 1 is a set of vertex-disjoint paths and cycles.
        # We just need to find the path from b_i to b_j for each pair.
        
        # Build a list of edges with positive flow
        flow_edges = []
        for u in range(num_nodes):
            for v in range(num_nodes):
                if cap[u, v] > 0:
                    f = cap[u, v] - residual_graph[u, v]
                    if f > 0:
                        flow_edges.append((u, v, f))
                        
        # We need to trace specific paths.
        # Since the graph is small, we can do a DFS from b_i to b_j in the flow graph.
        
        def find_path_flow(start_node, end_node):
            # DFS in the directed graph of flow edges
            visited = set()
            path = []
            
            def dfs(node):
                if node == end_node:
                    path.append(node)
                    return True
                visited.add(node)
                for u, v, f in flow_edges:
                    if u == node and f > 0 and v not in visited:
                        path.append(v)
                        if dfs(v):
                            return True
                        path.pop()
                return False
                
            path.append(start_node)
            if dfs(start_node):
                return path
            return None
            
        result_paths = []
        for i in range(k):
            for j in range(i + 1, k):
                b_i = sorted_branches[i]
                b_j = sorted_branches[j]
                
                # Path starts at b_i (out) and ends at b_j (in)
                # In the flow network, flow enters b_i from S (if i is source) or from neighbors.
                # Flow leaves b_i to neighbors or to T (if i is sink).
                # Wait, our construction:
                # S -> b_i_in (if out_flow > 0)
                # b_i_in -> b_i_out
                # b_i_out -> neighbors
                # neighbors -> b_j_in
                # b_j_in -> b_j_out
                # b_j_out -> T (if in_flow > 0)
                
                # So the path in the flow network goes:
                # S -> ... -> b_i_in -> b_i_out -> ... -> b_j_in -> b_j_out -> T
                # We want the path from b_i to b_j in the original graph.
                # This corresponds to the segment from b_i_out to b_j_in in the flow network.
                # Or rather, the path from b_i to b_j includes the internal vertices.
                
                start_flow_node = 2 * b_i + 1  # b_i_out
                end_flow_node = 2 * b_j        # b_j_in
                
                # Find path in flow graph from start_flow_node to end_flow_node
                f_path = find_path_flow(start_flow_node, end_flow_node)
                
                if f_path is None:
                    # This shouldn't happen if max flow is correct
                    return None
                
                # Convert flow nodes back to original vertices
                # Flow node 2*v is v_in, 2*v+1 is v_out
                # The path in original graph is:
                # b_i, then internal vertices, then b_j
                # The flow path is: b_i_out, ..., b_j_in
                # We need to map these back.
                # If a node in f_path is 2*v, it's v_in.
                # If 2*v+1, it's v_out.
                # The internal vertices are the ones that are not b_i or b_j.
                
                orig_path = [b_i]
                for node in f_path:
                    if node == 2 * b_i + 1: # start
                        continue
                    if node == 2 * b_j: # end
                        # This is b_j_in, so the vertex is b_j
                        # But we add b_j at the end
                        continue
                    # It's an internal node
                    # It could be v_in or v_out.
                    # The vertex is v = node // 2
                    v = node // 2
                    if v != b_i and v != b_j:
                        orig_path.append(v)
                orig_path.append(b_j)
                
                result_paths.append(orig_path)
                
        return result_paths

    # Search for maximum k
    # Start with high degree vertices
    candidates = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    best_solution = None
    best_k = 0
    
    # Try k from 8 down to 2? 
    # N=24, max k is limited by N.
    # Let's try up to 8 first, then lower.
    
    # To be efficient, we generate subsets.
    # For k=8, C(24,8) is huge.
    # We restrict to top M candidates.
    
    for k in range(8, 1, -1):
        # Use top 12 or 15 candidates for efficiency
        pool_size = min(n, 15)
        pool = candidates[:pool_size]
        
        # If k is large, we might need to consider more vertices, but top 15 is a good heuristic.
        # If no solution found in top 15, we might miss one, but it's a trade-off for time.
        
        found = False
        for subset in combinations(pool, k):
            branches = list(subset)
            paths = check_clique_subdivision(branches)
            if paths is not None:
                best_solution = {
                    'branches': branches,
                    'paths': paths
                }
                best_k = k
                found = True
                break
        
        if found:
            break
            
    if best_solution is None:
        # Fallback: try smaller k with larger pool
        for k in range(5, 1, -1):
            pool_size = min(n, 20)
            pool = candidates[:pool_size]
            found = False
            for subset in combinations(pool, k):
                branches = list(subset)
                paths = check_clique_subdivision(branches)
                if paths is not None:
                    best_solution = {
                        'branches': branches,
                        'paths': paths
                    }
                    best_k = k
                    found = True
                    break
            if found:
                break
                
    if best_solution is None:
        # Last resort: any edge
        u, v = edges[0]
        best_solution = {
            'branches': [u, v],
            'paths': [[u, v]]
        }
        
    # Ensure types are correct
    best_solution['branches'] = [int(b) for b in best_solution['branches']]
    best_solution['paths'] = [[int(v) for v in p] for p in best_solution['paths']]
    
    return best_solution

if __name__ == "__main__":
    construct()
