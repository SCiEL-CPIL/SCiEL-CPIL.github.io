"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list for the original graph
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Sort vertices by degree descending for heuristic search
    vertices_by_degree = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)

    class Dinic:
        def __init__(self, n):
            self.n = n
            self.graph = [[] for _ in range(n)]
            
        def add_edge(self, u, v, cap):
            self.graph[u].append([v, cap, len(self.graph[v])])
            self.graph[v].append([u, 0, len(self.graph[u]) - 1])
            
        def bfs(self, s, t):
            self.level = [-1] * self.n
            self.level[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for e in self.graph[u]:
                    v, cap, rev = e
                    if cap > 0 and self.level[v] < 0:
                        self.level[v] = self.level[u] + 1
                        q.append(v)
            return self.level[t] >= 0
            
        def dfs(self, v, t, f):
            if v == t:
                return f
            for i in range(self.it[v], len(self.graph[v])):
                self.it[v] = i
                e = self.graph[v][i]
                v_next, cap, rev = e
                if cap > 0 and self.level[v] < self.level[v_next]:
                    d = self.dfs(v_next, t, min(f, cap))
                    if d > 0:
                        e[1] -= d
                        self.graph[v_next][rev][1] += d
                        return d
            return 0
            
        def max_flow(self, s, t):
            flow = 0
            INF = 10**18
            while self.bfs(s, t):
                self.it = [0] * self.n
                while True:
                    f = self.dfs(s, t, INF)
                    if f == 0:
                        break
                    flow += f
            return flow

    def check_branch_set(B):
        """
        Returns paths dict if valid, else None.
        B: list of branch vertices.
        """
        k = len(B)
        if k < 2:
            return None
        Bset = set(B)
        num_pairs = k * (k - 1) // 2
        pair_list = []
        for i in range(k):
            for j in range(i + 1, k):
                pair_list.append((i, j))
                
        # Node indices in flow graph:
        # v_in = v
        # v_out = v + n
        # src_idx = 2*n + i
        # sink_idx = 2*n + num_pairs + i
        total_nodes = 2 * n + 2 * num_pairs
        dinic = Dinic(total_nodes)
        
        # Add vertex splitting edges
        for v in range(n):
            if v in Bset:
                dinic.add_edge(v, v + n, 10**9) # INF capacity for branches
            else:
                dinic.add_edge(v, v + n, 1)     # Capacity 1 for non-branches
        
        # Add graph edges
        for u, v in edges:
            if u in Bset and v in Bset:
                continue # Direct edge between branches is not allowed in path interior, but path can be direct? 
                # "Path endpoints must be their branch pair... interiors must be pairwise disjoint and avoid every branch."
                # If u and v are both branches, a path of length 1 [u, v] has no interior. This is valid.
                # However, in the flow model, we route from src -> u_in -> u_out -> v_in -> v_out -> sink?
                # No, the path is from u to v.
                # The flow model connects src -> u_in, u_out -> sink.
                # If u and v are adjacent, we need a path u -> v.
                # In the split graph: u_in -> u_out is INF.
                # Edge u-v: u_out -> v_in (cap 1) and v_out -> u_in (cap 1).
                # If we send flow from src(u,v) to u_in, then u_in->u_out, then u_out->v_in, then v_in->v_out, then v_out->sink.
                # This works.
                
            # We need to add edges for the flow to traverse the original graph edges.
            # The flow goes from in to out.
            # Original edge (u,v) allows flow u_out -> v_in and v_out -> u_in.
            # BUT, if u is a branch, it has INF capacity in->out.
            # If u is non-branch, it has cap 1.
            
            # The edges in the flow graph corresponding to original edges:
            # u_out -> v_in with capacity 1 (since it consumes vertex v? No, capacity is on vertices).
            # Wait, the standard vertex-disjoint paths formulation:
            # Split each vertex v into v_in, v_out.
            # Edge v_in -> v_out with capacity 1 (or INF for sources/sinks).
            # For each original edge (u,v), add edges u_out -> v_in and v_out -> u_in with capacity 1 (or INF?).
            # Actually, edge capacities should be large enough, but vertex capacities restrict the flow.
            # So u_out -> v_in can have capacity 1 (or more, doesn't matter much as long as it's >=1) because v_in -> v_out is the bottleneck for v.
            # However, if multiple edges go into v_in, they share the capacity of v_in -> v_out.
            # So we can set capacity of u_out -> v_in to 1.
            
            dinic.add_edge(u + n, v, 1)
            dinic.add_edge(v + n, u, 1)

        # Add super source and sink connections for each pair
        SS = 2 * n
        TT = 2 * n + num_pairs
        # We don't need a single SS/TT if we just want to check if we can route 1 unit for each pair independently?
        # No, they share the graph. We need a global max flow.
        # Create a super source S and super sink T.
        S = total_nodes
        T = total_nodes + 1
        dinic.n = total_nodes + 2
        dinic.graph.append([])
        dinic.graph.append([])
        
        for idx, (i, j) in enumerate(pair_list):
            b1, b2 = B[i], B[j]
            src_node = 2 * n + idx
            sink_node = 2 * n + num_pairs + idx
            
            # Connect Super Source to pair source
            dinic.add_edge(S, src_node, 1)
            
            # Connect pair source to branch in-nodes
            dinic.add_edge(src_node, b1, 10**9)
            dinic.add_edge(src_node, b2, 10**9)
            
            # Connect branch out-nodes to pair sink
            dinic.add_edge(b1 + n, sink_node, 10**9)
            dinic.add_edge(b2 + n, sink_node, 10**9)
            
            # Connect pair sink to Super Sink
            dinic.add_edge(sink_node, T, 1)
            
        max_f = dinic.max_flow(S, T)
        
        if max_f != num_pairs:
            return None
        
        # Extract paths
        # We need to trace the flow.
        # For each pair idx, find the path in the residual graph?
        # Or simpler: The flow is 1 on each path.
        # We can reconstruct by following the flow edges.
        # Since capacities are 1 on internal edges, the flow decomposition is unique (mostly).
        
        paths = {}
        for idx, (i, j) in enumerate(pair_list):
            src_node = 2 * n + idx
            sink_node = 2 * n + num_pairs + idx
            b1, b2 = B[i], B[j]
            
            # Determine which branch is the start and which is the end
            # Check if flow went S -> src_node -> b1 or b2
            # Look at residual capacity or just try to find path from src_node to sink_node in the "flow graph"
            # The edges with positive flow are those where original_cap - current_cap > 0?
            # In Dinic, we don't store original cap easily.
            # But we can just do a BFS from src_node to sink_node in the graph of edges that *could* carry flow.
            # Actually, since flow is 1, we can just find *any* path from src_node to sink_node in the original topology of the flow graph,
            # restricted to edges that have capacity > 0 in the *initial* graph? No, we need the edges that were used.
            
            # Better: Reconstruct from the residual graph.
            # An edge (u,v) has flow if the reverse edge (v,u) has capacity > 0 in the residual graph?
            # In Dinic implementation above:
            # graph[u] contains [v, cap, rev].
            # graph[v] contains [u, 0, rev2].
            # If flow went u->v, then graph[u][idx][1] decreases, graph[v][rev][1] increases.
            # So if graph[v][rev][1] > 0, then flow went u->v.
            
            # Let's build a map of flow edges.
            # Actually, simpler: Just do a BFS from src_node to sink_node in the graph where we only traverse edges (u,v) if there is positive flow.
            # How to detect positive flow?
            # We can store the flow during max_flow? No, standard Dinic doesn't.
            # But we can check: for each edge in graph[u], if the reverse edge has capacity > 0, then this edge carried flow.
            
            flow_adj = [[] for _ in range(total_nodes + 2)]
            for u in range(total_nodes + 2):
                for e in dinic.graph[u]:
                    v, cap, rev = e
                    # Check reverse edge
                    rev_edge = dinic.graph[v][rev]
                    if rev_edge[1] > 0:
                        # Flow went from u to v
                        flow_adj[u].append(v)
                        
            # Now BFS from src_node to sink_node using flow_adj
            # But we only want the path in the *original* vertices, not the flow graph nodes.
            # The path goes:
            # src_node -> (b1 or b2) -> ... -> (other branch) -> sink_node
            # We need to map back to original vertices.
            
            # Find first hop from src_node
            # src_node connects to b1 and b2.
            # Check which one has flow.
            start_branch = None
            end_branch = None
            if b1 in flow_adj[src_node]:
                start_branch = b1
                end_branch = b2
            elif b2 in flow_adj[src_node]:
                start_branch = b2
                end_branch = b1
            else:
                return None # Should not happen if max_f is correct
                
            # Now find path from start_branch to end_branch in the original graph using flow edges.
            # The flow edges between original vertices correspond to:
            # u_in -> u_out (if u is on path)
            # u_out -> v_in (if edge u-v is on path)
            
            # We can just BFS from start_branch to end_branch in a graph where:
            # Nodes are original vertices 0..n-1.
            # Edge u->v exists if flow went u_out -> v_in.
            # Also, we must ensure we don't use other branches.
            
            # Let's extract the sequence of original vertices.
            # Current node in flow graph: start_branch (which is start_branch_in? No, src_node -> b1_in).
            # So we are at b1_in.
            # Next must be b1_out (flow b1_in -> b1_out).
            # Then b1_out -> some v_in.
            
            path = [start_branch]
            curr_node = start_branch # This is b1_in
            target = end_branch
            
            visited_flow_nodes = set()
            visited_flow_nodes.add(src_node)
            visited_flow_nodes.add(curr_node)
            
            while curr_node != target:
                # From curr_node (which is x_in), go to x_out
                x = curr_node
                next_node = x + n
                if next_node not in flow_adj[curr_node]:
                     # Maybe x is the target? No, we are not at target.
                     # If x is a branch, it must go out.
                     return None
                visited_flow_nodes.add(next_node)
                curr_node = next_node
                path.append(x) # x is the branch or non-branch vertex
                
                # From x_out, go to some y_in
                found_next = False
                for y_in in flow_adj[curr_node]:
                    # y_in is an original vertex index (0..n-1)
                    if y_in in Bset and y_in != target:
                        continue # Cannot go through other branches
                    if y_in in visited_flow_nodes:
                        continue
                    # Check if y_in is a valid next step
                    # We just need to follow the flow
                    visited_flow_nodes.add(y_in)
                    curr_node = y_in
                    path.append(y_in)
                    found_next = True
                    break
                if not found_next:
                    return None
                    
            # The path ends at end_branch (which is y_in).
            # We need to make sure the last step is valid.
            # The path list contains original vertices.
            # [start, ..., end]
            # Check if it's a valid simple path in original graph.
            # And endpoints are branches.
            
            # Optimization: The path extracted might include the "in" node of the target.
            # The flow goes ... -> end_branch_in -> end_branch_out -> sink_node.
            # Our BFS stopped when curr_node == target (which is end_branch_in).
            # So path is [start, ..., end].
            
            # Verify path validity
            if len(path) < 2:
                return None
            if path[0] != start_branch or path[-1] != end_branch:
                return None
            for i in range(len(path) - 1):
                if (path[i], path[i+1]) not in [(u,v) for u,v in edges] and (path[i+1], path[i]) not in [(u,v) for u,v in edges]:
                     # Check adjacency
                     if path[i+1] not in adj[path[i]]:
                         return None
                         
            # Check interiors are disjoint and avoid branches
            # This is guaranteed by the flow if we constructed it correctly?
            # Yes, vertex capacities 1 ensure disjointness.
            
            paths[(i, j)] = path
            
        return paths

    # Search for the largest B
    # Heuristic: Try to add vertices in order of degree.
    # Start with empty set? Or a small set?
    # Let's try to find a large clique-like structure.
    # Greedy with backtracking.
    
    best_B = None
    best_paths = None
    
    # Initial candidates: high degree vertices
    candidates = vertices_by_degree
    
    # Try to build a set B
    # We can try all subsets of the top K candidates?
    # Or just greedy add.
    
    current_B = []
    
    # To be safe and fast, let's try a few strategies:
    # 1. Greedy: Add vertices one by one if possible.
    # 2. Randomized Greedy: Try multiple times with different orders.
    
    import random
    random.seed(42)
    
    for _ in range(10): # Try 10 times with different shuffles of candidates
        cand = candidates[:]
        random.shuffle(cand)
        # But keep high degree ones prioritized?
        # Let's just shuffle the whole list but maybe bias towards high degree.
        
        B = []
        for v in cand:
            new_B = B + [v]
            paths = check_branch_set(new_B)
            if paths is not None:
                B = new_B
                if best_B is None or len(B) > len(best_B):
                    best_B = B
                    best_paths = paths
        
        # If we found a large set, maybe we can stop?
        # Max possible is n.
        if best_B and len(best_B) == n:
            break
            
    if best_B is None:
        # Fallback: Find any edge
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
        
    branches = list(best_B)
    result_paths = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            # check_branch_set returns paths keyed by (i,j) indices in B
            # We need to map back to the order in 'branches'
            # In check_branch_set, B is the list passed in.
            # So paths[(i,j)] corresponds to branches[i], branches[j].
            result_paths.append(list(best_paths[(i, j)]))
            
    return {'branches': branches, 'paths': result_paths}

# EVOLVE-BLOCK-END
