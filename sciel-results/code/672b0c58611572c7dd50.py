"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

from collections import deque

def construct():
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    es = set()
    for u, v in edges:
        es.add((u, v))
        es.add((v, u))
    
    def max_flow(s, t, cap, N):
        # Edmonds-Karp
        flow = 0
        while True:
            parent = [-1] * N
            parent[s] = s
            q = deque([s])
            while q and parent[t] == -1:
                u = q.popleft()
                for v in range(N):
                    if parent[v] == -1 and cap[u][v] > 0:
                        parent[v] = u
                        q.append(v)
            if parent[t] == -1:
                break
            # find bottleneck
            path_flow = float('inf')
            v = t
            while v != s:
                u = parent[v]
                path_flow = min(path_flow, cap[u][v])
                v = u
            # augment
            v = t
            while v != s:
                u = parent[v]
                cap[u][v] -= path_flow
                cap[v][u] += path_flow
                v = u
            flow += path_flow
        return flow
    
    def try_branch_set(B):
        """Try to find vertex-disjoint paths for all pairs in B. Returns paths dict or None."""
        k = len(B)
        Bset = set(B)
        # Node indices: for each vertex v, we have v_in and v_out
        # v_in = v, v_out = v + n
        # For each pair (i,j) of branches, we need a source and sink
        # Source for pair (i,j): 2*n + i*k + j
        # Sink for pair (i,j): 2*n + k*k + i*k + j
        num_pairs = k * (k - 1) // 2
        pair_list = []
        for i in range(k):
            for j in range(i + 1, k):
                pair_list.append((i, j))
        N = 2 * n + 2 * num_pairs
        INF = 10**9
        cap = [[0] * N for _ in range(N)]
        
        def add_edge(u, v, c):
            cap[u][v] += c
            cap[v][u] += c  # residual
        
        # For each vertex, add in->out edge
        for v in range(n):
            if v in Bset:
                add_edge(v, v + n, INF)
            else:
                add_edge(v, v + n, 1)
        
        # For each original edge (u,v), add connections
        for u, v in edges:
            if u in Bset and v in Bset:
                continue
            if u in Bset:
                # u is branch: u_out -> v_in
                add_edge(u + n, v, 1)
            if v in Bset:
                # v is branch: v_out -> u_in
                add_edge(v + n, u, 1)
            if u not in Bset and v not in Bset:
                # both non-branch: u_out -> v_in and v_out -> u_in
                add_edge(u + n, v, 1)
                add_edge(v + n, u, 1)
        
        # For each pair, add source and sink
        for idx, (i, j) in enumerate(pair_list):
            b1, b2 = B[i], B[j]
            src = 2 * n + idx
            sink = 2 * n + num_pairs + idx
            # source -> branch1_in and branch2_in
            add_edge(src, b1, INF)
            add_edge(src, b2, INF)
            # branch1_out and branch2_out -> sink
            add_edge(b1 + n, sink, INF)
            add_edge(b2 + n, sink, INF)
        
        # Check if we can get flow 1 for each pair
        # We need to verify each pair gets exactly 1 unit of flow
        # Run max flow from each source to its sink, but they share capacity
        # Better: run a single max flow with all sources connected to a super source
        # Actually, let's just check: for each pair, can we route 1 unit?
        # The issue is that paths must be vertex-disjoint, so we need to check all simultaneously.
        
        # Let's use a super source and super sink approach
        # Super source SS = N, super sink TT = N+1
        N2 = N + 2
        SS = N
        TT = N + 1
        cap2 = [[0] * N2 for _ in range(N2)]
        for i in range(N):
            for j in range(N):
                cap2[i][j] = cap[i][j]
        
        for idx, (i, j) in enumerate(pair_list):
            src = 2 * n + idx
            add_edge_ss = lambda a, b, c: (cap2.__setitem__(a, [cap2[a][k] + (c if k == b else 0) for k in range(N2)]), cap2.__setitem__(b, [cap2[b][k] + (c if k == a else 0) for k in range(N2)]))
            # Simpler: just add to cap2
            cap2[SS][src] += 1
            cap2[src][SS] += 1
            sink = 2 * n + num_pairs + idx
            cap2[sink][TT] += 1
            cap2[TT][sink] += 1
        
        flow = max_flow(SS, TT, cap2, N2)
        if flow != num_pairs:
            return None
        
        # Extract paths
        # For each pair, find the path from source to sink
        paths = {}
        for idx, (i, j) in enumerate(pair_list):
            src = 2 * n + idx
            sink = 2 * n + num_pairs + idx
            # Find path in residual graph from src to sink
            # We need to trace back through the flow
            # Since flow is 1, there's a unique path
            # BFS from src to sink in the graph where cap[u][v] < original (meaning flow went through)
            # Actually, let's just do a BFS on the original cap graph following edges with positive flow
            # Flow on edge (u,v) = original_cap - cap[u][v] ... but we modified cap
            # Let's track: an edge (u,v) has flow if cap2[v][u] > 0 (residual went back)
            # Simpler: BFS from src, following edges where cap2[u][v] < initial (but we don't have initial)
            # Let's just do a simple path finding: from src, go to b1 or b2, then through non-branch vertices, to the other branch, to sink
            
            # Reconstruct: the path goes src -> (b1 or b2) -> ... -> (other branch) -> sink
            # Find which branch is entered first
            # Check cap2[src][b1] and cap2[src][b2] - one of them should have 0 flow (cap reduced)
            # Actually cap2[src][b1] was INF, if flow went through, cap2[b1][src] = 1
            b1, b2 = B[i], B[j]
            if cap2[b1][src] > 0:
                first, second = b1, b2
            else:
                first, second = b2, b1
            
            # Now find path from first to second through non-branch vertices
            # BFS from first, avoiding other branches
            path = [first]
            visited = {first}
            q = deque([(first, [first])])
            found = False
            while q:
                u, p = q.popleft()
                if u == second:
                    path = p
                    found = True
                    break
                for v in adj[u]:
                    if v in visited or v in Bset:
                        continue
                    # Check if edge u_out -> v_in has flow
                    # Flow went through if cap2[v][u + n] > 0 (residual)
                    # The edge was u+n -> v with cap 1, if flow=1, cap2[v][u+n] = 1
                    if cap2[v][u + n] > 0:
                        visited.add(v)
                        q.append((v, p + [v]))
            if not found:
                return None
            # path is [first, ..., second]
            paths[(i, j)] = path
        
        return paths
    
    # Find a large branch set
    # Strategy: Try all pairs as seeds, then greedily expand.
    # Sort vertices by degree to prioritize "central" vertices in expansion.
    degrees = [0] * n
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    
    # Create a list of vertices sorted by degree descending
    vertices_by_degree = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    best_B = None
    best_paths = None
    best_size = 0
    
    # Try all pairs (a, b) as the initial branch set
    # To save time, we can prioritize pairs where a and b are connected or high degree
    # But with n=24, all pairs is feasible.
    for a in range(n):
        for b in range(a + 1, n):
            # Optional: skip if not adjacent? The paths can be length > 1, so they don't need to be adjacent.
            # However, if they are not adjacent, the path must go through other vertices.
            # Let's try all pairs.
            
            B = [a, b]
            paths = try_branch_set(B)
            if paths is None:
                continue
                
            current_B = B
            current_paths = paths
            
            # Greedily add vertices
            # We try to add vertices in order of decreasing degree (excluding those already in B)
            # We iterate through the sorted list and try to add each one if it fits.
            # If one fails, we try the next. We don't remove previously added ones in this simple greedy.
            # To improve, we could try to add them in a loop until no more can be added.
            
            changed = True
            while changed:
                changed = False
                for v in vertices_by_degree:
                    if v in current_B:
                        continue
                    new_B = current_B + [v]
                    new_paths = try_branch_set(new_B)
                    if new_paths is not None:
                        current_B = new_B
                        current_paths = new_paths
                        changed = True
                        # We could break here to restart the loop with the new set,
                        # but continuing to check others in the same pass is also okay.
                        # Restarting is often better for greedy.
                        break 
                        
            if len(current_B) > best_size:
                best_size = len(current_B)
                best_B = current_B
                best_paths = current_paths
                
    if best_B is None:
        # Fallback: just an edge
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
    
    # Build result
    branches = list(best_B)
    result_paths = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            result_paths.append(list(best_paths[(i, j)]))
    
    return {'branches': branches, 'paths': result_paths}
# EVOLVE-BLOCK-END
