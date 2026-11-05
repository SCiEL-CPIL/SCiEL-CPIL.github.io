"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def construct():
    n = P['n']
    edges = P['edge_order']
    m = len(edges)
    
    # Build edge position lookup
    edge_pos = {}
    for idx, (u, v) in enumerate(edges):
        a, b = min(u, v), max(u, v)
        edge_pos[(a, b)] = idx + 1  # 1-based
    
    def cycle_cost(cycle):
        """Compute the max edge position in the cycle."""
        max_pos = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            a, b = min(u, v), max(u, v)
            pos = edge_pos.get((a, b), m + 1)
            if pos > max_pos:
                max_pos = pos
        return max_pos
    
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 80.0
    
    # Step 1: Greedy construction
    parent = list(range(n))
    rank = [0] * n
    comp_size = [1] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True
    
    degree = [0] * n
    selected_edges = []
    
    for u, v in edges:
        if degree[u] >= 2 or degree[v] >= 2:
            continue
        fu, fv = find(u), find(v)
        if fu == fv:
            if comp_size[fu] == n:
                degree[u] += 1
                degree[v] += 1
                selected_edges.append((u, v))
                break
            else:
                continue
        else:
            su, sv = comp_size[fu], comp_size[fv]
            degree[u] += 1
            degree[v] += 1
            selected_edges.append((u, v))
            new_size = su + sv
            if su < sv:
                parent[fu] = fv
                comp_size[fv] = new_size
            else:
                parent[fv] = fu
                comp_size[fu] = new_size
    
    # Extract cycle from greedy solution
    if len(selected_edges) == n:
        adj = {i: [] for i in range(n)}
        for u, v in selected_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        cycle = [0]
        prev = -1
        cur = 0
        for _ in range(n - 1):
            nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
            cycle.append(nxt)
            prev = cur
            cur = nxt
    else:
        cycle = list(range(n))
    
    best_cycle = cycle
    best_cost = cycle_cost(best_cycle)
    
    # Step 2: Local search with 2-opt moves
    def local_search(cyc, cost, dl):
        current = list(cyc)
        cur_cost = cost
        improved = True
        while improved and time.time() < dl:
            improved = False
            for i in range(1, n - 1):
                if time.time() >= dl:
                    break
                for j in range(i + 2, n):
                    if time.time() >= dl:
                        break
                    # 2-opt: reverse segment [i..j]
                    new_cycle = current[:i] + current[i:j+1][::-1] + current[j+1:]
                    new_cost = cycle_cost(new_cycle)
                    if new_cost < cur_cost:
                        current = new_cycle
                        cur_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
        return current, cur_cost
    
    if time.time() < deadline:
        best_cycle, best_cost = local_search(best_cycle, best_cost, deadline)
    
    # Step 3: Try to find a cycle with all edges below a threshold
    # Binary search on the threshold
    lo, hi = n, best_cost - 1  # We already have a solution at best_cost
    if lo <= hi and time.time() < deadline:
        # Try to find a lower threshold
        def check_threshold(thr):
            """Check if there's a Hamilton cycle using only edges with pos <= thr."""
            # Build adjacency of allowed edges
            adj_allowed = {i: [] for i in range(n)}
            for (a, b), pos in edge_pos.items():
                if pos <= thr:
                    adj_allowed[a].append(b)
                    adj_allowed[b].append(a)
            
            # DFS with backtracking
            visited = [False] * n
            path = [0]
            visited[0] = True
            
            def dfs():
                if len(path) == n:
                    # Check if last vertex connects to first
                    last = path[-1]
                    first = path[0]
                    a, b = min(last, first), max(last, first)
                    if edge_pos.get((a, b), m + 1) <= thr:
                        return True
                    return False
                
                cur = path[-1]
                # Sort neighbors by their edge position to the unvisited set (heuristic)
                nbrs = sorted(adj_allowed[cur], key=lambda x: edge_pos.get((min(cur,x), max(cur,x)), m+1))
                
                for nb in nbrs:
                    if not visited[nb]:
                        # Prune: if nb is the first vertex and we haven't visited all, skip
                        if nb == path[0] and len(path) < n - 1:
                            continue
                        visited[nb] = True
                        path.append(nb)
                        if dfs():
                            return True
                        path.pop()
                        visited[nb] = False
                return False
            
            if dfs():
                return list(path)
            return None
        
        # Binary search for minimum threshold
        best_found = None
        best_found_cost = best_cost
        
        # Try decreasing thresholds
        for thr in range(best_cost - 1, n - 1, -1):
            if time.time() >= deadline:
                break
            cyc = check_threshold(thr)
            if cyc is not None:
                c = cycle_cost(cyc)
                if c < best_found_cost:
                    best_found_cost = c
                    best_found = cyc
                    break
        
        if best_found is not None:
            best_cycle = best_found
            best_cost = best_found_cost
    
    # Final local search polish
    if time.time() < deadline - 5:
        best_cycle, best_cost = local_search(best_cycle, best_cost, deadline - 2)
    
    return best_cycle

# EVOLVE-BLOCK-END
