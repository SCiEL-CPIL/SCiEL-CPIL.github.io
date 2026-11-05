# EVOLVE-BLOCK-START
import random
import time

def _path_count_depth4(start, end, adj, n, forbidden_edge=None):
    """
    Count simple paths of length 4 from start to end in graph adj.
    If forbidden_edge is (u, v), the path cannot use that edge.
    Path: start -> x -> y -> z -> end.
    """
    if forbidden_edge:
        u, v = forbidden_edge
        # If start==u and end==v or start==v and end==u, we just need to avoid that direct edge, 
        # but since we are looking for length 4, the direct edge is not used in the path steps 
        # unless the path is length 1. Here length is 4, so the edge (start, end) is not part of the path 
        # unless start/end are intermediate, which they are not.
        # Wait, the forbidden edge is the one being added/removed. 
        # If we are checking if adding (u,v) creates a C6, we look for path v -> ... -> u of length 4.
        # The path itself does not contain the edge (u,v) because it's length 4.
        # So forbidden_edge logic is only needed if we are checking for shorter cycles or if the path 
        # could use the edge? No, a simple path of length 4 from v to u does not use the edge (v,u) 
        # because that would be length 1.
        # So we just need to count simple paths of length 4.
        pass
    
    count = 0
    # DFS from start, depth 0
    # We need to reach end at depth 4.
    # Visited set to ensure simple path.
    
    def dfs(node, depth, visited):
        nonlocal count
        if depth == 4:
            if node == end:
                count += 1
            return
        
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                dfs(neighbor, depth + 1, visited)
                visited.remove(neighbor)
                
    visited = {start}
    dfs(start, 0, visited)
    return count

def _count_c6s_for_edge(u, v, adj, n):
    """
    Count number of C6s that contain the edge (u, v).
    A C6 containing (u, v) corresponds to a path of length 4 from v to u in G \ {(u,v)}.
    Since the path is length 4, it doesn't use (u,v) anyway.
    """
    return _path_count_depth4(v, u, adj, n)

def _total_c6_count_fast(colors, n):
    """
    Calculate total number of monochromatic C6s.
    Sum over all edges (u,v) of count_C6s_involving(u,v) / 6? 
    No, each C6 has 6 edges. If we sum counts for all edges, each C6 is counted 6 times.
    So total C6s = (1/6) * sum_{(u,v)} _count_c6s_for_edge(u, v, G_c)
    """
    adj = [[] for _ in range(n)]
    # Build adjacency lists for each color
    adj_by_color = [[[] for _ in range(n)] for _ in range(4)]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj_by_color[c][i].append(j)
            adj_by_color[c][j].append(i)
            idx += 1
            
    total_c6s = 0
    for c in range(4):
        g_adj = adj_by_color[c]
        # Sum over all edges in this color
        for i in range(n):
            for j in g_adj[i]:
                if i < j: # Count each edge once
                    cnt = _path_count_depth4(j, i, g_adj, n)
                    total_c6s += cnt
                    
    return total_c6s // 6

def construct():
    """
    Find a 4-coloring of K_n with no monochromatic C6, maximizing n.
    Uses local search with fast C6 counting via edge-based path counting.
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Try n=17 first, then 16, etc.
    for n in [17, 16, 15, 14, 13]:
        m = n * (n - 1) // 2
        
        # Try multiple random restarts
        for restart in range(20):
            if time.time() - start_time > time_limit:
                break
                
            random.seed(42 + restart)
            colors = [random.randint(0, 3) for _ in range(m)]
            
            # Initial evaluation
            current_c6s = _total_c6_count_fast(colors, n)
            
            if current_c6s == 0:
                return {'n': n, 'colors': colors}
                
            # Local Search
            best_colors = colors[:]
            best_c6s = current_c6s
            
            # Precompute edge indices
            edge_indices = {}
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    edge_indices[(i, j)] = idx
                    idx += 1
                    
            max_iter = 5000
            for it in range(max_iter):
                if time.time() - start_time > time_limit:
                    break
                    
                # Pick a random edge to flip
                e_idx = random.randint(0, m - 1)
                u = -1
                v = -1
                # Find u, v for e_idx
                # To make this fast, we can precompute a list of (u,v) for each index
                # Or just iterate. n<=17, m<=136. Iteration is fine.
                k = e_idx
                for i in range(n):
                    for j in range(i + 1, n):
                        if k == 0:
                            u, v = i, j
                            break
                        k -= 1
                    if u != -1:
                        break
                            
                old_c = colors[e_idx]
                new_c = random.randint(0, 3)
                if new_c == old_c:
                    continue
                    
                # Calculate delta
                # 1. Remove edge (u,v) from color old_c
                # 2. Add edge (u,v) to color new_c
                
                # We need to count C6s removed and added.
                # C6s removed: C6s in G_old_c that contain (u,v).
                # C6s added: C6s in G_new_c + (u,v) that contain (u,v).
                
                # To do this efficiently, we need the current adjacency lists.
                # Maintaining them dynamically is complex. 
                # Alternative: Recalculate total C6s? 
                # _total_c6_count_fast is O(m * d^4). For n=17, m=136, d~8. 136 * 4096 ~ 500k ops.
                # 5000 iters * 500k = 2.5 billion ops. Too slow in Python.
                
                # We need a faster delta calculation.
                # Let's maintain adj_by_color.
                
                # Let's restart the loop with a maintained state.
                break # Break inner loop to implement maintained state version
                
            # If we broke out of inner loop, implement maintained state search
            # For simplicity in this edit, let's just do the naive re-calculation but limit iterations
            # Or better, implement the maintained state properly.
            
            # --- Maintained State Search ---
            adj_by_color = [[[] for _ in range(n)] for _ in range(4)]
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[edge_indices[(i, j)]]
                    adj_by_color[c][i].append(j)
                    adj_by_color[c][j].append(i)
                    
            # Initial count
            current_c6s = _total_c6_count_fast(colors, n)
            
            for it in range(2000): # Reduced iterations due to higher per-iter cost? No, delta is cheaper.
                if time.time() - start_time > time_limit:
                    break
                    
                if current_c6s == 0:
                    return {'n': n, 'colors': colors}
                    
                e_idx = random.randint(0, m - 1)
                u = -1
                v = -1
                k = e_idx
                for i in range(n):
                    for j in range(i + 1, n):
                        if k == 0:
                            u, v = i, j
                            break
                        k -= 1
                    if u != -1:
                        break
                            
                old_c = colors[e_idx]
                new_c = random.randint(0, 3)
                if new_c == old_c:
                    continue
                    
                # 1. Remove (u,v) from old_c
                # Find index of v in adj_by_color[old_c][u] and remove
                if v in adj_by_color[old_c][u]:
                    adj_by_color[old_c][u].remove(v)
                if u in adj_by_color[old_c][v]:
                    adj_by_color[old_c][v].remove(u)
                
                # Count C6s lost: C6s in old_c containing (u,v)
                lost_c6s = _path_count_depth4(v, u, adj_by_color[old_c], n)
                
                # 2. Add (u,v) to new_c
                adj_by_color[new_c][u].append(v)
                adj_by_color[new_c][v].append(u)
                
                # Count C6s gained: C6s in new_c containing (u,v)
                gained_c6s = _path_count_depth4(v, u, adj_by_color[new_c], n)
                
                delta = gained_c6s - lost_c6s
                new_c6s = current_c6s + delta
                
                if new_c6s <= current_c6s:
                    # Accept
                    colors[e_idx] = new_c
                    current_c6s = new_c6s
                    if current_c6s == 0:
                        return {'n': n, 'colors': colors}
                else:
                    # Revert
                    # Remove from new_c
                    if v in adj_by_color[new_c][u]:
                        adj_by_color[new_c][u].remove(v)
                    if u in adj_by_color[new_c][v]:
                        adj_by_color[new_c][v].remove(u)
                    # Add to old_c
                    adj_by_color[old_c][u].append(v)
                    adj_by_color[old_c][v].append(u)

    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
# EVOLVE-BLOCK-END
