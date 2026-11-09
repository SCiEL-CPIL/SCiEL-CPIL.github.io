import time
import random
import math

def _count_c6_using_edge(u, v, adj):
    """
    Count the number of simple paths of length 4 from u to v in the graph defined by adj.
    The path is u-a-b-c-d-v.
    u and v are not in adj[u] or adj[v] respectively for the purpose of the path interior, 
    but we assume adj is the adjacency list of the graph WITHOUT the edge (u,v).
    Actually, it's safer to pass the full adjacency and just skip u and v in the loops.
    """
    count = 0
    # a must be neighbor of u, a != v
    for a in adj[u]:
        if a == v:
            continue
        # b must be neighbor of a, b != u, b != v
        for b in adj[a]:
            if b == u or b == v:
                continue
            # c must be neighbor of b, c != u, c != v, c != a
            for c in adj[b]:
                if c == u or c == v or c == a:
                    continue
                # d must be neighbor of c, d != u, d != v, d != b
                # and d must be neighbor of v
                for d in adj[c]:
                    if d == u or d == v or d == b:
                        continue
                    # Check if d is connected to v
                    if d in adj[v]:
                        count += 1
    return count

def _calculate_total_cost(n, colors, adj):
    """
    Calculate the total number of C6s in the coloring.
    colors is a list of length n*n//2, colors[i] is the color of edge (i, j) where j > i.
    We need to sum the cost for each color class.
    Cost of a color class is the number of C6s.
    A C6 is counted 6 times if we sum over edges? No.
    A C6 has 6 edges. If we sum _count_c6_using_edge for all edges in the graph, 
    each C6 is counted 6 times (once for each edge).
    So total_cost = sum(_count_c6_using_edge(u, v, adj_c)) / 6 for all (u,v) in color c.
    """
    total = 0
    # Rebuild adjacency lists for each color
    adj = [[[] for _ in range(n)] for _ in range(4)]
    # Map edge (i,j) to index
    # index = i*n - i*(i+1)//2 + j - i - 1 ? 
    # Let's just use a dict or 2D array for colors for speed in the main loop
    # For this function, we just use the passed adj if it's built.
    # But we need adj per color.
    pass # This function is not used directly in the fast loop, we maintain cost incrementally.

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Target n
    # Try n=17 first
    for n in [17, 16, 15]:
        if time.time() > deadline:
            break
            
        # Initialize random coloring
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Helper to get edge index
        def edge_idx(i, j):
            if i > j: i, j = j, i
            return i * n - i * (i + 1) // 2 + j - i - 1

        # Build adjacency lists for each color
        adj = [[[] for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_idx(i, j)]
                adj[c][i].append(j)
                adj[c][j].append(i)
        
        # Calculate initial cost
        # cost[c] = number of C6s in color c
        # We sum counts over all edges in color c, then divide by 6
        cost = [0] * 4
        for c in range(4):
            sum_counts = 0
            for i in range(n):
                for j in adj[c][i]:
                    if i < j: # Only count each edge once
                        # We need adj without edge (i,j) for the count?
                        # _count_c6_using_edge skips u and v in the interior, 
                        # but it uses adj[u] which includes v.
                        # The logic in _count_c6_using_edge:
                        # for a in adj[u]: if a==v: continue.
                        # So it effectively ignores the edge (u,v) for the first step.
                        # Does it ignore it for the last step?
                        # if d in adj[v]: count += 1.
                        # If d==u, it's skipped earlier.
                        # So it counts paths u-a-b-c-d-v where (u,v) is not used in the path interior.
                        # This is correct for counting C6s containing edge (u,v).
                        sum_counts += _count_c6_using_edge(i, j, adj[c])
            cost[c] = sum_counts // 6
        
        total_cost = sum(cost)
        if total_cost == 0:
            return {'n': n, 'colors': colors}
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        steps = 0
        max_steps = 200000
        
        best_cost = total_cost
        best_colors = colors.copy()
        
        while time.time() < deadline and steps < max_steps:
            steps += 1
            
            # Pick random edge
            e = random.randint(0, num_edges - 1)
            # Find i, j from e
            # Inverse of edge_idx
            # e = i*n - i*(i+1)//2 + j - i - 1
            # This is tricky to invert directly. 
            # Better to pick i, j directly.
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = colors[edge_idx(i, j)]
            new_c = random.randint(0, 3)
            if old_c == new_c:
                continue
            
            # Calculate delta cost
            # Remove edge (i,j) from old_c, add to new_c
            # Delta cost for old_c: - (count of C6s using (i,j) in old_c) / 6
            # Delta cost for new_c: + (count of C6s using (i,j) in new_c) / 6
            # Note: The count _count_c6_using_edge(i, j, adj) assumes the edge is NOT present?
            # My implementation:
            # for a in adj[u]: if a==v: continue.
            # So it ignores the edge (u,v) in the adjacency list for the start.
            # But for the end: if d in adj[v].
            # If the edge (u,v) is in adj, and d==u, it's skipped.
            # So it correctly counts C6s that USE the edge (u,v) as the closing edge?
            # No, the path is u-a-b-c-d-v. The edge (u,v) is NOT part of the path u-a-b-c-d-v.
            # The C6 is u-a-b-c-d-v-u.
            # So the edge (u,v) IS part of the cycle.
            # My function counts paths from u to v of length 4.
            # Adding the edge (u,v) closes the cycle.
            # So the count returned is the number of C6s that contain the edge (u,v).
            # This is correct.
            
            # However, when we change the color, the adjacency lists change.
            # For old_c, we remove (i,j). The count of C6s using (i,j) in old_c 
            # is calculated on the graph WITH (i,j) present?
            # My function ignores (i,j) in the path interior, so it works for the graph WITH (i,j).
            # Wait, if (i,j) is removed, do we lose any C6s that don't use (i,j)? No.
            # Do we gain any? No.
            # So the number of C6s in old_c that DO NOT use (i,j) remains the same.
            # The number of C6s in old_c that DO use (i,j) is removed.
            # So delta_old = - count_c6_using_edge(i, j, adj[old_c]) / 6.
            
            # For new_c, we add (i,j).
            # The number of C6s in new_c that DO NOT use (i,j) remains the same.
            # The number of C6s in new_c that DO use (i,j) is added.
            # So delta_new = + count_c6_using_edge(i, j, adj[new_c]) / 6.
            # BUT, the count for new_c must be calculated on the graph BEFORE adding (i,j).
            # My function ignores (i,j) in the path, so it works on the graph WITHOUT (i,j) as well.
            # So we can just use the current adj lists.
            
            cnt_old = _count_c6_using_edge(i, j, adj[old_c])
            cnt_new = _count_c6_using_edge(i, j, adj[new_c])
            
            delta = (cnt_new - cnt_old) / 6.0
            
            # Acceptance
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                colors[edge_idx(i, j)] = new_c
                # Update adj lists
                adj[old_c][i].remove(j)
                adj[old_c][j].remove(i)
                adj[new_c][i].append(j)
                adj[new_c][j].append(i)
                
                total_cost += delta
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_colors = colors.copy()
                    
                if best_cost == 0:
                    return {'n': n, 'colors': best_colors}
            
            T *= cooling
    
    # Fallback to seed
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
