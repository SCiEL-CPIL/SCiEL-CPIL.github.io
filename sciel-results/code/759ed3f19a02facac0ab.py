import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n = 31
    q = 5
    mod = q
    
    # Enumerate all points in PG(2, q)
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    adj = [set() for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % mod == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    
    m = len(edges)
    
    # Ensure we have exactly 91 edges
    # The polarity graph has 90 edges. We need to add 1.
    # We pick the edge that adds the fewest C4s.
    
    if m < 91:
        edge_set = set()
        for e in edges:
            edge_set.add((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]))
        
        best_cost = float('inf')
        best_edge = None
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edge_set:
                    # Count C4s formed by adding (i, j)
                    # A C4 is i-w-x-j-i.
                    # w in N(i), x in N(j), (w,x) in E.
                    count = 0
                    for w in adj[i]:
                        for x in adj[j]:
                            if w == x: continue
                            if x in adj[w]:
                                count += 1
                    # This counts each C4 twice?
                    # i-w-x-j-i.
                    # w is neighbor of i. x is neighbor of j.
                    # Edge (w,x) exists.
                    # Is it counted once?
                    # For a specific C4 i-w-x-j-i, w is fixed, x is fixed.
                    # Yes, this loop counts each C4 exactly once.
                    if count < best_cost:
                        best_cost = count
                        best_edge = (i, j)
        
        if best_edge:
            u, v = best_edge
            edges.append([u, v])
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
            
    # Initialize fast local search state
    # common[i][j] = |N(i) intersect N(j)|
    common = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i+1, n):
            common[i][j] = len(adj[i] & adj[j])
            common[j][i] = common[i][j]
    
    # SumC4 = 2 * TotalC4s
    # TotalC4s = 0.5 * sum_{i<j} C(common[i][j], 2)
    sum_c4 = 0
    for i in range(n):
        for j in range(i+1, n):
            c = common[i][j]
            if c >= 2:
                sum_c4 += c * (c - 1)
                
    best_sum_c4 = sum_c4
    best_adj = [set(s) for s in adj]
    
    # Fast Local Search
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9999
    
    def delta_sum_c4_remove(u, v):
        # Remove edge (u, v)
        # Delta SumC4 = sum_{k in N(v)\{u}} (1 - common[u][k]) + sum_{k in N(u)\{v}} (1 - common[v][k])
        d = 0
        for k in adj[v]:
            if k != u:
                d += (1 - common[u][k])
        for k in adj[u]:
            if k != v:
                d += (1 - common[v][k])
        return d

    def apply_remove(u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        # Update common
        for k in adj[v]:
            if k != u:
                common[u][k] -= 1
                common[k][u] -= 1
        for k in adj[u]:
            if k != v:
                common[v][k] -= 1
                common[k][v] -= 1
        # common[u][v] is unchanged (as derived)
        
    def apply_add(u, v):
        adj[u].add(v)
        adj[v].add(u)
        # Update common
        for k in adj[v]:
            if k != u:
                common[u][k] += 1
                common[k][u] += 1
        for k in adj[u]:
            if k != v:
                common[v][k] += 1
                common[k][v] += 1

    def delta_sum_c4_add(u, v):
        # Add edge (u, v)
        # Delta SumC4 = sum_{k in N(v)\{u}} common[u][k] + sum_{k in N(u)\{v}} common[v][k]
        # Note: This must be calculated BEFORE adding the edge, using current common.
        d = 0
        for k in adj[v]:
            if k != u:
                d += common[u][k]
        for k in adj[u]:
            if k != v:
                d += common[v][k]
        return d

    while time.time() < deadline - 2.0:
        # Pick random edge (u, v)
        u = random.randint(0, n-1)
        if not adj[u]:
            continue
        v = random.choice(list(adj[u]))
        
        # Pick random non-edge (x, y)
        x = random.randint(0, n-1)
        y = random.randint(0, n-1)
        if x == y:
            continue
        if y in adj[x]:
            continue
            
        # Calculate delta
        d_rem = delta_sum_c4_remove(u, v)
        
        # Apply remove
        apply_remove(u, v)
        
        # Calculate delta add (on modified graph)
        d_add = delta_sum_c4_add(x, y)
        
        # Apply add
        apply_add(x, y)
        
        delta_total = d_rem + d_add
        new_sum_c4 = sum_c4 + delta_total
        
        # Acceptance
        if new_sum_c4 < best_sum_c4:
            best_sum_c4 = new_sum_c4
            best_adj = [set(s) for s in adj]
            
        if new_sum_c4 <= sum_c4 or random.random() < np.exp((sum_c4 - new_sum_c4) / T):
            sum_c4 = new_sum_c4
        else:
            # Revert
            # Revert add
            apply_remove(x, y)
            # Revert remove (which is add)
            apply_add(u, v)
            
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Convert best_adj to edges
    best_edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                best_edges.append([u, v])
                
    assert len(best_edges) == 91, f"Expected 91 edges, got {len(best_edges)}"
    
    return best_edges
