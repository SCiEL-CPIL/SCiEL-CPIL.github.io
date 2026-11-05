import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
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
    
    # Build adjacency matrix for fast C4 counting
    # adj_mat[i][j] = 1 if edge (i,j) exists
    adj_mat = np.zeros((n, n), dtype=np.int32)
    for e in edges:
        u, v = e[0], e[1]
        adj_mat[u][v] = 1
        adj_mat[v][u] = 1
    
    def count_total_c4_fast(adj_mat):
        """Count all C4s using matrix operations. Each C4 is counted 4 times (once per vertex pair as opposite corners)."""
        # Number of C4s = sum over all pairs (u,v) of C(common_neighbors(u,v), 2) / 2
        # Actually: for each pair (u,v), the number of C4s where u,v are opposite corners
        # is C(|N(u) ∩ N(v)|, 2). Each C4 has 2 pairs of opposite corners, so divide by 2.
        # Total C4s = (1/2) * sum_{u < v} C(|N(u) ∩ N(v)|, 2)
        # But wait, this counts each C4 twice (once for each pair of opposite corners).
        # Actually each C4 v1-v2-v3-v4-v1 has opposite pairs (v1,v3) and (v2,v4).
        # So each C4 is counted exactly twice. Hence divide by 2.
        
        total = 0
        # For each pair (u,v), compute common neighbors
        # common = adj_mat[u] & adj_mat[v] (element-wise)
        # count = sum(common)
        # c4s_for_pair = count * (count-1) // 2
        for u in range(n):
            for v in range(u+1, n):
                common = np.sum(adj_mat[u] & adj_mat[v])
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2
    
    def count_c4s_involving_edge(u, v, adj_mat):
        """Count C4s that include edge (u,v). 
        A C4 containing edge (u,v) looks like u-v-a-b-u where a~v, b~u, a~b, a!=b, a!=u, b!=v.
        So we need pairs (a,b) where a in N(v)\\{u}, b in N(u)\\{v}, a != b, and a~b."""
        count = 0
        # For each a in N(v) \ {u}, for each b in N(u) \ {v}, check if a~b and a != b
        nu = set(np.where(adj_mat[u] == 1)[0]) - {v}
        nv = set(np.where(adj_mat[v] == 1)[0]) - {u}
        for a in nv:
            for b in nu:
                if a == b:
                    continue
                if adj_mat[a][b] == 1:
                    count += 1
        return count
    
    # Ensure we have exactly 91 edges
    if m < 91:
        # Greedily add edges with minimum C4 contribution
        edge_set = set()
        for e in edges:
            edge_set.add((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]))
        
        while len(edges) < 91:
            best_cost = float('inf')
            best_edge = None
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) not in edge_set:
                        # Temporarily add edge and count C4s
                        adj_mat[i][j] = 1
                        adj_mat[j][i] = 1
                        cost = count_c4s_involving_edge(i, j, adj_mat)
                        adj_mat[i][j] = 0
                        adj_mat[j][i] = 0
                        if cost < best_cost:
                            best_cost = cost
                            best_edge = (i, j)
            
            if best_edge:
                u, v = best_edge
                edges.append([u, v])
                adj[u].add(v)
                adj[v].add(u)
                adj_mat[u][v] = 1
                adj_mat[v][u] = 1
                edge_set.add((u, v))
            else:
                break

    elif m > 91:
        # Greedily remove edges with highest C4 participation
        while len(edges) > 91:
            max_part = -1
            max_edge = None
            for e in edges:
                u, v = e[0], e[1]
                part = count_c4s_involving_edge(u, v, adj_mat)
                if part > max_part:
                    max_part = part
                    max_edge = e
            
            if max_edge:
                u, v = max_edge[0], max_edge[1]
                edges.remove(max_edge)
                adj[u].remove(v)
                adj[v].remove(u)
                adj_mat[u][v] = 0
                adj_mat[v][u] = 0
            else:
                break

    assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
    
    # Local Search with incremental C4 counting
    current_adj = [set(s) for s in adj]
    current_adj_mat = adj_mat.copy()
    current_score = count_total_c4_fast(current_adj_mat)
    
    best_adj_mat = current_adj_mat.copy()
    best_score = current_score
    
    # SA Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    iteration = 0
    while time.time() < deadline - 3.0:
        iteration += 1
        
        # Generate random move: remove edge (u,v), add edge (x,y)
        u = random.randint(0, n-1)
        if not current_adj[u]:
            continue
        v = random.choice(list(current_adj[u]))
        
        x = random.randint(0, n-1)
        y = random.randint(0, n-1)
        if x == y:
            continue
        if y in current_adj[x]:
            continue
        
        # Check if this is a valid swap (not removing the same edge we're adding)
        if (u == x and v == y) or (u == y and v == x):
            continue
        
        # Save state for rollback
        old_adj_mat = current_adj_mat.copy()
        
        # Remove (u,v)
        current_adj[u].remove(v)
        current_adj[v].remove(u)
        current_adj_mat[u][v] = 0
        current_adj_mat[v][u] = 0
        
        # Add (x,y)
        current_adj[x].add(y)
        current_adj[y].add(x)
        current_adj_mat[x][y] = 1
        current_adj_mat[y][x] = 1
        
        # Compute new score
        new_score = count_total_c4_fast(current_adj_mat)
        
        # Acceptance
        if new_score < best_score:
            best_score = new_score
            best_adj_mat = current_adj_mat.copy()
            
        if new_score <= current_score or (T > T_min and random.random() < np.exp((current_score - new_score) / T)):
            current_score = new_score
        else:
            # Reject move, revert
            current_adj_mat = old_adj_mat
            current_adj[u].add(v)
            current_adj[v].add(u)
            current_adj[x].remove(y)
            current_adj[y].remove(x)
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Convert best_adj_mat back to edges list
    best_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj_mat[u][v] == 1:
                best_edges.append([u, v])
                
    assert len(best_edges) == 91, f"Expected 91 edges, got {len(best_edges)}"
    
    return best_edges
