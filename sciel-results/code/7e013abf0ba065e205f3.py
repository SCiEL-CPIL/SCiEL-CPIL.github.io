import time
import itertools

def count_c4(adj, n):
    """Count number of 4-cycles in graph given by adjacency sets."""
    count = 0
    for i in range(n):
        for j in adj[i]:
            if i < j:
                common = adj[i] & adj[j]
                for k in common:
                    for l in adj[k]:
                        if l > j and l in adj[i] and l in adj[j]:
                            pass
    # Better: count pairs of common neighbors
    count = 0
    for i in range(n):
        for j in adj[i]:
            if i < j:
                common = adj[i] & adj[j]
                count += len(common) * (len(common) - 1) // 2
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate points of PG(2,5)
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    pts = (1, (y * pow(x, 4, 5)) % 5, (z * pow(x, 4, 5)) % 5)
                elif y != 0:
                    pts = (0, 1, (z * pow(y, 4, 5)) % 5)
                else:
                    pts = (0, 0, 1)
                if pts not in points:
                    points.append(pts)
    
    n = len(points)
    
    # 2. Build polarity graph
    adj = [set() for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            b_val = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
            if b_val == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    
    if len(edges) != 90:
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
    
    # 3. Find best edge to add
    min_count = float('inf')
    best_edge = None
    
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if j in adj[i]:
                continue
            count = 0
            for x in adj[i]:
                if adj[x] & adj[j]:
                    count += 1
            if count < min_count:
                min_count = count
                best_edge = [i, j]
    
    if best_edge is None:
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
    
    # Add best edge
    edges.append(best_edge)
    adj[best_edge[0]].add(best_edge[1])
    adj[best_edge[1]].add(best_edge[0])
    
    # 4. Local search: try to improve by swapping edges
    current_c4 = count_c4(adj, n)
    best_edges = list(edges)
    best_c4 = current_c4
    
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        
        improved = False
        # Try removing one edge and adding another
        for rem_idx in range(len(edges)):
            if time.time() > deadline - 10:
                break
            
            u, v = edges[rem_idx]
            # Remove edge
            adj[u].remove(v)
            adj[v].remove(u)
            
            # Try adding each non-edge
            for i in range(n):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, n):
                    if j in adj[i]:
                        continue
                    if i == u and j == v:
                        continue
                    
                    # Temporarily add
                    adj[i].add(j)
                    adj[j].add(i)
                    
                    new_c4 = count_c4(adj, n)
                    
                    # Remove
                    adj[i].remove(j)
                    adj[j].remove(i)
                    
                    if new_c4 < best_c4:
                        best_c4 = new_c4
                        best_edges = edges[:rem_idx] + edges[rem_idx+1:] + [[i, j]]
                        improved = True
            
            # Restore removed edge
            adj[u].add(v)
            adj[v].add(u)
        
        if not improved:
            break
    
    return best_edges
