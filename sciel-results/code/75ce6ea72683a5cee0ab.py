import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    adj = np.zeros((n, n), dtype=np.int8)
    edges = []
    
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edges.append([i, j])
                adj[i, j] = 1
                adj[j, i] = 1
    
    # If we don't have exactly 90 edges, adjust to get close
    if len(edges) < 90:
        existing = set(tuple(sorted(e)) for e in edges)
        while len(edges) < 90 and time.time() < deadline:
            best_edge = None
            best_cost = float('inf')
            for i in range(n):
                for j in range(i+1, n):
                    if (i,j) in existing:
                        continue
                    # Heuristic: count common neighbors as proxy for 4-cycles
                    common = np.sum(adj[i] & adj[j])
                    if common < best_cost:
                        best_cost = common
                        best_edge = [i, j]
            if best_edge:
                edges.append(best_edge)
                i, j = best_edge
                adj[i, j] = 1
                adj[j, i] = 1
                existing.add(tuple(sorted(best_edge)))
            else:
                break
    
    elif len(edges) > 90:
        # Remove excess edges (keep first 90)
        edges = edges[:90]
        adj = np.zeros((n, n), dtype=np.int8)
        for e in edges:
            i, j = e
            adj[i, j] = 1
            adj[j, i] = 1
    
    # Now add one more edge to get 91, minimizing new 4-cycles
    best_edge = None
    best_cost = float('inf')
    
    neighbors = [list(np.where(adj[i] == 1)[0]) for i in range(n)]
    
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i+1, n):
            if adj[i, j]:
                continue
            
            count = 0
            for a in neighbors[i]:
                if a == j:
                    continue
                for b in neighbors[j]:
                    if b == i or b == a:
                        continue
                    if adj[a, b]:
                        count += 1
            
            if count < best_cost:
                best_cost = count
                best_edge = [i, j]
            
            if time.time() > deadline:
                break
    
    if best_edge is not None:
        edges.append(best_edge)
    
    # Ensure exactly 91 edges
    if len(edges) != 91:
        existing = set(tuple(sorted(e)) for e in edges)
        for i in range(n):
            for j in range(i+1, n):
                if len(edges) >= 91:
                    break
                if (i,j) not in existing:
                    edges.append([i,j])
                    existing.add((i,j))
            if len(edges) >= 91:
                break
    
    return edges[:91]
