import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110

    # 1. Generate vertices of PG(2, 5)
    pts = []
    for x in range(5):
        for y in range(5):
            pts.append((x, y, 1))
    for y in range(5):
        pts.append((1, y, 0))
    pts.append((0, 1, 0))
    
    n = len(pts)
    assert n == 31

    # 2. Define Polarity and build base graph
    def is_adjacent(p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        return (x1*x2 + y1*y2 + z1*z2) % 5 == 0

    adj = {i: set() for i in range(31)}
    for i in range(31):
        for j in range(i+1, 31):
            if is_adjacent(pts[i], pts[j]):
                adj[i].add(j)
                adj[j].add(i)
    
    edge_count = sum(len(s) for s in adj.values()) // 2
    assert edge_count == 90, f"Edge count is {edge_count}"

    # 3. Find the non-edge that creates the fewest C4s
    # A new C4 is created for every path of length 3 between u and v in the base graph.
    # Path u-w-x-v.
    # Count = sum_{w in N(u)} |N(w) intersect N(v)|
    
    best_edge = None
    min_c4s = float('inf')
    
    for u in range(31):
        if time.time() > deadline - 10:
            break
        for v in range(u + 1, 31):
            if v in adj[u]:
                continue
            
            # Calculate number of new C4s
            count = 0
            for w in adj[u]:
                # Count common neighbors of w and v
                # Since base graph is C4-free, this is 0 or 1
                # We can just check intersection
                # N(w) and N(v)
                # Optimization: precompute or just iterate
                # Given small size, simple loop is fine
                common = 0
                for x in adj[w]:
                    if x in adj[v]:
                        common += 1
                count += common
            
            if count < min_c4s:
                min_c4s = count
                best_edge = (u, v)
                
    # 4. Add the best edge
    if best_edge:
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
    else:
        # Fallback: add any non-edge
        for u in range(31):
            for v in range(u + 1, 31):
                if v not in adj[u]:
                    adj[u].add(v)
                    adj[v].add(u)
                    break
            else:
                continue
            break

    # 5. Construct the list of edges
    edges = []
    for i in range(31):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
                
    assert len(edges) == 91
    
    return edges
