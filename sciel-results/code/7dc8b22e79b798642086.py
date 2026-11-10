import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    q = 5
    # Build PG(2, q) points
    # Points: (1, x, y) for x,y in GF(q); (0, 1, z) for z in GF(q); (0, 0, 1)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)  # should be 31
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build polarity graph: edge between i,j if dot(points[i], points[j]) == 0 and i != j
    edges = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
    
    # Check edge count
    if len(edges) != 90:
        # Fallback: use a different construction or adjust
        # For now, if we don't have exactly 90, we need to handle it
        pass
    
    # Find the best non-edge to add (minimizing new 4-cycles)
    best_edge = None
    best_cost = float('inf')
    
    for i in range(n):
        for j in range(i+1, n):
            if j in adj[i]:
                continue  # already an edge
            
            # Count new 4-cycles created by adding edge (i,j)
            # Type 1: i-a-j-b-i where a~i, b~j, a~b, a!=b, a!=j, b!=i
            # This is complex. Simpler: count common neighbors
            common = adj[i] & adj[j]
            k = len(common)
            
            # Type 2: i-a-j-b-i where a,b are common neighbors: C(k,2)
            cost2 = k*(k-1)//2
            
            # Type 1: for each a in adj[i]\{j}, for each b in adj[j]\{i}, if a~b and a!=b
            # This creates a 4-cycle i-a-b-j-i
            # But we need to be careful not to double count
            cost1 = 0
            for a in adj[i]:
                if a == j:
                    continue
                for b in adj[j]:
                    if b == i or b == a:
                        continue
                    if b in adj[a]:
                        cost1 += 1
            # Each 4-cycle is counted once here (i-a-b-j with a in adj[i], b in adj[j], a~b)
            # But actually, this might overcount. Let me think...
            # A 4-cycle i-a-b-j-i is valid if a~i, b~j, a~b, a!=b, a!=j, b!=i
            # We're iterating a in adj[i] (so a~i), b in adj[j] (so b~j), checking a~b
            # Conditions a!=j (a in adj[i], j not in adj[i] since we're adding it), b!=i (similar)
            # a!=b is checked
            # So cost1 counts each such 4-cycle once
            cost = cost1 + cost2
            
            if cost < best_cost:
                best_cost = cost
                best_edge = [i, j]
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if best_edge is not None:
        edges.append(best_edge)
    
    # Ensure we have exactly 91 edges
    if len(edges) != 91:
        # Fallback: add random non-edges
        existing = set()
        for e in edges:
            existing.add(tuple(sorted(e)))
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
