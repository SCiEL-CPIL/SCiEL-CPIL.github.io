import random
import time
from collections import deque

def construct(k, g):
    # ... LUW code ...
    
    # New: Randomized Search for Circulant
    start_time = time.time()
    timeout = 140.0
    
    # Moore bound approx
    # n_min = 426 for k=5, g=9
    
    # Try circulant graphs
    # Degree k=5 requires n even and S = {s1, n-s1, s2, n-s2, n/2}
    # We choose s1, s2, s3, s4 from 1..n/2-1 distinct.
    
    for n in range(500, 2000, 50):
        if time.time() - start_time > timeout:
            break
            
        if n % 2 != 0:
            continue
            
        # Try random S
        found = False
        for _ in range(50): # 50 attempts per n
            if time.time() - start_time > timeout:
                break
                
            # Pick 4 distinct random steps
            steps = random.sample(range(1, n//2), 4)
            S = set(steps + [n - s for s in steps] + [n//2])
            
            # Check simple
            if len(S) != 5: continue # Should be 5
            
            # Check Girth
            # Build adj list
            adj = [[] for _ in range(n)]
            for i in range(n):
                for s in S:
                    j = (i + s) % n
                    adj[i].append(j)
            
            # Verify degree
            if any(len(a) != k for a in adj): continue
            
            # Girth Check
            girth_ok = True
            for u in range(n):
                # BFS
                dist = [-1] * n
                dist[u] = 0
                q = deque([u])
                while q:
                    x = q.popleft()
                    for y in adj[x]:
                        if dist[y] == -1:
                            dist[y] = dist[x] + 1
                            q.append(y)
                        elif y != u: # y is visited, not parent
                            # Cycle length dist[x] + dist[y] + 1
                            # Actually, if y is visited, and y is not parent of x
                            # In BFS tree, if edge (x,y) is not in tree, cycle is dist[x] + dist[y] + 1
                            # But we need to be careful.
                            # Simpler: if dist[y] < dist[x], then y is parent?
                            # If dist[y] == dist[x] - 1, it's parent.
                            # If dist[y] >= dist[x], it's a cross edge or back edge.
                            # Cycle length = dist[x] + dist[y] + 1.
                            cycle_len = dist[x] + dist[y] + 1
                            if cycle_len < g:
                                girth_ok = False
                                break
                    if not girth_ok: break
                if not girth_ok: break
            
            if girth_ok:
                # Convert to edge list
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
    # Fallback to LUW
    # ... LUW implementation ...
