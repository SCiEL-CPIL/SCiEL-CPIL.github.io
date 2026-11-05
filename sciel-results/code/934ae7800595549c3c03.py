import random
from collections import deque

def construct(k, g):
    # Specialized for k=3, g=16
    if k == 3 and g == 16:
        # Search for a 3-regular circulant graph with girth >= 16
        # N must be even for bipartite (girth even)
        # Start from Moore bound 512
        for N in range(512, 1500, 2):
            # Try random step sets
            for _ in range(500):
                # Pick 3 distinct odd steps
                steps = random.sample(range(1, N//2, 2), 3)
                # Ensure no step is its own inverse (N/2 is even if N%4==0, odd if N%4==2)
                # If N%4==2, N/2 is odd. We should avoid N/2 to keep degree 3 with 3 distinct steps?
                # If we use N/2, it's 1 edge. We need 2 more steps.
                # Let's stick to steps < N/2 to be safe.
                
                # Check if steps are distinct and not inverses
                # Since all are < N/2, they are distinct from their inverses.
                # Just need s1 != s2 != s3.
                
                # Build adjacency
                adj = [[] for _ in range(N)]
                for u in range(N):
                    for s in steps:
                        v = (u + s) % N
                        adj[u].append(v)
                        # Since undirected, we add both ways?
                        # No, if we iterate all u, we add u->v and v->u later.
                        # But we need to ensure no multi-edges.
                        # If s1 + s2 = 0 mod N, then u+s1 = u-s2.
                        # Since s_i are distinct and < N/2, s_i != -s_j.
                        # So edges are unique.
                        
                # Check simple graph: no loops (s != 0), no multi-edges.
                # s != 0 is true.
                # Multi-edges if s_i = s_j or s_i = -s_j.
                # We ensured distinct s_i < N/2.
                
                # Compute Girth via BFS from 0
                # dist[v] = shortest distance from 0
                # parent[v] = previous node
                dist = [-1] * N
                parent = [-1] * N
                q = deque([0])
                dist[0] = 0
                girth = float('inf')
                
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif parent[u] != v: # Not backtracking
                            # Found a cycle
                            # Cycle length = dist[u] + dist[v] + 1
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < girth:
                                girth = cycle_len
                            # We can break early if girth < g
                            if girth < g:
                                break
                    if girth < g:
                        break
                        
                if girth >= g:
                    # Found a valid graph
                    edges = []
                    for u in range(N):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return N, edges
        # Fallback: LUW construction
        # ... (copy seed logic)
        
    # General fallback
    # ...
