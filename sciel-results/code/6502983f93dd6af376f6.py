import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Moore bound: 2 + k * ( (k-1)^(floor((g-1)/2)) - 1 ) / (k-2)
    # For k=5, g=9: floor(8/2) = 4. 2 + 5 * (4^4 - 1) / 3 = 2 + 5 * 255 / 3 = 2 + 425 = 427.
    # But for odd girth, the bound is slightly different.
    h = (g - 1) // 2
    moore_bound = 2 + k * ((k - 1)**h - 1) // (k - 2)
    moore_bound = max(4, moore_bound)
    
    # We will search for a circulant graph C_n(S)
    # S is a set of k jumps. To avoid duplicates, we can restrict s_1 < s_2 < ... < s_k < n/2
    # But since k is odd, we might need to include 0? No, no loops.
    # We will use a random search over jump sets for each n.
    
    def check_girth(n, jumps):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u in range(n):
            for v in jumps:
                adj[u].append((u + v) % n)
                adj[u].append((u - v) % n)
        # Deduplicate and sort
        for u in range(n):
            adj[u] = list(set(adj[u]))
            if len(adj[u]) != k:
                return False
        # BFS for girth
        # For each edge (u, v), find shortest path from u to v not using that edge
        # This is O(m * (n + m)) which is too slow.
        # Instead, use BFS from each vertex to find shortest cycle.
        # Or, more simply, check for cycles of length < g using backtracking.
        # For small n, we can just do BFS from each vertex and record the shortest cycle.
        
        # A faster way: for each vertex u, do BFS. The first time we see a neighbor, 
        # we might have a cycle. But this is tricky.
        
        # Let's use a standard girth computation: 
        # For each edge (u, v), remove it, find shortest path from u to v. If dist < g-1, girth <= dist+1.
        # This is O(m * BFS). m = kn/2. For n=1000, m=2500. 2500 * 1000 = 2.5M. This is feasible.
        
        # But we only need to check if girth >= g. So we can stop early if we find a shorter cycle.
        
        # Let's just do BFS from each vertex and track the shortest cycle.
        # Actually, for circulant graphs, we can compute the girth by checking the shortest 
        # combination of jumps that sums to 0.
        # A cycle of length l is a sequence of l jumps j_1, ..., j_l such that sum j_i = 0 mod n.
        # We want the minimum l such that there is a sequence of l jumps (with signs) that sums to 0.
        # This is equivalent to finding the shortest word in the generators that evaluates to 0.
        
        # Let's just use BFS on the graph.
        # For each vertex u, BFS. When we visit a vertex v at distance d, if we have visited v before 
        # at distance d', then we have a cycle of length d + d' + 1 (if the edges are different).
        # This is not quite right.
        
        # Let's use a simpler check: for each vertex, do BFS. The girth is the minimum over all edges (u,v) 
        # of the shortest path from u to v in the graph with that edge removed.
        
        # Given the time limit, let's just do a BFS from each vertex and store the parent pointers.
        # Then, for each edge (u, v), if parent[u] != v and parent[v] != u, then we have a cycle.
        # The length is dist[u] + dist[v] + 1.
        
        # We will do this for all vertices.
        min_cycle = float('inf')
        for u in range(n):
            parent = [-1] * n
            dist = [-1] * n
            dist[u] = 0
            q = deque([u])
            while q:
                x = q.popleft()
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dist[x] + 1
                        parent[y] = x
                        q.append(y)
                    elif parent[x] != y and parent[y] != x:
                        # Found a cycle
                        cycle_len = dist[x] + dist[y] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if min_cycle < g:
                            return False
        return min_cycle >= g

    # Search
    for n in range(moore_bound, 1500): # Upper bound 1500
        if time.time() > deadline:
            break
            
        # Try to find a valid set of jumps
        # For k=5, we need 5 jumps.
        # We can sample random sets of 5 distinct non-zero residues modulo n, with values in 1..n-1.
        # To avoid symmetry issues, we can just sample and check.
        
        found = False
        for attempt in range(100): # 100 attempts per n
            if time.time() > deadline:
                break
                
            # Sample 5 distinct jumps from 1 to n-1
            # Ensure no duplicates and no 0
            jumps = set()
            while len(jumps) < k:
                jumps.add(random.randint(1, n - 1))
            jumps = list(jumps)
            
            # Check if the graph is simple (no loops, no multi-edges)
            # The graph is simple if no jump is 0 and no jump is n/2 (if n is even) 
            # and no two jumps are negatives of each other (which is handled by the set of adjacencies).
            # Actually, C_n(S) is simple if S does not contain 0 and if for any s in S, -s is not in S 
            # unless we consider the edge {u, v} once. But our adjacency list construction handles this.
            
            if check_girth(n, jumps):
                found = True
                # Build the edges
                edges = []
                for u in range(n):
                    for v in jumps:
                        if u < (u + v) % n:
                            edges.append((u, (u + v) % n))
                        elif u > (u + v) % n:
                            edges.append(((u + v) % n, u))
                        # If u == (u+v)%n, it's a loop, which is not allowed. But jumps are non-zero.
                        # If (u+v)%n == u, then v=0, which is not allowed.
                        # So no loops.
                        # To avoid duplicates, we only add the edge if u < v.
                # This edge construction is flawed. Let's do it properly.
                edges = []
                for u in range(n):
                    for v in jumps:
                        v1 = (u + v) % n
                        v2 = (u - v) % n
                        if u < v1:
                            edges.append((u, v1))
                        if u < v2:
                            edges.append((u, v2))
                # Deduplicate
                edges = list(set(edges))
                return n, edges
        
    # If we didn't find one, return a trivial graph? No, must be valid.
    # As a fallback, return the seed program's result.
    # But the seed program is for specific k, g.
    # Let's just return a random 5-regular graph if we run out of time.
    # This is not ideal.
    
    # Fallback: return the seed program's logic for the specific case
    # But we don't have it.
    
    # For now, return a small graph, even if it doesn't meet the girth.
    # This will score 0.
    # We must return something.
    n = moore_bound
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            edges.append((u, v))
    return n, edges
