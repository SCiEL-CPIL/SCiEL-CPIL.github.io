import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    r = g // 2
    if g % 2 == 0:
        n_min = 2 * sum((k - 1) ** i for i in range(r))
    else:
        n_min = (k - 1) * sum((k - 1) ** i for i in range(r))
    
    if n_min % 2 != 0:
        n_min += 1
        
    # We will search n from n_min upwards
    # We limit the search space to avoid infinite loops
    max_n = 1500
    
    for n in range(n_min, max_n + 1, 2):
        if time.time() > deadline:
            break
            
        # Number of attempts per n
        # Reduce attempts for larger n if time is tight
        attempts = 50 if n < 600 else 20
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
                
            adj = [set() for _ in range(n)]
            deg = [0] * n
            target_edges = (n * k) // 2
            edges_added = 0
            
            # Use a list of vertices with available degree to speed up selection?
            # Simple random selection is O(1) per pick, but might have high collision rate.
            # For n=500, k=3, it's fine.
            
            while edges_added < target_edges:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                
                if u == v or deg[u] >= k or deg[v] >= k:
                    continue
                if v in adj[u]:
                    continue
                
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges_added += 1
                
            if any(d != k for d in deg):
                continue
                
            # Check Girth
            valid = True
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                queue = deque([start])
                dist[start] = 0
                parent[start] = -1
                
                while queue and valid:
                    u = queue.popleft()
                    if dist[u] >= g // 2:
                        # We can prune here? 
                        # If dist[u] >= g//2, any cycle found through u 
                        # involving a back edge to a node with dist >= 0 
                        # will be length >= dist[u] + dist[v] + 1.
                        # If dist[v] >= 1, length >= dist[u] + 1 + 1 > g.
                        # So we only need to check edges to nodes with small dist.
                        # But BFS ensures we find shortest paths.
                        # If we haven't found a short cycle yet, and we are deep, 
                        # it's safe to stop expanding this branch?
                        # Actually, if dist[u] >= g, we are safe.
                        pass
                        
                    for v in adj[u]:
                        if v == parent[u]:
                            continue
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                edge_list = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edge_list.append((i, j))
                return n, edge_list
                
    # Fallback: Return a valid graph if we find one quickly in the last seconds?
    # Or just return the last n with edges (invalid).
    # To ensure we don't return invalid, we can try a very large n 
    # where a simple cycle construction might work? 
    # No, 3-regular simple graph with large girth is hard to construct simply.
    
    # If we fail, we return a dummy.
    return n_min, []
