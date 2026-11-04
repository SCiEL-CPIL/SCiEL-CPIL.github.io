import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def generate_random_k_regular(N, k):
        # Configuration model
        while True:
            # 1. Create half-edges
            half_edges = []
            for i in range(N):
                for _ in range(k):
                    half_edges.append(i)
            
            random.shuffle(half_edges)
            
            # 2. Pair them
            adj = [set() for _ in range(N)]
            valid = True
            for i in range(0, len(half_edges), 2):
                u = half_edges[i]
                v = half_edges[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            
            if not valid:
                continue
            
            # 3. Check connectivity
            visited = [False] * N
            queue = deque([0])
            visited[0] = True
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            if all(visited):
                # Convert to list of lists for easier manipulation
                return [list(s) for s in adj]
            
    def get_girth(adj, N):
        min_len = float('inf')
        for u in range(N):
            dist = [-1] * N
            prev = [-1] * N
            dist[u] = 0
            queue = deque([u])
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        prev[neighbor] = node
                        queue.append(neighbor)
                    else:
                        if neighbor != prev[node]:
                            cycle_len = dist[node] + dist[neighbor] + 1
                            if cycle_len < min_len:
                                min_len = cycle_len
        return min_len if min_len != float('inf') else float('inf')

    def try_find_graph(N, k, g):
        # Generate initial graph
        adj = None
        for _ in range(100): # Max attempts to generate valid connected graph
            if time.time() > deadline:
                return None, None
            adj = generate_random_k_regular(N, k)
            if adj:
                break
        if not adj:
            return None, None

        # Local search
        for _ in range(20000):
            if time.time() > deadline:
                return None, None
            
            girth = get_girth(adj, N)
            if girth >= g:
                # Extract edges
                edges = []
                for u in range(N):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return N, edges
            
            # Pick random edge to swap
            # Build list of edges
            edge_list = []
            for u in range(N):
                for v in adj[u]:
                    if u < v:
                        edge_list.append((u, v))
            
            if len(edge_list) < 2:
                return None, None
                
            random.shuffle(edge_list)
            e1 = edge_list[0]
            e2 = edge_list[1]
            
            u1, v1 = e1
            u2, v2 = e2
            
            # Check if they share a vertex
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
                
            # Try swap (u1, u2) and (v1, v2)
            # Check if new edges exist
            if u2 in adj[u1] or v2 in adj[v1]:
                continue
            if u1 == u2 or v1 == v2: # Loops
                continue

            # Perform swap
            adj[u1].remove(v1)
            adj[u2].remove(v1) # Wait, v1 is connected to u1. v1 is NOT connected to u2.
            # Correct swap logic:
            # Remove (u1, v1) and (u2, v2)
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            # Add (u1, u2) and (v1, v2)
            adj[u1].append(u2)
            adj[u2].append(u1)
            adj[v1].append(v2)
            adj[v2].append(v1)

    # Target sizes
    target_ns = [936, 950, 1000, 1200, 1500, 2000, 4000]
    
    best_n = None
    best_edges = None
    
    for N in target_ns:
        if time.time() > deadline:
            break
        n_found, edges_found = try_find_graph(N, k, g)
        if n_found is not None:
            best_n = n_found
            best_edges = edges_found
            break

    if best_n is None:
        n_found, edges_found = try_find_graph(10000, k, g)
        if n_found:
            best_n = n_found
            best_edges = edges_found
        else:
            return (0, [])

    return (best_n, best_edges)
