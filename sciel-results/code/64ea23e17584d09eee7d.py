import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(adj, n, remaining_edges):
        # DFS to find longest cycle
        best_len = 0
        best_cycle = []
        
        for start in range(n):
            if not adj[start]:
                continue
            
            # Heuristic: Start from nodes with higher degree?
            visited = {start: 0}
            parent = {start: None}
            in_stack = {start}
            
            def dfs(u):
                nonlocal best_len, best_cycle
                for v in sorted(adj[u]):
                    e = (min(u, v), max(u, v))
                    if e not in remaining_edges:
                        continue
                    if v not in visited:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        in_stack.add(v)
                        dfs(v)
                        in_stack.remove(v)
                    elif v in in_stack:
                        length = visited[u] - visited[v] + 1
                        if length >= 3 and length > best_len:
                            cycle = [u]
                            cur = u
                            while cur != v:
                                cur = parent[cur]
                                cycle.append(cur)
                            best_cycle = cycle
                            best_len = length

            dfs(start)
            if time.time() > deadline:
                break
                
        return best_cycle

    def greedy_decompose(edges, n):
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        remaining_edges = set(edges)
        
        while remaining_edges and time.time() < deadline:
            cycle = find_longest_cycle(adj, n, remaining_edges)
            if cycle:
                pieces.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e in remaining_edges:
                        remaining_edges.discard(e)
                        adj[u].remove(v)
                        adj[v].remove(u)
            else:
                # No more cycles, rest are single edges
                for e in remaining_edges:
                    pieces.append(list(e))
                break
                
        return pieces, remaining_edges

    def local_improve(pieces, remaining_edges, n, adj_single):
        # adj_single is adjacency of remaining_edges (singletons)
        # We look for a cycle in pieces and a path in adj_single connecting two of its vertices
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Sort pieces by length descending to try extending long cycles first?
            # Or just iterate.
            for i, piece in enumerate(pieces):
                if len(piece) < 3:
                    continue
                # Try to extend this cycle
                # Pick two vertices u, v in piece
                # Check if there is a path in adj_single between u and v
                # To speed up, only check pairs that are "close" or just BFS from u
                u = piece[0]
                # BFS in adj_single from u
                visited = {u: -1}
                queue = deque([u])
                found_v = None
                while queue:
                    curr = queue.popleft()
                    for neighbor in adj_single.get(curr, []):
                        if neighbor not in visited:
                            visited[neighbor] = curr
                            if neighbor in piece and neighbor != u:
                                found_v = neighbor
                                break
                            queue.append(neighbor)
                    if found_v:
                        break
                if found_v:
                    # Reconstruct path
                    path = [found_v]
                    curr = found_v
                    while curr != u:
                        curr = visited[curr]
                        path.append(curr)
                    path.reverse() # u to found_v
                    
                    # New cycle: piece + path
                    # piece is a list of vertices.
                    # We need to merge them.
                    # Let's just create a new piece list.
                    # The cycle is piece[0]...piece[-1] + path[1:] (since u=piece[0] is shared)
                    # Actually, the cycle in the graph is the union of edges.
                    # The vertices are piece + path[1:]
                    new_cycle = piece + path[1:]
                    
                    # Remove old piece and single edges in path
                    pieces.pop(i)
                    for j in range(len(path)-1):
                        a, b = path[j], path[j+1]
                        e = (min(a,b), max(a,b))
                        if e in remaining_edges:
                            remaining_edges.discard(e)
                            adj_single[a].remove(b)
                            adj_single[b].remove(a)
                    
                    pieces.append(new_cycle)
                    improved = True
                    break # Restart loop as pieces changed
                    
        return pieces, remaining_edges

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() > deadline:
             # Fallback for remaining specs
             edges = instance(*spec)
             n = spec[0]
             # Just return single edges? Or greedy?
             # Greedy is fast.
             p, r = greedy_decompose(edges, n)
             results.append(p)
             continue

        edges = instance(*spec)
        n = spec[0]
        
        # Run greedy multiple times with slight randomization?
        # Or just one good greedy + local improve.
        # Let's try 5 restarts of greedy to find a better starting point.
        best_pieces = None
        best_count = float('inf')
        
        for _ in range(5):
             if time.time() > deadline:
                 break
             p, r = greedy_decompose(edges, n)
             count = len(p)
             if count < best_count:
                 best_count = count
                 best_pieces = p
                 best_remaining = r
                 
        if best_pieces is None:
             best_pieces, best_remaining = greedy_decompose(edges, n)
             
        # Local improvement
        adj_single = {i: set() for i in range(n)}
        for u, v in best_remaining:
            adj_single[u].add(v)
            adj_single[v].add(u)
            
        best_pieces, best_remaining = local_improve(best_pieces, best_remaining, n, adj_single)
        
        # Final check: any remaining edges?
        # local_improve removes them from remaining_edges.
        # If any left, they are singletons.
        for e in best_remaining:
             best_pieces.append(list(e))
             
        results.append(best_pieces)
        
    return results
