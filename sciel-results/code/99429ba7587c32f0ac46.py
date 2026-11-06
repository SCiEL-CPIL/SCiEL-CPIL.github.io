# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        if not edges:
            return []
            
        avail_edges = set(edges)
        n = max(max(u, v) for u, v in edges) + 1
        
        pieces = []
        
        while avail_edges:
            if time.time() > deadline:
                break
            
            # Build adjacency list for current available edges
            adj = defaultdict(list)
            for u, v in avail_edges:
                adj[u].append(v)
                adj[v].append(u)
            
            visited = [False] * n
            parent = [-1] * n
            depth = [-1] * n
            best_cycle_vertices = None
            best_len = 0
            
            # DFS to find the longest cycle formed by a back edge in the DFS tree
            for start in range(n):
                if visited[start]:
                    continue
                
                stack = [start]
                visited[start] = True
                parent[start] = -1
                depth[start] = 0
                
                while stack:
                    u = stack.pop()
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            parent[v] = u
                            depth[v] = depth[u] + 1
                            stack.append(v)
                        elif v != parent[u] and depth[v] < depth[u]:
                            # Found a back edge to an ancestor
                            cyc_len = depth[u] - depth[v] + 1
                            if cyc_len > best_len:
                                best_len = cyc_len
                                # Reconstruct cycle
                                path = []
                                curr = u
                                while curr != v:
                                    path.append(curr)
                                    curr = parent[curr]
                                path.append(v)
                                best_cycle_vertices = path
            
            if best_cycle_vertices:
                # Extract edges of the cycle
                cycle_edges = set()
                verts = best_cycle_vertices
                for i in range(len(verts)):
                    u, v = verts[i], verts[(i + 1) % len(verts)]
                    e = tuple(sorted((u, v)))
                    cycle_edges.add(e)
                
                # Verify and remove
                if all(e in avail_edges for e in cycle_edges):
                    avail_edges.difference_update(cycle_edges)
                    pieces.append(verts)
                else:
                    # Safety break if state is inconsistent
                    break
            else:
                # No cycle found. Remaining edges form a forest.
                for e in avail_edges:
                    pieces.append(list(e))
                break
        
        # If we broke due to deadline or other issues, cover remaining edges
        if avail_edges:
            for e in avail_edges:
                pieces.append(list(e))
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        res = decompose_graph(edges)
        results.append(res)
        
    return results
# EVOLVE-BLOCK-END
