import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 58.0

    def build_k33(offset):
        edges = []
        part_a = [offset + i for i in range(3)]
        part_b = [offset + i for i in range(3, 6)]
        for u in part_a:
            for v in part_b:
                edges.append([u, v])
        return edges

    # Build two K_{3,3} components
    edges = []
    edges.extend(build_k33(0))
    edges.extend(build_k33(6))
    
    # Connect them with a single bridge edge to make the graph connected
    # This does not create any new cycles, so cycle lengths remain {4, 6}
    edges.append([0, 6])

    n = 12
    degree = [0] * n
    edge_set = set()
    
    for edge in edges:
        if len(edge) != 2:
            return []
        u, v = edge
        if not isinstance(u, int) or not isinstance(v, int):
            return []
        if isinstance(u, bool) or isinstance(v, bool):
            return []
        if u == v:
            return []
        if not (0 <= u < n and 0 <= v < n):
            return []
        if (u, v) in edge_set or (v, u) in edge_set:
            return []
        edge_set.add((u, v))
        edge_set.add((v, u))
        degree[u] += 1
        degree[v] += 1
    
    if any(d < 3 for d in degree):
        return []
        
    # Bipartite check
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    color = [-1] * n
    for i in range(n):
        if color[i] == -1:
            color[i] = 0
            queue = [i]
            while queue:
                node = queue.pop(0)
                for neighbor in adj[node]:
                    if color[neighbor] == -1:
                        color[neighbor] = 1 - color[node]
                        queue.append(neighbor)
                    elif color[neighbor] == color[node]:
                        return []
                        
    return edges
