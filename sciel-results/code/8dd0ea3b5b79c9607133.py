import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 58.0  # Safety margin before 60s limit
    
    # We will construct two disjoint copies of K_{3,3}
    # This gives cycle lengths {4, 6} in each component, so overall {4, 6}
    # Raw value = 2, which is optimal for min-degree-3 girth>=4 graphs
    
    def build_k33(offset):
        """Build K_{3,3} with vertices offset to offset+0..offset+5"""
        edges = []
        # Part A: offset, offset+1, offset+2
        # Part B: offset+3, offset+4, offset+5
        part_a = [offset + i for i in range(3)]
        part_b = [offset + i for i in range(3, 6)]
        for u in part_a:
            for v in part_b:
                edges.append([u, v])
        return edges
    
    # Build two components
    edges = []
    edges.extend(build_k33(0))   # Vertices 0-5
    edges.extend(build_k33(6))   # Vertices 6-11
    
    # Verify constraints
    n = 12
    degree = [0] * n
    edge_set = set()
    
    for edge in edges:
        if len(edge) != 2:
            return []
        u, v = edge
        if not (isinstance(u, int) and isinstance(v, int) and not isinstance(u, bool) and not isinstance(v, bool)):
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
    
    # Check that graph is bipartite (no odd cycles) by attempting 2-coloring
    # and verify it contains at least one cycle
    color = [-1] * n
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
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
                        return []  # Odd cycle found
    
    # Verify there is at least one cycle (graph is not a forest)
    # Since min degree is 3 and n=12, it must have cycles
    # But let's confirm by checking if edges >= vertices in any component
    visited = [False] * n
    has_cycle = False
    for i in range(n):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            component_vertices = 0
            component_edges = 0
            while stack:
                node = stack.pop()
                component_vertices += 1
                for neighbor in adj[node]:
                    component_edges += 1
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            component_edges //= 2  # Each edge counted twice
            if component_edges >= component_vertices:
                has_cycle = True
                break
    
    if not has_cycle:
        return []
    
    # Return edges as list of [u, v]
    return edges
