import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit for safety
    
    n = 12
    
    # 1. Construct a known valid graph with value 2: Two disjoint K_{3,3} components.
    # K_{3,3} is 3-regular, bipartite (girth 4), and has cycles of length 4 and 6.
    # Vertices 0-5: Partitions A={0,1,2}, B={3,4,5}
    # Vertices 6-11: Partitions C={6,7,8}, D={9,10,11}
    
    edges = []
    # First K_{3,3}
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Second K_{3,3}
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    # This graph is valid:
    # - Simple, undirected
    # - Degree 3 for all vertices
    # - Triangle-free (bipartite)
    # - Contains cycles (4 and 6)
    # - Distinct cycle lengths: {4, 6} -> Count = 2
    
    best_val = 2
    best_edges = edges
    
    # 2. Attempt to find a graph with value 1 via random search (optional, likely impossible but cheap to try)
    # We will run a short random search for 3-regular triangle-free graphs.
    
    def is_triangle_free(edge_list):
        adj = {i: set() for i in range(n)}
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w in adj[u] and w != u:
                        return False
        return True

    def get_cycle_lengths(edge_list):
        adj = {i: set() for i in range(n)}
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
            
        lengths = set()
        
        def dfs(start, curr, path_len, visited):
            for neighbor in adj[curr]:
                if neighbor == start and path_len > 1:
                    lengths.add(path_len + 1)
                elif neighbor not in visited and neighbor > start:
                    visited.add(neighbor)
                    dfs(start, neighbor, path_len + 1, visited)
                    visited.remove(neighbor)
                    
        for i in range(n):
            dfs(i, i, 1, {i})
            
        return lengths

    def random_3_regular_graph(n):
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        
        # Limit attempts to avoid infinite loop
        for _ in range(100):
            random.shuffle(stubs)
            edges = []
            valid = True
            seen = set()
            for i in range(0, n*3, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if u > v:
                    u, v = v, u
                if (u, v) in seen:
                    valid = False
                    break
                seen.add((u, v))
                edges.append([u, v])
            
            if valid:
                # Check triangle-free
                if is_triangle_free(edges):
                    return edges
        return None

    # Search for value 1
    search_iterations = 1000
    for _ in range(search_iterations):
        if time.time() > deadline:
            break
        edges = random_3_regular_graph(n)
        if not edges:
            continue
        
        lengths = get_cycle_lengths(edges)
        if not lengths:
            continue
            
        val = len(lengths)
        if val < best_val:
            best_val = val
            best_edges = edges
            if best_val == 1:
                break

    return best_edges

# EVOLVE-BLOCK-END
