import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 seconds buffer
    
    n = 40
    
    def is_c6_free(adj):
        """
        Exact check for C6 using BFS for each edge.
        """
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    # Check if there is a simple path of length 5 from u to v
                    # that does not use the edge (u, v) directly?
                    # The path is u-x1-x2-x3-x4-v.
                    # We can do a BFS from u to v with max depth 5.
                    # We must ensure the path is simple.
                    
                    # BFS
                    # State: (node, path)
                    # To avoid storing full paths, we can use a visited set per BFS?
                    # No, we need to avoid revisiting nodes in the path.
                    
                    # Since depth is only 5, we can use a recursive DFS with a visited set.
                    
                    def dfs(node, target, depth, visited):
                        if depth == 5:
                            return (adj[node] >> target) & 1
                        for next_node in range(n):
                            if next_node in visited:
                                continue
                            if (adj[node] >> next_node) & 1:
                                visited.add(next_node)
                                if dfs(next_node, target, depth + 1, visited):
                                    return True
                                visited.remove(next_node)
                        return False
                    
                    visited = {u}
                    if dfs(u, v, 0, visited):
                        return False
        return True

    best_edges = []
    best_count = 0
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize random graph
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.6:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Local Search
        current_count = sum(bin(adj[u]).count('1') for u in range(n)) // 2
        
        for step in range(100):
            if time.time() > deadline:
                break
                
            # Try to find a C6
            c6_found = False
            for _ in range(50): # 50 attempts
                verts = random.sample(range(n), 6)
                # Check if these 6 vertices form a C6
                # Check degrees and connectivity as before
                edge_count = 0
                for a in range(6):
                    for b in range(a + 1, 6):
                        u = verts[a]
                        v = verts[b]
                        if (adj[u] >> v) & 1:
                            edge_count += 1
                if edge_count == 6:
                    degrees = [0] * 6
                    for a in range(6):
                        for b in range(6):
                            if a != b:
                                u = verts[a]
                                v = verts[b]
                                if (adj[u] >> v) & 1:
                                    degrees[a] += 1
                    if all(d == 2 for d in degrees):
                        # Check connectivity
                        visited = {0}
                        queue = [0]
                        while queue:
                            node = queue.pop(0)
                            for neighbor in range(6):
                                if neighbor in visited:
                                    continue
                                u = verts[node]
                                v = verts[neighbor]
                                if (adj[u] >> v) & 1:
                                    visited.add(neighbor)
                                    queue.append(neighbor)
                        if len(visited) == 6:
                            c6_found = True
                            # Remove a random edge
                            i = random.randint(0, 5)
                            u = verts[i]
                            v = verts[(i + 1) % 6]
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            current_count -= 1
                            break
            
            if not c6_found:
                break
        
        # Exact verification
        if is_c6_free(adj):
            if current_count > best_count:
                best_count = current_count
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj[u] >> v) & 1:
                            edges.append([u, v])
                best_edges = edges

    return best_edges

# EVOLVE-BLOCK-END
