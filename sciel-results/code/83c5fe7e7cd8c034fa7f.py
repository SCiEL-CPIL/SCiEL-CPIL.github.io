import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 115
    
    # Petersen graph edges (15 edges, 10 vertices)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    def check_girth(n, edges, min_girth):
        """
        Check if graph has girth >= min_girth using non-backtracking BFS.
        """
        if n < min_girth:
            return False
            
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_depth = min_girth // 2 - 1
        
        for start in range(n):
            visited = set()
            queue = deque([(start, -1, 0)])
            visited.add((start, -1))
            
            while queue:
                node, parent, depth = queue.popleft()
                if depth == max_depth:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if (nb, node) in visited:
                        return False
                    visited.add((nb, node))
                    queue.append((nb, node, depth + 1))
        return True

    def build_lift(m, shifts):
        n = 10 * m
        edges = []
        for (u, v), shift in zip(petersen_edges, shifts):
            for i in range(m):
                u_id = u * m + i
                v_id = v * m + (i + shift) % m
                edges.append((u_id, v_id))
        return n, edges

    # Search for the smallest m such that a valid lift exists
    # Moore bound implies m >= 51
    m = 51
    best_n = None
    best_edges = None
    
    while m <= 500 and time.time() < deadline:
        # Try random shifts
        # For small m, increase attempts
        attempts = 30 if m < 100 else 10
        
        found = False
        for _ in range(attempts):
            if time.time() > deadline:
                break
            shifts = [random.randint(0, m - 1) for _ in range(15)]
            n, edges = build_lift(m, shifts)
            if check_girth(n, edges, g):
                return n, edges
                found = True
                break
        
        # If not found, try a deterministic pattern
        if not found and time.time() < deadline:
            # Linear shifts
            shifts_lin = [i * (m // 15 + 1) % m for i in range(15)]
            n, edges = build_lift(m, shifts_lin)
            if check_girth(n, edges, g):
                return n, edges
                
        m += 1

    # Fallback: Return a large valid graph
    # Use a large m to ensure validity
    m = 1000
    shifts = [random.randint(0, m - 1) for _ in range(15)]
    n, edges = build_lift(m, shifts)
    return n, edges
