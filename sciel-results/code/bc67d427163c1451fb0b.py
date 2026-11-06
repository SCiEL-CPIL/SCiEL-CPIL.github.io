import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    N = 256
    valid_vertices = []
    for v in range(1, N + 1):
        s = 2 * v
        r = int(math.isqrt(s))
        if r * r != s:
            valid_vertices.append(v)
        
    squares = set(i*i for i in range(2, 24)) # up to 23^2
    
    conflicts = {v: set() for v in valid_vertices}
    for i in range(len(valid_vertices)):
        u = valid_vertices[i]
        for j in range(i + 1, len(valid_vertices)):
            v = valid_vertices[j]
            if u + v in squares:
                conflicts[u].add(v)
                conflicts[v].add(u)
                
    best_set = set(range(1, N + 1, 3)) # Fallback
    best_size = len(best_set)
    
    def greedy_min_degree(vertices, conflicts):
        selected = set()
        available = set(vertices)
        
        while available:
            min_deg = float('inf')
            candidates = []
            # Optimization: if available is small, just iterate
            for v in available:
                # Count neighbors in available
                deg = 0
                for u in conflicts[v]:
                    if u in available:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    candidates = [v]
                elif deg == min_deg:
                    candidates.append(v)
            
            v = random.choice(candidates)
            selected.add(v)
            # Remove v and neighbors
            to_remove = {v}
            to_remove.update(u for u in conflicts[v] if u in available)
            available.difference_update(to_remove)
            
        return selected

    # Run restarts
    while time.time() < deadline:
        result = greedy_min_degree(set(valid_vertices), conflicts)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
            
    return list(best_set)
