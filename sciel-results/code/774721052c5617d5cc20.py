import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 20
    vertices = list(range(n))
    
    def build_greedy(order):
        edges = set()
        # For fast neighbor lookup
        # We can just use the set of edges
        
        def has_path_len_3(u, v):
            # u -> a -> b -> v
            # Find neighbors of u
            # To speed up, maintain neighbor sets?
            # Given small N, linear scan is fine.
            for (x, y) in edges:
                if x == u:
                    a = y
                    # Find neighbors of a
                    for (x2, y2) in edges:
                        if x2 == a:
                            b = y2
                            # Check if b connected to v
                            if (b, v) in edges or (v, b) in edges:
                                return True
                        elif y2 == a:
                            b = x2
                            if (b, v) in edges or (v, b) in edges:
                                return True
            return False

        for i in range(n):
            for j in range(i+1, n):
                u, v = order[i], order[j]
                if (u, v) not in edges and (v, u) not in edges:
                    if not has_path_len_3(u, v):
                        edges.add((u, v))
        return edges

    best_edges = set()
    best_count = 0
    
    for _ in range(200):
        random.shuffle(vertices)
        edges = build_greedy(vertices)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            
        if time.time() > deadline - 20.0:
            break
            
    # Local search: try to add any edge
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in best_edges and (j, i) not in best_edges:
                    # Check validity
                    # Re-implement check for best_edges
                    def has_path_len_3_global(u, v, edges):
                        for (x, y) in edges:
                            if x == u:
                                a = y
                                for (x2, y2) in edges:
                                    if x2 == a:
                                        b = y2
                                        if (b, v) in edges or (v, b) in edges:
                                            return True
                                    elif y2 == a:
                                        b = x2
                                        if (b, v) in edges or (v, b) in edges:
                                            return True
                        return False
                    if not has_path_len_3_global(i, j, best_edges):
                        best_edges.add((i, j))
                        improved = True
                        if time.time() > deadline - 5.0:
                            break
            if time.time() > deadline - 5.0:
                break

    result = []
    for u, v in best_edges:
        result.append([u, v, 0])
        
    return result
