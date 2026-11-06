import itertools
import random

def construct():
    n = 18
    # Initialize with Turan graph T_3(18)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = set()
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    edges.add((u, v))
    
    def is_k4_free(edges):
        vertices = list(range(n))
        for combo in itertools.combinations(vertices, 4):
            u, v, w, x = combo
            if ((u,v) in edges or (v,u) in edges) and \
               ((u,w) in edges or (w,u) in edges) and \
               ((u,x) in edges or (x,u) in edges) and \
               ((v,w) in edges or (w,v) in edges) and \
               ((v,x) in edges or (x,v) in edges) and \
               ((w,x) in edges or (x,w) in edges):
                return False
        return True

    def max_triangle_free_size(edges):
        vertices = list(range(n))
        max_size = 0
        # Check all subsets of size 12 down to 1?
        # We want the largest. Start from 12.
        for size in range(12, 0, -1):
            found = False
            for combo in itertools.combinations(vertices, size):
                # Check if induced subgraph is triangle-free
                is_tf = True
                for i in range(size):
                    for j in range(i+1, size):
                        for k in range(j+1, size):
                            u, v, w = combo[i], combo[j], combo[k]
                            if ((u,v) in edges or (v,u) in edges) and \
                               ((u,w) in edges or (w,u) in edges) and \
                               ((v,w) in edges or (w,v) in edges):
                                is_tf = False
                                break
                        if not is_tf:
                            break
                    if not is_tf:
                        break
                if is_tf:
                    return size
            # If no triangle-free subset of this size, continue to smaller
        return 0

    current_edges = edges
    current_score = max_triangle_free_size(current_edges)
    
    # Local search
    for _ in range(5000):
        # Pick a random edge to toggle or swap
        all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_pairs)
        improved = False
        for u, v in all_pairs:
            new_edges = set(current_edges)
            if (u, v) in new_edges:
                new_edges.remove((u, v))
            else:
                new_edges.add((u, v))
            
            if not is_k4_free(new_edges):
                continue
                
            new_score = max_triangle_free_size(new_edges)
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
                improved = True
                break
                
        if not improved:
            # Try a random restart or different mutation?
            # For simplicity, just break if no improvement in a while
            pass

    return [[u, v] for u, v in current_edges]
