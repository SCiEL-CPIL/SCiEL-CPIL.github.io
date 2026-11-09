import random
import time

def construct():
    # Try to find a 30-vertex coloring using backtracking
    colors_30 = solve_30()
    if colors_30 is not None:
        return {'n': 30, 'colors': colors_30}
    
    # Fallback to n=29 Paley graph
    n_base = 29
    q = n_base
    squares = set()
    for x in range(1, q):
        squares.add((x * x) % q)
    
    colors_base = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if (j - i) % q in squares:
                colors_base.append(0)
            else:
                colors_base.append(1)
                
    return {'n': n_base, 'colors': colors_base}

def solve_30():
    n = 30
    num_edges = n * (n - 1) // 2
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    # Try a few different symmetry breakings and edge orderings
    for attempt in range(10):
        deadline = time.time() + 10.0
        
        colors = [-1] * num_edges
        adj0 = [set() for _ in range(n)]
        adj1 = [set() for _ in range(n)]
        
        # Symmetry breaking: Fix (0,1) to 0, (0,2) to 0
        colors[0] = 0
        adj0[0].add(1); adj0[1].add(0)
        colors[1] = 0
        adj0[0].add(2); adj0[2].add(0)
        
        if check_k222(adj0, n) or check_k222(adj1, n):
            continue
            
        remaining_indices = list(range(2, num_edges))
        random.shuffle(remaining_indices)
        
        def backtrack(idx_pos):
            nonlocal deadline
            if time.time() > deadline:
                return None
            
            if idx_pos == len(remaining_indices):
                return colors[:]
                
            idx = remaining_indices[idx_pos]
            i, j = edges[idx]
            
            # Try color 0
            colors[idx] = 0
            adj0[i].add(j); adj0[j].add(i)
            if not check_k222(adj0, n):
                res = backtrack(idx_pos + 1)
                if res is not None:
                    return res
            adj0[i].remove(j); adj0[j].remove(i)
            
            # Try color 1
            colors[idx] = 1
            adj1[i].add(j); adj1[j].add(i)
            if not check_k222(adj1, n):
                res = backtrack(idx_pos + 1)
                if res is not None:
                    return res
            adj1[i].remove(j); adj1[j].remove(i)
            
            colors[idx] = -1
            return None
            
        result = backtrack(0)
        if result is not None:
            return result
            
    return None

def check_k222(adj, n):
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u].intersection(adj[v])
            common.discard(u)
            common.discard(v)
            
            common_list = list(common)
            len_c = len(common_list)
            for i in range(len_c):
                b1 = common_list[i]
                for j in range(i + 1, len_c):
                    b2 = common_list[j]
                    common_b1_b2 = adj[b1].intersection(adj[b2])
                    candidates = 0
                    for c in common_b1_b2:
                        if c in common and c != b1 and c != b2:
                            candidates += 1
                            if candidates >= 2:
                                return True
    return False
