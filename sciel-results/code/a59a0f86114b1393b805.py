import math
import random

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def construct():
    # 1. Filter valid vertices (2*v not square)
    valid_vertices = []
    for i in range(1, 257):
        if not is_square(2 * i):
            valid_vertices.append(i)
            
    V = set(valid_vertices)
    
    # 2. Build adjacency list
    adj = {v: set() for v in V}
    v_list = list(V)
    for i in range(len(v_list)):
        u = v_list[i]
        for j in range(i + 1, len(v_list)):
            v = v_list[j]
            if is_square(u + v):
                adj[u].add(v)
                adj[v].add(u)
                
    best_set = set()
    best_size = 0
    
    # Try multiple random restarts
    for _ in range(100):
        S = set()
        U = set(V)
        
        # Greedy Initialization with Random Order
        u_list = list(U)
        random.shuffle(u_list)
        for u in u_list:
            if u in U:
                if not (adj[u] & S):
                    S.add(u)
                    U.remove(u)
                    
        # Local Search
        improved = True
        while improved:
            improved = False
            # 1. Add any non-conflicting node
            for u in list(U):
                if not (adj[u] & S):
                    S.add(u)
                    U.remove(u)
                    improved = True
                    break
            
            if not improved:
                # 2. Try 1-for-1 swap
                for u in list(U):
                    neighbors_in_S = adj[u] & S
                    if len(neighbors_in_S) == 1:
                        v = list(neighbors_in_S)[0]
                        S.remove(v)
                        U.add(v)
                        S.add(u)
                        U.remove(u)
                        improved = True
                        break
                        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
    return list(best_set)
