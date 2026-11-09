import numpy as np
import random
import time

def construct():
    # Approach: Local Search on Cyclic Graphs (Circulants)
    for n in range(36, 28, -1):
        found = _find_cyclic_valid_graph(n)
        if found is not None:
            return found
            
    # Fallback: Paley Graph on 29 vertices
    return _paley_graph(29)

def _find_cyclic_valid_graph(n):
    start_time = time.time()
    timeout = 10.0
    half_n = n // 2
    max_mask = 1 << half_n
    
    for attempt in range(100):
        if time.time() - start_time > timeout:
            break
            
        state = random.randint(0, max_mask - 1)
        
        for step in range(500):
            if time.time() - start_time > timeout:
                break
                
            violations = _count_violations_cyclic(state, n)
            
            if violations == 0:
                return _state_to_graph(state, n)
                
            bit_idx = random.randint(0, half_n - 1)
            new_state = state ^ (1 << bit_idx)
            new_violations = _count_violations_cyclic(new_state, n)
            
            if new_violations <= violations:
                state = new_state
            elif random.random() < 0.1:
                state = new_state
                
    return None

def _count_violations_cyclic(state, n):
    S = [i + 1 for i in range(n // 2) if state & (1 << i)]
    
    adj = [[] for _ in range(n)]
    for u in range(n):
        for s in S:
            v1 = (u + s) % n
            v2 = (u - s) % n
            adj[u].append(v1)
            if v1 != v2:
                adj[u].append(v2)
    for u in range(n):
        adj[u] = list(set(adj[u]))
        
    # Check K4
    k4_found = False
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            common_uv = set(adj[u]) & set(adj[v])
            for w in common_uv:
                if w <= v:
                    continue
                common_uvw = common_uv & set(adj[w])
                for x in common_uvw:
                    if x > w:
                        k4_found = True
                        break
                if k4_found:
                    break
            if k4_found:
                break
        if k4_found:
            break
            
    if k4_found:
        return 1
        
    # Check Independent Set of size 6
    adj_set = [set(lst) for lst in adj]
    
    def greedy_indep_size(vertices_order):
        IS = []
        for u in vertices_order:
            ok = True
            for v in IS:
                if v in adj_set[u]:
                    ok = False
                    break
            if ok:
                IS.append(u)
        return len(IS)
        
    max_indep = 0
    # Try 5 random orders
    for _ in range(5):
        vertices = list(range(n))
        random.shuffle(vertices)
        size = greedy_indep_size(vertices)
        if size > max_indep:
            max_indep = size
            if max_indep >= 6:
                return 1
                
    return 0

def _state_to_graph(state, n):
    S = [i + 1 for i in range(n // 2) if state & (1 << i)]
    edges = []
    for u in range(n):
        for s in S:
            v = (u + s) % n
            if u < v:
                edges.append([u, v])
    return {'n': n, 'edges': edges}

def _paley_graph(n):
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
        
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            diff = (v - u) % n
            if diff in squares:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
