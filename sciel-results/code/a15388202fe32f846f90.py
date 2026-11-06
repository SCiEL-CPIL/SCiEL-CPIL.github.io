import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    squares = set()
    k = 1
    while k * k <= 2 * n:
        squares.add(k * k)
        k += 1
        
    valid_vertices = []
    for i in range(1, n + 1):
        if 2 * i not in squares:
            valid_vertices.append(i)
            
    m = len(valid_vertices)
    val_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    adj = [0] * m
    for i in range(m):
        vi = valid_vertices[i]
        for j in range(i + 1, m):
            vj = valid_vertices[j]
            if vi + vj in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    degrees = [bin(adj[i]).count('1') for i in range(m)]
    sorted_indices = sorted(range(m), key=lambda x: -degrees[x])
    
    best_size = 0
    best_set = []
    
    def get_vertex(candidates_mask):
        for idx in sorted_indices:
            if candidates_mask & (1 << idx):
                return idx
        return -1
        
    def popcount(x):
        return bin(x).count('1')
        
    def dfs(candidates_mask, current_size, current_set_indices):
        nonlocal best_size, best_set, deadline
        
        if time.time() > deadline:
            return
            
        if current_size + popcount(candidates_mask) <= best_size:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set_indices[:]
            return
            
        v = get_vertex(candidates_mask)
        if v == -1:
            return
            
        new_candidates_inc = candidates_mask & ~(adj[v] | (1 << v))
        current_set_indices.append(v)
        dfs(new_candidates_inc, current_size + 1, current_set_indices)
        current_set_indices.pop()
        
        if time.time() > deadline:
            return
            
        new_candidates_exc = candidates_mask & ~(1 << v)
        dfs(new_candidates_exc, current_size, current_set_indices)
        
    initial_mask = (1 << m) - 1
    dfs(initial_mask, 0, [])
    
    result = [valid_vertices[i] for i in best_set]
    
    if not result:
        if valid_vertices:
            result = [valid_vertices[0]]
        else:
            result = [1]
            
    return result
