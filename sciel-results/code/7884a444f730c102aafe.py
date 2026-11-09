import random
import time
from itertools import combinations

def is_k4_free(adj):
    n = len(adj)
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] & (1 << b)): continue
            for c in range(b+1, n):
                if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)): continue
                if adj[a] & adj[b] & adj[c]:
                    return False
    return True

def get_largest_triangle_free(adj, upper_bound=None):
    n = len(adj)
    if upper_bound is None:
        upper_bound = n
    
    # Precompute triangles
    triangles = []
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] & (1 << b)): continue
            common = adj[a] & adj[b]
            c_mask = common & ~((1 << (b+1)) - 1)
            while c_mask:
                c_bit = c_mask & -c_mask
                c = c_bit.bit_length() - 1
                triangles.append((a, b, c))
                c_mask ^= c_bit
    
    if not triangles:
        return n

    # Check subsets from upper_bound down to 1
    for k in range(upper_bound, 0, -1):
        for subset in combinations(range(n), k):
            m = 0
            for v in subset:
                m |= (1 << v)
            
            # Check if any triangle is fully contained in m
            has_triangle = False
            for (a, b, c) in triangles:
                if (m >> a) & 1 and (m >> b) & 1 and (m >> c) & 1:
                    has_triangle = True
                    break
            if not has_triangle:
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    best_l = 18
    best_graph = None
    n = 18
    
    # Generate a random K4-free graph
    def gen_random_k4_free():
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        for _ in range(200):
            if is_k4_free(adj):
                return adj
            found = False
            for a in range(n):
                for b in range(a+1, n):
                    if not (adj[a] & (1 << b)): continue
                    for c in range(b+1, n):
                        if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)): continue
                        common = adj[a] & adj[b] & adj[c]
                        if common:
                            d = (common & -common).bit_length() - 1
                            adj[a] &= ~(1 << d)
                            adj[d] &= ~(1 << a)
                            found = True
                            break
                    if found: break
                if found: break
            if not found:
                return adj
        return adj
    
    # Generate Turán graph T(18, 3) as a structured starting point
    def gen_turan_graph():
        adj = [0] * n
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        return adj
    
    # Simulated annealing style local search
    def local_search(adj, current_l, max_iters=500):
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try removing edges
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj[i] & (1 << j)):
                        continue
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                    if is_k4_free(adj):
                        l2 = get_largest_triangle_free(adj, upper_bound=current_l)
                        if l2 < current_l:
                            current_l = l2
                            improved = True
                        else:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                    else:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Try adding edges
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        continue
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    if is_k4_free(adj):
                        l2 = get_largest_triangle_free(adj, upper_bound=current_l)
                        if l2 < current_l:
                            current_l = l2
                            improved = True
                        else:
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                    else:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
        
        return current_l
    
    # Start with Turán graph
    adj = gen_turan_graph()
    l = get_largest_triangle_free(adj)
    best_l = l
    best_graph = adj[:]
    
    # Local search from Turán
    l = local_search(adj, l)
    if l < best_l:
        best_l = l
        best_graph = adj[:]
    
    # Random restarts
    while time.time() < deadline:
        adj = gen_random_k4_free()
        l = get_largest_triangle_free(adj)
        
        if l < best_l:
            best_l = l
            best_graph = adj[:]
        
        # Local search
        if time.time() < deadline - 5:
            l = local_search(adj, l, max_iters=100)
            if l < best_l:
                best_l = l
                best_graph = adj[:]
        
        if best_l <= 5:
            break
    
    # Convert best_graph to list of edges
    edges = []
    if best_graph is None:
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
        
    for i in range(n):
        for j in range(i+1, n):
            if best_graph[i] & (1 << j):
                edges.append([i, j])
                
    return edges
