import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0
    n = 31
    target_edges = 91
    random.seed(42)
    
    def generate_polarity_graph():
        q = 5
        points = []
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        for m in range(q):
            points.append((1, m, 0))
        points.append((0, 1, 0))
        
        edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[j]
                if (x1 * x2 + y1 * y2 + z1 * z2) % q == 0:
                    edges.add((i, j))
        return edges
    
    def count_c4(adj):
        common = adj @ adj.T
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = common[i, j]
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2
    
    def delta_swap(adj, common, a, b, c, d):
        if adj[c, d] or not adj[a, b]:
            return 0
        
        # Compute delta in S = sum_{i<j} C(common[i,j], 2)
        # Then delta_C4 = delta_S // 2
        
        delta_S = 0
        
        # Removal of (a,b)
        for x in range(n):
            if x == a or x == b:
                continue
            i, j = (a, x) if a < x else (x, a)
            old_c = common[i, j]
            new_c = old_c - adj[b, x]
            delta_S += new_c * (new_c - 1) // 2 - old_c * (old_c - 1) // 2
            
            i, j = (b, x) if b < x else (x, b)
            old_c = common[i, j]
            new_c = old_c - adj[a, x]
            delta_S += new_c * (new_c - 1) // 2 - old_c * (old_c - 1) // 2
        
        # Addition of (c,d)
        for x in range(n):
            if x == c or x == d:
                continue
            adj_d_x = adj[d, x]
            if (d == a and x == b) or (d == b and x == a):
                adj_d_x = 0
            adj_c_x = adj[c, x]
            if (c == a and x == b) or (c == b and x == a):
                adj_c_x = 0
            
            i, j = (c, x) if c < x else (x, c)
            old_c = common[i, j]
            new_c = old_c + adj_d_x
            delta_S += new_c * (new_c - 1) // 2 - old_c * (old_c - 1) // 2
            
            i, j = (d, x) if d < x else (x, d)
            old_c = common[i, j]
            new_c = old_c + adj_c_x
            delta_S += new_c * (new_c - 1) // 2 - old_c * (old_c - 1) // 2
        
        return delta_S // 2
    
    init_edges = generate_polarity_graph()
    
    if len(init_edges) > target_edges:
        edge_list = list(init_edges)
        random.shuffle(edge_list)
        while len(init_edges) > target_edges:
            init_edges.discard(edge_list.pop())
    elif len(init_edges) < target_edges:
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        non_edges = [e for e in all_edges if e not in init_edges]
        random.shuffle(non_edges)
        while len(init_edges) < target_edges and non_edges:
            init_edges.add(non_edges.pop())
    
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in init_edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    common = adj @ adj.T
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_edges = set(init_edges)
    
    all_edges_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    T = 5.0
    T_min = 0.05
    cooling = 0.9995
    no_improve = 0
    max_no_improve = 200
    
    while time.time() - start_time < deadline - 5:
        if T < T_min:
            break
        
        current_edge_list = list(init_edges)
        a, b = random.choice(current_edge_list)
        
        non_edges = [e for e in all_edges_list if e not in init_edges]
        if not non_edges:
            break
        c, d = random.choice(non_edges)
        
        delta = delta_swap(adj, common, a, b, c, d)
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            init_edges.discard((a, b))
            init_edges.add((c, d))
            adj[a, b] = 0
            adj[b, a] = 0
            adj[c, d] = 1
            adj[d, c] = 1
            common = adj @ adj.T
            current_c4 += delta
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = set(init_edges)
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > max_no_improve:
                no_improve = 0
                init_edges = set(best_edges)
                adj = np.zeros((n, n), dtype=np.int8)
                for u, v in init_edges:
                    adj[u, v] = 1
                    adj[v, u] = 1
                common = adj @ adj.T
                current_c4 = best_c4
                T = max(T * 0.5, T_min * 2)
        
        T *= cooling
    
    return [[u, v] for u, v in best_edges]
