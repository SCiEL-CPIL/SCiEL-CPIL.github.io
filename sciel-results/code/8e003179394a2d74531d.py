import random
import time

def construct():
    N = 16
    edges = []
    edge_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edges.append((i, j))
            edge_idx[(i, j)] = idx
            idx += 1
            
    k4s = []
    k4_containing_edge = [[] for _ in range(len(edges))]
    
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                for l in range(k+1, N):
                    e1 = edge_idx[(i, j)]
                    e2 = edge_idx[(k, l)]
                    e3 = edge_idx[(i, k)]
                    e4 = edge_idx[(j, l)]
                    e5 = edge_idx[(i, l)]
                    e6 = edge_idx[(j, k)]
                    k4_idx = len(k4s)
                    k4s.append((e1, e2, e3, e4, e5, e6))
                    
                    for e in [e1, e2, e3, e4, e5, e6]:
                        k4_containing_edge[e].append(k4_idx)

    def get_k4_bad(k4_idx, colors):
        e1, e2, e3, e4, e5, e6 = k4s[k4_idx]
        bad = 0
        if colors[e1] == colors[e2]:
            bad += 1
        if colors[e3] == colors[e4]:
            bad += 1
        if colors[e5] == colors[e6]:
            bad += 1
        return 1 if bad >= 2 else 0

    def try_solve(K, time_limit=15):
        start_time = time.time()
        colors = [random.randint(0, K-1) for _ in range(len(edges))]
        
        k4_bad = [get_k4_bad(i, colors) for i in range(len(k4s))]
        curr_cost = sum(k4_bad)
        
        if curr_cost == 0:
            return colors

        T = 10.0
        T_min = 0.05
        alpha = 0.9995

        while time.time() - start_time < time_limit:
            if T < T_min:
                break
                
            e_idx = random.randint(0, len(edges)-1)
            old_c = colors[e_idx]
            
            if K > 1:
                new_c = random.randint(0, K-1)
                if new_c == old_c:
                    continue
            else:
                continue
                
            colors[e_idx] = new_c
            
            delta = 0
            affected_k4s = k4_containing_edge[e_idx]
            old_bad_vals = []
            
            for k4_idx in affected_k4s:
                old_bad = k4_bad[k4_idx]
                new_bad = get_k4_bad(k4_idx, colors)
                delta += new_bad - old_bad
                k4_bad[k4_idx] = new_bad
                old_bad_vals.append((k4_idx, old_bad))
            
            new_cost = curr_cost + delta
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                curr_cost = new_cost
            else:
                colors[e_idx] = old_c
                for k4_idx, old_bad in old_bad_vals:
                    k4_bad[k4_idx] = old_bad
                
            T *= alpha
            
        if curr_cost == 0:
            return colors
        return None

    # Try K=13, 14, 15
    for K in [13, 14, 15]:
        res = try_solve(K, time_limit=30)
        if res:
            return res
            
    # Fallback to known valid solution (17 colors)
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
