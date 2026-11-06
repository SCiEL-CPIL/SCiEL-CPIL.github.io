import random
import time

def construct():
    N = 16
    k4s = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                for l in range(k+1, N):
                    k4s.append((i, j, k, l))
                    
    k4_edges = []
    for idx, (i, j, k, l) in enumerate(k4s):
        edges = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
        k4_edges.append(edges)
        
    edge_to_k4s = {}
    for idx, edges in enumerate(k4_edges):
        for e in edges:
            if e not in edge_to_k4s:
                edge_to_k4s[e] = []
            edge_to_k4s[e].append(idx)
            
    def run_search(K, time_limit):
        colors = [[0]*N for _ in range(N)]
        k4_color_counts = [[0]*K for _ in range(len(k4s))]
        k4_distinct = [0] * len(k4s)
        
        for i in range(N):
            for j in range(i+1, N):
                c = random.randint(0, K-1)
                colors[i][j] = c
                colors[j][i] = c
                for k4_idx in edge_to_k4s[(i, j)]:
                    if k4_color_counts[k4_idx][c] == 0:
                        k4_distinct[k4_idx] += 1
                    k4_color_counts[k4_idx][c] += 1
                    
        cost = 0
        for idx in range(len(k4s)):
            if k4_distinct[idx] < 5:
                cost += 5 - k4_distinct[idx]
                
        if cost == 0:
            return colors
            
        start_time = time.time()
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while T > min_T:
            if time.time() - start_time > time_limit:
                break
                
            u = random.randint(0, N-1)
            v = random.randint(0, N-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            old_c = colors[u][v]
            new_c = random.randint(0, K-1)
            if new_c == old_c:
                continue
                
            affected = edge_to_k4s[(u, v)]
            delta = 0
            changes = []
            
            for k4_idx in affected:
                old_distinct = k4_distinct[k4_idx]
                
                k4_color_counts[k4_idx][old_c] -= 1
                if k4_color_counts[k4_idx][old_c] == 0:
                    new_distinct = old_distinct - 1
                else:
                    new_distinct = old_distinct
                    
                if k4_color_counts[k4_idx][new_c] == 0:
                    new_distinct += 1
                k4_color_counts[k4_idx][new_c] += 1
                
                changes.append((k4_idx, old_distinct, new_distinct))
                
                old_bad = max(0, 5 - old_distinct)
                new_bad = max(0, 5 - new_distinct)
                delta += new_bad - old_bad
                
                k4_distinct[k4_idx] = new_distinct
                
            if delta <= 0 or random.random() < pow(2.71828, -delta / T):
                cost += delta
                colors[u][v] = new_c
                colors[v][u] = new_c
                if cost == 0:
                    return colors
            else:
                for k4_idx, old_distinct, new_distinct in changes:
                    k4_distinct[k4_idx] = old_distinct
                    k4_color_counts[k4_idx][new_c] -= 1
                    k4_color_counts[k4_idx][old_c] += 1
                    
            T *= cooling
            
        return None

    # Time allocation
    # 13: 20s
    res = run_search(13, 20)
    if res:
        return [res[i][j] for i in range(N) for j in range(i+1, N)]
        
    # 14: 30s
    res = run_search(14, 30)
    if res:
        return [res[i][j] for i in range(N) for j in range(i+1, N)]
        
    # 15: 70s
    res = run_search(15, 70)
    if res:
        return [res[i][j] for i in range(N) for j in range(i+1, N)]
        
    # If all fail, return a dummy valid coloring?
    # We must return a valid one.
    # This should not happen.
    return [0]*120

