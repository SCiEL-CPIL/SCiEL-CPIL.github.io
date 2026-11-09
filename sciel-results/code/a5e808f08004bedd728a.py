import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 16
    num_edges = 120
    num_k4 = 2184
    
    edge_indices = []
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = idx
            edge_map[(j, i)] = idx
            edge_indices.append((i, j))
            idx += 1
            
    k4_edges = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    e1 = edge_map[(a, b)]
                    e2 = edge_map[(a, c)]
                    e3 = edge_map[(a, d)]
                    e4 = edge_map[(b, c)]
                    e5 = edge_map[(b, d)]
                    e6 = edge_map[(c, d)]
                    k4_edges.append((e1, e2, e3, e4, e5, e6))
                    
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
            
    def solve_optimized(K):
        if time.time() > deadline:
            return None
            
        colors = [random.randint(0, K - 1) for _ in range(num_edges)]
        
        # k4_counts[k4_idx] is a list of counts for each color
        k4_counts = [[0] * K for _ in range(num_k4)]
        k4_distinct = [0] * num_k4
        
        for k4_idx, edges in enumerate(k4_edges):
            for e in edges:
                k4_counts[k4_idx][colors[e]] += 1
            distinct = 0
            for c in range(K):
                if k4_counts[k4_idx][c] > 0:
                    distinct += 1
            k4_distinct[k4_idx] = distinct
            
        bad_count = sum(1 for d in k4_distinct if d < 5)
        
        if bad_count == 0:
            return colors

        best_colors = colors.copy()
        best_bad = bad_count
        
        temp = 1.0
        cooling = 0.9999
        max_steps = 200000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            e = random.randint(0, num_edges - 1)
            old_color = colors[e]
            new_color = random.randint(0, K - 1)
            if new_color == old_color:
                continue
                
            delta = 0
            for k4_idx in edge_to_k4s[e]:
                counts = k4_counts[k4_idx]
                distinct = k4_distinct[k4_idx]
                
                # Calculate change in distinct
                # old_color count decreases
                if counts[old_color] == 1:
                    distinct -= 1
                # new_color count increases
                if counts[new_color] == 0:
                    distinct += 1
                    
                bad_before = 1 if k4_distinct[k4_idx] < 5 else 0
                bad_after = 1 if distinct < 5 else 0
                delta += bad_after - bad_before
                
            new_bad = bad_count + delta
            
            if new_bad < bad_count or (new_bad == bad_count and random.random() < 0.5) or (new_bad > bad_count and random.random() < pow(2.71828, -new_bad / temp)):
                # Apply
                for k4_idx in edge_to_k4s[e]:
                    counts = k4_counts[k4_idx]
                    if counts[old_color] == 1:
                        k4_distinct[k4_idx] -= 1
                    if counts[new_color] == 0:
                        k4_distinct[k4_idx] += 1
                    counts[old_color] -= 1
                    counts[new_color] += 1
                colors[e] = new_color
                bad_count = new_bad
                
                if bad_count == 0:
                    return colors
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors.copy()
                    
            temp *= cooling
            if temp < 0.01:
                break
                
        return best_colors

    # Try 15 colors
    res = solve_optimized(15)
    if res:
        return res
        
    # Try 14 colors
    if time.time() < deadline - 10:
        res = solve_optimized(14)
        if res:
            return res
            
    # Try 13 colors
    if time.time() < deadline - 2:
        res = solve_optimized(13)
        if res:
            return res
            
    # Fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
