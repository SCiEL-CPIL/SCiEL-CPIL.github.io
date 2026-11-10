import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: i for i, e in enumerate(edges)}
    
    k4s = list(combinations(range(n), 4))
    k4_edges = []
    for k4 in k4s:
        es = list(combinations(k4, 2))
        k4_edges.append([edge_index[e] for e in es])
        
    edge_k4s = [[] for _ in range(len(edges))]
    for i, k4_e in enumerate(k4_edges):
        for e_idx in k4_e:
            edge_k4s[e_idx].append(i)
            
    MAX_COLORS = 20
    num_edges = len(edges)
    num_k4s = len(k4s)
    
    def solve(start_colors):
        colors = [0] * num_edges
        global_counts = [0] * MAX_COLORS
        k4_counts = [[0] * MAX_COLORS for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        
        # Init
        for i in range(num_edges):
            c = random.randint(0, start_colors - 1)
            colors[i] = c
            global_counts[c] += 1
            for k4_idx in edge_k4s[i]:
                k4_counts[k4_idx][c] += 1
                
        for k4_idx in range(num_k4s):
            k4_distinct[k4_idx] = sum(1 for c in range(MAX_COLORS) if k4_counts[k4_idx][c] > 0)
            
        current_violations = sum(1 for d in k4_distinct if d < 5)
        current_used = sum(1 for c in range(MAX_COLORS) if global_counts[c] > 0)
        current_cost = 10000 * current_violations + current_used
        
        best_cost = current_cost
        best_colors = colors[:]
        
        temp = 1.0
        cooling = 0.99995
        min_temp = 0.01
        
        while time.time() < deadline and temp > min_temp:
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            new_color = random.randint(0, start_colors - 1)
            
            if new_color == old_color:
                continue
                
            # Apply
            affected_k4s = edge_k4s[e_idx]
            
            # Update global
            global_counts[old_color] -= 1
            global_counts[new_color] += 1
            
            # Update K4s
            for k4_idx in affected_k4s:
                # Old color
                k4_counts[k4_idx][old_color] -= 1
                if k4_counts[k4_idx][old_color] == 0:
                    k4_distinct[k4_idx] -= 1
                # New color
                k4_counts[k4_idx][new_color] += 1
                if k4_counts[k4_idx][new_color] == 1:
                    k4_distinct[k4_idx] += 1
                    
            # Calculate new cost
            # To avoid summing all K4s, we can track violations incrementally?
            # It's easier to just sum if num_k4s is 1820. 1820 ops is fast.
            new_violations = sum(1 for d in k4_distinct if d < 5)
            new_used = sum(1 for c in range(MAX_COLORS) if global_counts[c] > 0)
            new_cost = 10000 * new_violations + new_used
            
            if new_cost < current_cost or (random.random() < temp and new_cost < current_cost + 50):
                current_cost = new_cost
                current_violations = new_violations
                current_used = new_used
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            else:
                # Revert
                colors[e_idx] = old_color
                global_counts[old_color] += 1
                global_counts[new_color] -= 1
                for k4_idx in affected_k4s:
                    k4_counts[k4_idx][new_color] -= 1
                    if k4_counts[k4_idx][new_color] == 0:
                        k4_distinct[k4_idx] -= 1
                    k4_counts[k4_idx][old_color] += 1
                    if k4_counts[k4_idx][old_color] == 1:
                        k4_distinct[k4_idx] += 1
                            
            temp *= cooling
            
        return best_colors, best_cost

    best_overall_colors = None
    best_overall_cost = float('inf')
    
    for sc in [13, 14, 15]:
        if time.time() > deadline:
            break
        cols, cost = solve(sc)
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_colors = cols
            
    if best_overall_colors is None:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
    return best_overall_colors
