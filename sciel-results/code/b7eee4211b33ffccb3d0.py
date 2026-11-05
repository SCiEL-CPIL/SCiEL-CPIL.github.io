import random
import time
import math

def construct():
    n = 16
    edges = []
    edge_idx = {}
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[(i, j)] = len(edges)
            edges.append((i, j))
    
    k4s = []
    k4_edge_indices = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    k4s.append((a, b, c, d))
                    e_indices = [
                        edge_idx[(a, b)],
                        edge_idx[(a, c)],
                        edge_idx[(a, d)],
                        edge_idx[(b, c)],
                        edge_idx[(b, d)],
                        edge_idx[(c, d)]
                    ]
                    k4_edge_indices.append(e_indices)
    
    edge_to_k4s = [[] for _ in range(len(edges))]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)
            
    num_k4s = len(k4s)
    
    def count_violations(colors):
        v = 0
        for e_indices in k4_edge_indices:
            c_set = set()
            for e_idx in e_indices:
                c_set.add(colors[e_idx])
            if len(c_set) < 5:
                v += 1
        return v

    def get_num_colors(colors):
        return len(set(colors))

    def solve_sa(max_palette_size, time_limit):
        random.seed(42) # Fixed seed for reproducibility in this run
        colors = [random.randint(0, max_palette_size - 1) for _ in range(len(edges))]
        violations = count_violations(colors)
        num_colors = get_num_colors(colors)
        
        color_counts = [0] * max_palette_size
        for c in colors:
            color_counts[c] += 1
            
        best_colors = colors[:]
        best_violations = violations
        best_num_colors = num_colors
        
        start_time = time.time()
        T = 10.0
        cooling = 0.99995
        
        while time.time() < start_time + time_limit:
            e_idx = random.randint(0, len(edges) - 1)
            old_color = colors[e_idx]
            new_color = random.randint(0, max_palette_size - 1)
            
            if old_color == new_color:
                continue
            
            delta_v = 0
            for k4_idx in edge_to_k4s[e_idx]:
                e_indices = k4_edge_indices[k4_idx]
                c_list = [colors[e] for e in e_indices]
                
                old_set = set(c_list)
                old_len = len(old_set)
                
                new_set = set()
                for c in c_list:
                    if c == old_color:
                        new_set.add(new_color)
                    else:
                        new_set.add(c)
                new_len = len(new_set)
                
                old_violation = 1 if old_len < 5 else 0
                new_violation = 1 if new_len < 5 else 0
                
                if old_violation != new_violation:
                    delta_v += 1 if new_violation else -1
            
            new_violations = violations + delta_v
            
            delta_nc = 0
            if color_counts[old_color] == 1:
                delta_nc -= 1
            if color_counts[new_color] == 0:
                delta_nc += 1
            
            new_num_colors = num_colors + delta_nc
            
            old_cost = violations * 1000 + num_colors
            new_cost = new_violations * 1000 + new_num_colors
            
            if new_cost < old_cost:
                colors[e_idx] = new_color
                color_counts[old_color] -= 1
                color_counts[new_color] += 1
                violations = new_violations
                num_colors = new_num_colors
                
                if violations == 0:
                    if num_colors < best_num_colors:
                        best_num_colors = num_colors
                        best_violations = 0
                        best_colors = colors[:]
            else:
                delta = new_cost - old_cost
                if random.random() < math.exp(-delta / T):
                    colors[e_idx] = new_color
                    color_counts[old_color] -= 1
                    color_counts[new_color] += 1
                    violations = new_violations
                    num_colors = new_num_colors
                    
                    if violations == 0:
                        if num_colors < best_num_colors:
                            best_num_colors = num_colors
                            best_violations = 0
                            best_colors = colors[:]
            
            T *= cooling
            if T < 0.01:
                break
                
        return best_colors, best_num_colors

    best_colors, best_num = solve_sa(15, 115.0)
    
    return best_colors
