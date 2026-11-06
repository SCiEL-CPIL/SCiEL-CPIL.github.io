import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Time allocation strategy
    # n=34: 60s
    # n=33: 40s
    # n=32: 10s
    # n=31: 5s
    
    targets = [
        (34, 60),
        (33, 40),
        (32, 10),
        (31, 5)
    ]
    
    for n, time_limit in targets:
        if time.time() > deadline - 2:
            break
            
        result = solve_fast(n, time_limit=time_limit)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback
    return {'n': 1, 'colors': [0]}

def solve_fast(n, time_limit=5):
    start_time = time.time()
    deadline = start_time + time_limit
    
    edges = list(combinations(range(n), 3))
    num_edges = len(edges)
    edge_index = {e: i for i, e in enumerate(edges)}
    
    k4s = list(combinations(range(n), 4))
    k5s = list(combinations(range(n), 5))
    
    # Precompute K4 edge lists
    k4_edge_lists = []
    for k4 in k4s:
        a, b, c, d = k4
        e1 = edge_index[(a,b,c)]
        e2 = edge_index[(a,b,d)]
        e3 = edge_index[(a,c,d)]
        e4 = edge_index[(b,c,d)]
        k4_edge_lists.append((e1, e2, e3, e4))
        
    # Precompute K5 edge lists
    k5_edge_lists = []
    for k5 in k5s:
        k5_edge_lists.append([edge_index[e] for e in combinations(k5, 3)])
        
    # Map edge to K4s and K5s it belongs to, with "others" precomputed
    # For K4: others are 3 edges.
    # For K5: others are 9 edges.
    
    edge_to_k4s = [[] for _ in range(num_edges)]
    edge_to_k5s = [[] for _ in range(num_edges)]
    
    for k4_idx, (e1, e2, e3, e4) in enumerate(k4_edge_lists):
        edge_to_k4s[e1].append((k4_idx, (e2, e3, e4)))
        edge_to_k4s[e2].append((k4_idx, (e1, e3, e4)))
        edge_to_k4s[e3].append((k4_idx, (e1, e2, e4)))
        edge_to_k4s[e4].append((k4_idx, (e1, e2, e3)))
        
    for k5_idx, e_list in enumerate(k5_edge_lists):
        for pos, e_idx in enumerate(e_list):
            others = tuple(e_list[j] for j in range(10) if j != pos)
            edge_to_k5s[e_idx].append((k5_idx, others))
            
    # Initial random coloring
    colors = [random.randint(0, 1) for _ in range(num_edges)]
    
    # Initial violations
    total_violations = 0
    for e1, e2, e3, e4 in k4_edge_lists:
        if colors[e1]==0 and colors[e2]==0 and colors[e3]==0 and colors[e4]==0:
            total_violations += 1
    for e_list in k5_edge_lists:
        if all(colors[e]==1 for e in e_list):
            total_violations += 1
            
    if total_violations == 0:
        return colors
        
    best_colors = colors[:]
    best_violations = total_violations
    
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        new_color = 1 - old_color
        
        delta_v = 0
        
        # K4s
        for k4_idx, others in edge_to_k4s[e_idx]:
            o1, o2, o3 = others
            if old_color == 0:
                # Red -> Blue
                # Was Red violation if all 4 were Red.
                # Now e is Blue, so not Red violation.
                # It WAS a violation if others were all Red.
                if colors[o1]==0 and colors[o2]==0 and colors[o3]==0:
                    delta_v -= 1
            else:
                # Blue -> Red
                # Becomes Red violation if others are all Red.
                if colors[o1]==0 and colors[o2]==0 and colors[o3]==0:
                    delta_v += 1
                    
        # K5s
        for k5_idx, others in edge_to_k5s[e_idx]:
            # others is a tuple of 9 indices
            if old_color == 1:
                # Blue -> Red
                # Was Blue violation if all 10 were Blue.
                # Now e is Red, so not Blue violation.
                # It WAS a violation if others were all Blue.
                if all(colors[o]==1 for o in others):
                    delta_v -= 1
            else:
                # Red -> Blue
                # Becomes Blue violation if others are all Blue.
                if all(colors[o]==1 for o in others):
                    delta_v += 1
                    
        new_v = total_violations + delta_v
        
        if delta_v <= 0 or random.random() < math.exp(-delta_v / T):
            colors[e_idx] = new_color
            total_violations = new_v
            
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return best_colors
                    
        T *= cooling
        
    if best_violations == 0:
        return best_colors
    return None
