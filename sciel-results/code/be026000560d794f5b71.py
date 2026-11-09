"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = 135.0
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    def solve(n, max_steps=1500):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        idx_map = {t: i for i, t in enumerate(triples)}
        
        quad_list = list(combinations(range(n), 4))
        quint_list = list(combinations(range(n), 5))
        
        quad_tris = {quad: [idx_map[tri] for tri in combinations(quad, 3)] for quad in quad_list}
        quint_tris = {quint: [idx_map[tri] for tri in combinations(quint, 3)] for quint in quint_list}
        
        triple_to_quads = [[] for _ in range(num_triples)]
        for quad in quad_list:
            for ti in quad_tris[quad]:
                triple_to_quads[ti].append(quad)
        
        triple_to_quints = [[] for _ in range(num_triples)]
        for quint in quint_list:
            for ti in quint_tris[quint]:
                triple_to_quints[ti].append(quint)
        
        colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        
        red_v = sum(1 for q in quad_list if all(colors[i] == 0 for i in quad_tris[q]))
        blue_v = sum(1 for q in quint_list if all(colors[i] == 1 for i in quint_tris[q]))
        
        best_viol = red_v + blue_v
        best_colors = colors.copy()
        
        for step in range(max_steps):
            if time_left() < 2.0:
                break
            
            if red_v + blue_v == 0:
                return colors.tolist()
            
            if red_v + blue_v < best_viol:
                best_viol = red_v + blue_v
                best_colors = colors.copy()
            
            if red_v > 0 and (blue_v == 0 or random.random() < red_v / (red_v + blue_v)):
                while True:
                    quad = random.choice(quad_list)
                    if all(colors[i] == 0 for i in quad_tris[quad]):
                        break
                tri_idx = random.choice(quad_tris[quad])
            else:
                while True:
                    quint = random.choice(quint_list)
                    if all(colors[i] == 1 for i in quint_tris[quint]):
                        break
                tri_idx = random.choice(quint_tris[quint])
            
            affected_quads = triple_to_quads[tri_idx]
            affected_quints = triple_to_quints[tri_idx]
            
            old_quad_viol = {q: all(colors[i] == 0 for i in quad_tris[q]) for q in affected_quads}
            old_quint_viol = {q: all(colors[i] == 1 for i in quint_tris[q]) for q in affected_quints}
            
            colors[tri_idx] = 1 - colors[tri_idx]
            
            for q in affected_quads:
                new_viol = all(colors[i] == 0 for i in quad_tris[q])
                if new_viol and not old_quad_viol[q]:
                    red_v += 1
                elif not new_viol and old_quad_viol[q]:
                    red_v -= 1
            
            for q in affected_quints:
                new_viol = all(colors[i] == 1 for i in quint_tris[q])
                if new_viol and not old_quint_viol[q]:
                    blue_v += 1
                elif not new_viol and old_quint_viol[q]:
                    blue_v -= 1
        
        return None
    
    best_n = 9
    best_colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in combinations(range(9), 3)]
    
    for target_n in range(10, 35):
        if time_left() < 10.0:
            break
        
        result = solve(target_n, max_steps=1500)
        if result is not None:
            best_n = target_n
            best_colors = result
            break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
