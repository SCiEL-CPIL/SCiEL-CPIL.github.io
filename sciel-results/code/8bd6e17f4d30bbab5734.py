import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def solve_full(n):
        edges = list(combinations(range(n), 3))
        m = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        quads = list(combinations(range(n), 4))
        pents = list(combinations(range(n), 5))
        
        edge_to_quads = [[] for _ in range(m)]
        for q_idx, q in enumerate(quads):
            for e in combinations(q, 3):
                edge_to_quads[edge_to_idx[e]].append(q_idx)
                
        edge_to_pents = [[] for _ in range(m)]
        for p_idx, p in enumerate(pents):
            for e in combinations(p, 3):
                edge_to_pents[edge_to_idx[e]].append(p_idx)

        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        
        restart_count = 0
        max_restarts = 50
        
        while time.time() < deadline - 5.0 and restart_count < max_restarts:
            restart_count += 1
            T = 1.0
            
            # Random initialization
            colors = [random.randint(0, 1) for _ in range(m)]
            
            quad_red_counts = [0] * len(quads)
            pent_blue_counts = [0] * len(pents)
            
            for i, c in enumerate(colors):
                if c == 0: # Red
                    for q_idx in edge_to_quads[i]:
                        quad_red_counts[q_idx] += 1
                else: # Blue
                    for p_idx in edge_to_pents[i]:
                        pent_blue_counts[p_idx] += 1
                        
            violations = sum(1 for c in quad_red_counts if c == 4) + sum(1 for c in pent_blue_counts if c == 10)
            
            if violations == 0:
                return {'n': n, 'colors': colors}
                
            # Local search loop
            no_improve = 0
            max_no_improve = 10000
            while time.time() < deadline - 5.0 and no_improve < max_no_improve:
                i = random.randint(0, m - 1)
                old_c = colors[i]
                new_c = 1 - old_c
                
                delta_v = 0
                
                if old_c == 0: # Red -> Blue
                    for q_idx in edge_to_quads[i]:
                        if quad_red_counts[q_idx] == 4:
                            delta_v -= 1
                    for p_idx in edge_to_pents[i]:
                        if pent_blue_counts[p_idx] == 9:
                            delta_v += 1
                else: # Blue -> Red
                    for q_idx in edge_to_quads[i]:
                        if quad_red_counts[q_idx] == 3:
                            delta_v += 1
                    for p_idx in edge_to_pents[i]:
                        if pent_blue_counts[p_idx] == 10:
                            delta_v -= 1
                
                accept = False
                if delta_v < 0:
                    accept = True
                elif delta_v == 0:
                    accept = True
                else:
                    accept = random.random() < (2.718281828 ** (-delta_v / T))
                
                if accept:
                    colors[i] = new_c
                    if new_c == 0: # Now Red
                        for q_idx in edge_to_quads[i]:
                            quad_red_counts[q_idx] += 1
                        for p_idx in edge_to_pents[i]:
                            pent_blue_counts[p_idx] -= 1
                    else: # Now Blue
                        for q_idx in edge_to_quads[i]:
                            quad_red_counts[q_idx] -= 1
                        for p_idx in edge_to_pents[i]:
                            pent_blue_counts[p_idx] += 1
                            
                    violations += delta_v
                    no_improve = 0
                    T *= cooling
                    if T < T_min:
                        T = T_min
                    
                    if violations == 0:
                        return {'n': n, 'colors': colors}
                else:
                    no_improve += 1
        
        return None

    # Try n=34
    result = solve_full(34)
    if result:
        return result
        
    # Try smaller n
    for n in range(33, 1, -1):
        if time.time() > deadline - 10.0:
            break
        result = solve_full(n)
        if result:
            return result
            
    return {'n': 1, 'colors': []}

