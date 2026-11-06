import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Try n from 34 downwards
    for n in [34, 33, 32, 31]:
        if time.time() > deadline - 5.0:
            break
            
        # Precomputation for this n
        edges = list(combinations(range(n), 3))
        m = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # Map each edge to the quads and pents it belongs to
        quads = list(combinations(range(n), 4))
        pents = list(combinations(range(n), 5))
        
        edge_to_quads = [[] for _ in range(m)]
        for q_idx, q in enumerate(quads):
            q_edges = list(combinations(q, 3))
            for e in q_edges:
                e_idx = edge_to_idx[e]
                edge_to_quads[e_idx].append(q_idx)
                
        edge_to_pents = [[] for _ in range(m)]
        for p_idx, p in enumerate(pents):
            p_edges = list(combinations(p, 3))
            for e in p_edges:
                e_idx = edge_to_idx[e]
                edge_to_pents[e_idx].append(p_idx)

        best_violations = float('inf')
        best_colors = None
        
        # Simulated annealing for this n
        # Use most of the remaining time for n=34
        remaining = deadline - time.time()
        if n == 34:
            search_deadline = time.time() + remaining * 0.85
        else:
            search_deadline = time.time() + min(remaining, 10.0)
        
        # Temperature schedule
        T_start = 2.0
        T_end = 0.01
        T_factor = (T_end / T_start) ** (1.0 / max(1, int((search_deadline - time.time()) / 0.001)))
        
        while time.time() < search_deadline:
            colors = [random.randint(0, 1) for _ in range(m)]
            
            # Calculate initial violations
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
                
            if violations < best_violations:
                best_violations = violations
                best_colors = colors[:]
                
            # Simulated annealing inner loop
            T = T_start
            no_improve_count = 0
            while time.time() < search_deadline:
                # Pick a random edge to flip
                i = random.randint(0, m - 1)
                old_c = colors[i]
                new_c = 1 - old_c
                
                # Calculate delta violations
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
                
                # Accept with probability based on temperature
                if delta_v <= 0:
                    accept = True
                else:
                    # Boltzmann acceptance
                    if T > 0:
                        accept = (random.random() < math.exp(-delta_v / T))
                    else:
                        accept = False
                
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
                    
                    if violations == 0:
                        return {'n': n, 'colors': colors}
                        
                    if violations < best_violations:
                        best_violations = violations
                        best_colors = colors[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                
                # Cool down
                T *= T_factor
                if T < T_end:
                    T = T_end
                
                # If stuck for too long, restart with new random init
                if no_improve_count > 5000:
                    break

    # If no valid coloring found, return n=1 (trivially valid)
    return {'n': 1, 'colors': []}
