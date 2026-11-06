import time
import random
import itertools
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Fallback: trivial valid coloring for n=9
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(9), 3)]
    
    # Try to find a valid coloring for n from 34 down to 10
    for n in range(34, 9, -1):
        if time.time() > deadline - 20.0:
            break
        
        try:
            edges = list(itertools.combinations(range(n), 3))
            num_edges = len(edges)
            edge_index = {e: i for i, e in enumerate(edges)}
            
            sets4 = list(itertools.combinations(range(n), 4))
            sets5 = list(itertools.combinations(range(n), 5))
            num_sets4 = len(sets4)
            num_sets5 = len(sets5)
            
            # Build incidence: edge -> list of 4-set indices, edge -> list of 5-set indices
            edge_to_4sets = [[] for _ in range(num_edges)]
            for s4_idx, s4 in enumerate(sets4):
                for i in range(4):
                    triple = s4[:i] + s4[i+1:]
                    e_idx = edge_index[triple]
                    edge_to_4sets[e_idx].append(s4_idx)
                    
            edge_to_5sets = [[] for _ in range(num_edges)]
            for s5_idx, s5 in enumerate(sets5):
                for triple in itertools.combinations(s5, 3):
                    e_idx = edge_index[triple]
                    edge_to_5sets[e_idx].append(s5_idx)
                    
        except Exception:
            continue

        # Simulated Annealing
        found = False
        max_restarts = 8
        for restart in range(max_restarts):
            if time.time() > deadline - 10.0:
                break
                
            # Initialize with random coloring
            colors = np.random.randint(0, 2, size=num_edges)
            
            # Compute initial violation counts
            red_count_4 = np.zeros(num_sets4, dtype=np.int32)
            blue_count_5 = np.zeros(num_sets5, dtype=np.int32)
            
            for e_idx in range(num_edges):
                c = colors[e_idx]
                if c == 0:  # Red
                    for s4_idx in edge_to_4sets[e_idx]:
                        red_count_4[s4_idx] += 1
                else:  # Blue
                    for s5_idx in edge_to_5sets[e_idx]:
                        blue_count_5[s5_idx] += 1
                        
            viol4 = np.sum(red_count_4 == 4)
            viol5 = np.sum(blue_count_5 == 10)
            total_viol = viol4 + viol5
            
            if total_viol == 0:
                found = True
                break
                
            # Simulated Annealing parameters
            T = 1.0
            T_min = 0.01
            cooling = 0.995
            
            # We'll do multiple passes over edges, flipping randomly
            steps_per_pass = num_edges
            max_passes = 200
            
            for p in range(max_passes):
                if time.time() > deadline - 5.0:
                    break
                    
                if T < T_min:
                    T = T_min * 1.5  # Reheat slightly
                
                # Pick a random edge to flip
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                new_color = 1 - old_color
                
                # Compute delta in violations
                delta_viol = 0.0
                
                if old_color == 0:  # Red -> Blue
                    for s4_idx in edge_to_4sets[e_idx]:
                        old_c = red_count_4[s4_idx]
                        new_c = old_c - 1
                        if old_c == 4:
                            delta_viol -= 1.0
                        if new_c == 4:
                            delta_viol += 1.0
                        red_count_4[s4_idx] = new_c
                    for s5_idx in edge_to_5sets[e_idx]:
                        old_c = blue_count_5[s5_idx]
                        new_c = old_c + 1
                        if old_c == 10:
                            delta_viol -= 1.0
                        if new_c == 10:
                            delta_viol += 1.0
                        blue_count_5[s5_idx] = new_c
                else:  # Blue -> Red
                    for s5_idx in edge_to_5sets[e_idx]:
                        old_c = blue_count_5[s5_idx]
                        new_c = old_c - 1
                        if old_c == 10:
                            delta_viol -= 1.0
                        if new_c == 10:
                            delta_viol += 1.0
                        blue_count_5[s5_idx] = new_c
                    for s4_idx in edge_to_4sets[e_idx]:
                        old_c = red_count_4[s4_idx]
                        new_c = old_c + 1
                        if old_c == 4:
                            delta_viol -= 1.0
                        if new_c == 4:
                            delta_viol += 1.0
                        red_count_4[s4_idx] = new_c
                
                total_viol += delta_viol
                
                # Accept if improvement, or with probability exp(-delta/T)
                if delta_viol < 0 or (T > 0 and random.random() < np.exp(-delta_viol / T)):
                    colors[e_idx] = new_color
                    if total_viol == 0:
                        found = True
                        break
                else:
                    # Rollback
                    if old_color == 0:
                        for s4_idx in edge_to_4sets[e_idx]:
                            red_count_4[s4_idx] += 1
                        for s5_idx in edge_to_5sets[e_idx]:
                            blue_count_5[s5_idx] -= 1
                    else:
                        for s5_idx in edge_to_5sets[e_idx]:
                            blue_count_5[s5_idx] += 1
                        for s4_idx in edge_to_4sets[e_idx]:
                            red_count_4[s4_idx] -= 1
                    total_viol -= delta_viol
                
                T *= cooling
                
                if found:
                    break
                    
            if found:
                best_n = n
                best_colors = colors.tolist()
                break
        
        if found:
            break
    
    return {'n': best_n, 'colors': best_colors}
