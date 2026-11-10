import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin (evaluator kills at 150s)
    
    # Try n from 34 down to 1
    for n in range(34, 0, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 3:
            break
            
        # Allocate time based on n
        if n == 34:
            budget = min(120, deadline - elapsed - 2)
        elif n >= 30:
            budget = min(15, deadline - elapsed - 2)
        else:
            budget = min(5, deadline - elapsed - 2)
            
        if budget < 2:
            continue
            
        result = find_coloring(n, budget, start_time, deadline)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback
    return {'n': 1, 'colors': []}

def find_coloring(n, time_budget, global_start, deadline):
    if n < 4:
        M = n * (n - 1) * (n - 2) // 6
        return [random.randint(0, 1) for _ in range(M)]
        
    # Precompute structure
    edges = list(combinations(range(n), 3))
    M = len(edges)
    edge_index = {e: i for i, e in enumerate(edges)}
    
    # K4s (4 vertices, 4 triples)
    k4_list = []
    for quad in combinations(range(n), 4):
        k4_list.append([edge_index[t] for t in combinations(quad, 3)])
        
    # K5s (5 vertices, 10 triples)
    k5_list = []
    if n >= 5:
        for quint in combinations(range(n), 5):
            k5_list.append([edge_index[t] for t in combinations(quint, 3)])
            
    num_k4 = len(k4_list)
    num_k5 = len(k5_list)
    
    # Map edge to k4 indices and k5 indices
    edge_to_k4s = [[] for _ in range(M)]
    for i, k4_edges in enumerate(k4_list):
        for e in k4_edges:
            edge_to_k4s[e].append(i)
            
    edge_to_k5s = [[] for _ in range(M)]
    for i, k5_edges in enumerate(k5_list):
        for e in k5_edges:
            edge_to_k5s[e].append(i)
    
    # Convert to numpy arrays for speed
    k4_edges_arr = [np.array(k4_list[i], dtype=np.int32) for i in range(num_k4)]
    k5_edges_arr = [np.array(k5_list[i], dtype=np.int32) for i in range(num_k5)] if num_k5 > 0 else []
    
    best_result = None
    best_viol = float('inf')
    
    # Multiple restarts
    max_restarts = 100
    restart = 0
    
    while restart < max_restarts:
        elapsed = time.time() - global_start
        if elapsed > deadline - 3:
            break
            
        # Initialize random coloring
        colors = np.random.randint(0, 2, M, dtype=np.int8)
        
        # Initialize counts
        k4_counts = np.zeros(num_k4, dtype=np.int32)
        for i in range(num_k4):
            k4_counts[i] = np.sum(colors[k4_edges_arr[i]] == 0)
            
        k5_counts = np.zeros(num_k5, dtype=np.int32) if num_k5 > 0 else np.array([], dtype=np.int32)
        if num_k5 > 0:
            for i in range(num_k5):
                k5_counts[i] = np.sum(colors[k5_edges_arr[i]] == 1)
                
        red_viol = int(np.sum(k4_counts == 4))
        blue_viol = int(np.sum(k5_counts == 10)) if num_k5 > 0 else 0
        
        if red_viol == 0 and blue_viol == 0:
            return colors.tolist()
            
        total_viol = red_viol + blue_viol
        if total_viol < best_viol:
            best_viol = total_viol
            best_result = colors.copy()
            
        # Simulated Annealing
        T = 2.0
        T_end = 0.01
        cooling = 0.9995
        
        no_improve = 0
        local_best_viol = total_viol
        local_best_colors = colors.copy()
        
        # Target: run until time budget for this restart is used
        # Each restart gets a fraction of the budget
        restart_time_limit = min(8.0, time_budget / max(1, max_restarts - restart) * 2)
        restart_start = time.time()
        
        while T > T_end:
            # Check global deadline
            if time.time() - global_start > deadline - 3:
                break
            # Check restart time limit
            if time.time() - restart_start > restart_time_limit:
                break
                
            e = np.random.randint(0, M)
            old_val = colors[e]
            
            # Calculate delta
            delta_red = 0
            delta_blue = 0
            
            if old_val == 0:  # Red -> Blue
                for k4_idx in edge_to_k4s[e]:
                    if k4_counts[k4_idx] == 4:
                        delta_red -= 1
                if num_k5 > 0:
                    for k5_idx in edge_to_k5s[e]:
                        if k5_counts[k5_idx] == 9:
                            delta_blue += 1
            else:  # Blue -> Red
                for k4_idx in edge_to_k4s[e]:
                    if k4_counts[k4_idx] == 3:
                        delta_red += 1
                if num_k5 > 0:
                    for k5_idx in edge_to_k5s[e]:
                        if k5_counts[k5_idx] == 10:
                            delta_blue -= 1
                            
            delta_total = delta_red + delta_blue
            new_total = total_viol + delta_total
            
            # Acceptance criterion
            if delta_total < 0 or (delta_total == 0) or (delta_total > 0 and random.random() < np.exp(-delta_total / T)):
                # Apply flip
                colors[e] = 1 - old_val
                if old_val == 0:
                    for k4_idx in edge_to_k4s[e]:
                        k4_counts[k4_idx] -= 1
                    if num_k5 > 0:
                        for k5_idx in edge_to_k5s[e]:
                            k5_counts[k5_idx] += 1
                else:
                    for k4_idx in edge_to_k4s[e]:
                        k4_counts[k4_idx] += 1
                    if num_k5 > 0:
                        for k5_idx in edge_to_k5s[e]:
                            k5_counts[k5_idx] -= 1
                            
                total_viol = new_total
                red_viol += delta_red
                blue_viol += delta_blue
                
                if total_viol < local_best_viol:
                    local_best_viol = total_viol
                    local_best_colors = colors.copy()
                    no_improve = 0
                    if total_viol == 0:
                        return colors.tolist()
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
            T *= cooling
            
            # If stuck for too long, break out of this restart
            if no_improve > 5000:
                break
        
        # Update global best
        if local_best_viol < best_viol:
            best_viol = local_best_viol
            best_result = local_best_colors.copy()
            
        restart += 1
        
        # If we found a valid solution, return immediately
        if best_viol == 0:
            return best_result.tolist()
            
    # Return best found or None
    if best_result is not None and best_viol == 0:
        return best_result.tolist()
        
    return None
