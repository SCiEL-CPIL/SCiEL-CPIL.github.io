"""Rewritten program for R(4,5;3) hypergraph Ramsey construction using constraint propagation and local search."""
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    def get_precomputed(n):
        """Precompute triple indices and K4/K5 violation structures."""
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # K4 structures: each set of 4 vertices has 4 triples
        k4_list = []
        for k4 in combinations(range(n), 4):
            ts = []
            for i in range(4):
                t = tuple(sorted(k4[:i] + k4[i+1:]))
                ts.append(triple_to_idx[t])
            k4_list.append(ts)
        k4_arr = np.array(k4_list, dtype=np.int32) if k4_list else np.empty((0, 4), dtype=np.int32)
        
        # K5 structures: each set of 5 vertices has 10 triples
        k5_list = []
        for k5 in combinations(range(n), 5):
            ts = []
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[i] for i in combo]))
                ts.append(triple_to_idx[t])
            k5_list.append(ts)
        k5_arr = np.array(k5_list, dtype=np.int32) if k5_list else np.empty((0, 10), dtype=np.int32)
        
        return triples, num_triples, k4_arr, k5_arr

    def count_violations(colors, k4_arr, k5_arr):
        """Count red K4 and blue K5 violations."""
        if len(k4_arr) > 0:
            red_k4 = np.sum(colors[k4_arr], axis=1)
            red_viol = int(np.sum(red_k4 == 0))
        else:
            red_viol = 0
        
        if len(k5_arr) > 0:
            blue_k5 = np.sum(colors[k5_arr], axis=1)
            blue_viol = int(np.sum(blue_k5 == 10))
        else:
            blue_viol = 0
        
        return red_viol + blue_viol

    def local_search(n, k4_arr, k5_arr, time_budget, initial=None):
        """
        Local search with smart variable selection based on constraint tightness.
        Uses constraint propagation to guide which triples to flip.
        """
        num_triples = n * (n-1) * (n-2) // 6
        
        if initial is not None:
            colors = initial.copy()
        else:
            colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        
        current_viol = count_violations(colors, k4_arr, k5_arr)
        best_colors = colors.copy()
        best_viol = current_viol
        
        ls_start = time.time()
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < ls_start + time_budget:
            if current_viol == 0:
                break
            
            # Compute constraint pressure for each triple
            # For each triple, count how many K4s it's in that are "almost" red
            # and how many K5s it's in that are "almost" blue
            
            # Efficient approach: compute which K4s are close to violation
            if len(k4_arr) > 0:
                red_k4_sums = np.sum(colors[k4_arr], axis=1)
                # K4s with 0 or 1 red triples are dangerous
                dangerous_k4 = np.where(red_k4_sums <= 1)[0]
            else:
                dangerous_k4 = np.array([], dtype=np.int32)
            
            if len(k5_arr) > 0:
                blue_k5_sums = np.sum(colors[k5_arr], axis=1)
                # K5s with 8 or 9 blue triples are dangerous
                dangerous_k5 = np.where(blue_k5_sums >= 8)[0]
            else:
                dangerous_k5 = np.array([], dtype=np.int32)
            
            # Build a score for each triple based on how many dangerous structures it participates in
            triple_pressure = np.zeros(num_triples, dtype=np.int32)
            
            if len(dangerous_k4) > 0:
                for row in dangerous_k4:
                    for tri_idx in k4_arr[row]:
                        triple_pressure[tri_idx] += 1
            
            if len(dangerous_k5) > 0:
                for row in dangerous_k5:
                    for tri_idx in k5_arr[row]:
                        triple_pressure[tri_idx] += 1
            
            # Select triples to flip: prioritize high-pressure triples
            if np.any(triple_pressure > 0):
                # Weighted random selection based on pressure
                weights = triple_pressure.astype(np.float64)
                total_w = np.sum(weights)
                if total_w > 0:
                    probs = weights / total_w
                    # Flip 1-3 high-pressure triples
                    num_flips = random.randint(1, 3)
                    flip_indices = np.random.choice(num_triples, size=min(num_flips, num_triples), 
                                                      replace=False, p=probs)
                    for idx in flip_indices:
                        colors[idx] = 1 - colors[idx]
                else:
                    # Fallback to random
                    idx = random.randint(0, num_triples - 1)
                    colors[idx] = 1 - colors[idx]
            else:
                # No obvious pressure, do random flip
                num_flips = random.randint(1, 2)
                for _ in range(num_flips):
                    idx = random.randint(0, num_triples - 1)
                    colors[idx] = 1 - colors[idx]
            
            # Evaluate new state
            new_viol = count_violations(colors, k4_arr, k5_arr)
            
            if new_viol < current_viol:
                current_viol = new_viol
                best_colors = colors.copy()
                best_viol = current_viol
                no_improve_count = 0
            else:
                # Accept with some probability (simulated annealing style)
                # or reject
                if random.random() < 0.3:
                    current_viol = new_viol
                else:
                    # Revert
                    if np.any(triple_pressure > 0) and total_w > 0:
                        for idx in flip_indices:
                            colors[idx] = 1 - colors[idx]
                    else:
                        for _ in range(num_flips):
                            idx = random.randint(0, num_triples - 1)
                            colors[idx] = 1 - colors[idx]
                    no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Restart with some perturbation
                perturbation_size = random.randint(num_triples // 20, num_triples // 10)
                for _ in range(perturbation_size):
                    idx = random.randint(0, num_triples - 1)
                    colors[idx] = 1 - colors[idx]
                current_viol = count_violations(colors, k4_arr, k5_arr)
                no_improve_count = 0
        
        return best_colors, best_viol

    def constraint_guided_initialization(n, k4_arr, k5_arr):
        """
        Initialize coloring using constraint propagation logic.
        Start with a balanced random assignment, then apply constraint-driven adjustments.
        """
        num_triples = n * (n-1) * (n-2) // 6
        colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        
        # Apply a few rounds of constraint-based repair
        for _ in range(20):
            v = count_violations(colors, k4_arr, k5_arr)
            if v == 0:
                break
            
            # Find and fix the most violating structures
            if len(k4_arr) > 0:
                red_k4_sums = np.sum(colors[k4_arr], axis=1)
                viol_k4 = np.where(red_k4_sums == 0)[0]
                if len(viol_k4) > 0:
                    row = viol_k4[random.randint(0, len(viol_k4)-1)]
                    # Flip one of the triples in this K4 to blue
                    tri_idx = k4_arr[row][random.randint(0, 3)]
                    colors[tri_idx] = 1
            
            if len(k5_arr) > 0:
                blue_k5_sums = np.sum(colors[k5_arr], axis=1)
                viol_k5 = np.where(blue_k5_sums == 10)[0]
                if len(viol_k5) > 0:
                    row = viol_k5[random.randint(0, len(viol_k5)-1)]
                    # Flip one of the triples in this K5 to red
                    tri_idx = k5_arr[row][random.randint(0, 9)]
                    colors[tri_idx] = 0
        
        return colors

    # Main search strategy
    best_n = 9
    best_colors = None
    
    # Try n values from high to low, spending more time on higher n
    # Published construction is 34, so target 34 first
    
    n_targets = [34, 33, 32, 31, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 9]
    
    for n in n_targets:
        if time.time() > deadline - 10.0:
            break
        
        remaining_time = deadline - time.time()
        # Allocate time based on n: larger n gets more time
        if n >= 30:
            time_budget = min(remaining_time * 0.4, 40.0)
        elif n >= 20:
            time_budget = min(remaining_time * 0.3, 25.0)
        else:
            time_budget = min(remaining_time * 0.2, 15.0)
        
        if time_budget < 2.0:
            break
        
        triples, num_triples, k4_arr, k5_arr = get_precomputed(n)
        
        # Run multiple restarts of local search
        best_for_n = None
        best_viol_for_n = float('inf')
        
        num_restarts = max(1, int(time_budget / 5.0))
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            restart_budget = (deadline - 5.0 - time.time()) / (num_restarts - r)
            
            # Use constraint-guided initialization for first few restarts
            if r < 3:
                initial = constraint_guided_initialization(n, k4_arr, k5_arr)
            else:
                initial = None
            
            colors, viol = local_search(n, k4_arr, k5_arr, restart_budget, initial)
            
            if viol < best_viol_for_n:
                best_viol_for_n = viol
                best_for_n = colors
            
            if viol == 0:
                best_for_n = colors
                break
        
        if best_for_n is not None and best_viol_for_n == 0:
            best_n = n
            best_colors = best_for_n.tolist()
            break
        elif best_for_n is not None and best_viol_for_n < best_viol_for_n:
            # Keep track of best partial result
            if best_colors is None or best_viol_for_n < 0:
                best_n = n
                best_colors = best_for_n.tolist()
    
    # Fallback: return a valid small coloring
    if best_colors is None:
        n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        best_n = n
    
    return {'n': best_n, 'colors': best_colors}
