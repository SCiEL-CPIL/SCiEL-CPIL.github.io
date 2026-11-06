import time
import numpy as np
from itertools import combinations
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    def get_triple_index(a, b, c):
        return comb(c, 3) + comb(b, 2) + a

    def search_for_n(n, time_limit):
        M = comb(n, 3)
        triples = list(combinations(range(n), 3))
        
        # Create a dictionary mapping triple tuple to index
        tri_to_idx = {tri: i for i, tri in enumerate(triples)}
        
        # Precompute which triples are in each 4-set and 5-set
        triples_in_4sets = []
        for quad in combinations(range(n), 4):
            t_indices = []
            for i in range(4):
                tri = tuple(x for j, x in enumerate(quad) if j != i)
                t_indices.append(tri_to_idx[tri])
            triples_in_4sets.append(t_indices)
            
        triples_in_5sets = []
        for quint in combinations(range(n), 5):
            t_indices = []
            for tri in combinations(quint, 3):
                t_indices.append(tri_to_idx[tri])
            triples_in_5sets.append(t_indices)
            
        triples_in_4sets = np.array(triples_in_4sets, dtype=int)
        triples_in_5sets = np.array(triples_in_5sets, dtype=int)
        
        # For each triple, precompute which 4-sets and 5-sets it belongs to
        # This allows fast cost update when flipping a triple
        triple_to_4sets = [[] for _ in range(M)]
        for quad_idx, t_indices in enumerate(triples_in_4sets):
            for t_idx in t_indices:
                triple_to_4sets[t_idx].append(quad_idx)
                
        triple_to_5sets = [[] for _ in range(M)]
        for quint_idx, t_indices in enumerate(triples_in_5sets):
            for t_idx in t_indices:
                triple_to_5sets[t_idx].append(quint_idx)
        
        # Initial coloring: use modular heuristic
        colors = np.zeros(M, dtype=np.int8)
        for i, (a, b, c) in enumerate(triples):
            colors[i] = 0 if (a + b + c) % 3 == 0 else 1
        
        # Track violations: for each 4-set, count how many triples are red (0)
        # For each 5-set, count how many triples are blue (1)
        red_counts_4 = np.zeros(len(triples_in_4sets), dtype=np.int8)
        blue_counts_5 = np.zeros(len(triples_in_5sets), dtype=np.int8)
        
        for quad_idx, t_indices in enumerate(triples_in_4sets):
            red_counts_4[quad_idx] = np.sum(colors[t_indices] == 0)
            
        for quint_idx, t_indices in enumerate(triples_in_5sets):
            blue_counts_5[quint_idx] = np.sum(colors[t_indices] == 1)
        
        def get_current_violations():
            red_k4_violations = np.sum(red_counts_4 == 4)
            blue_k5_violations = np.sum(blue_counts_5 == 10)
            return red_k4_violations + blue_k5_violations
        
        current_violations = get_current_violations()
        
        if current_violations == 0:
            return colors.tolist()
        
        # Greedy: repeatedly find the triple whose flip reduces violations the most
        max_iterations = 100000
        iteration = 0
        
        while iteration < max_iterations and time.time() < start_time + time_limit:
            iteration += 1
            
            # Find best flip: for each triple, compute delta in violations if we flip it
            best_delta = 0
            best_idx = -1
            
            for t_idx in range(M):
                old_val = colors[t_idx]
                new_val = 1 - old_val
                
                delta = 0
                
                # Check effect on 4-sets containing this triple
                for quad_idx in triple_to_4sets[t_idx]:
                    old_red = red_counts_4[quad_idx]
                    if old_val == 0:
                        # Was red, now blue: red count decreases by 1
                        new_red = old_red - 1
                        # Violation if old_red==4 and new_red!=4, or vice versa
                        if old_red == 4 and new_red != 4:
                            delta -= 1  # removes a violation
                        elif old_red != 4 and new_red == 4:
                            delta += 1  # adds a violation
                    else:
                        # Was blue, now red: red count increases by 1
                        new_red = old_red + 1
                        if old_red == 4 and new_red != 4:
                            delta -= 1
                        elif old_red != 4 and new_red == 4:
                            delta += 1
                
                # Check effect on 5-sets containing this triple
                for quint_idx in triple_to_5sets[t_idx]:
                    old_blue = blue_counts_5[quint_idx]
                    if old_val == 1:
                        # Was blue, now red: blue count decreases by 1
                        new_blue = old_blue - 1
                        if old_blue == 10 and new_blue != 10:
                            delta -= 1
                        elif old_blue != 10 and new_blue == 10:
                            delta += 1
                    else:
                        # Was red, now blue: blue count increases by 1
                        new_blue = old_blue + 1
                        if old_blue == 10 and new_blue != 10:
                            delta -= 1
                        elif old_blue != 10 and new_blue == 10:
                            delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_idx = t_idx
            
            if best_idx == -1:
                # No improving move found; try random restarts or break
                break
            
            # Apply the best flip
            old_val = colors[best_idx]
            new_val = 1 - old_val
            colors[best_idx] = new_val
            
            # Update counts
            for quad_idx in triple_to_4sets[best_idx]:
                if old_val == 0:
                    red_counts_4[quad_idx] -= 1
                else:
                    red_counts_4[quad_idx] += 1
                    
            for quint_idx in triple_to_5sets[best_idx]:
                if old_val == 1:
                    blue_counts_5[quint_idx] -= 1
                else:
                    blue_counts_5[quint_idx] += 1
            
            current_violations = get_current_violations()
            if current_violations == 0:
                return colors.tolist()
                
        # If greedy didn't work, try simulated annealing as fallback
        T = 1.0
        cooling = 0.99995
        min_T = 0.005
        
        while time.time() < start_time + time_limit:
            t_idx = np.random.randint(0, M)
            old_val = colors[t_idx]
            new_val = 1 - old_val
            
            # Compute delta
            delta = 0
            for quad_idx in triple_to_4sets[t_idx]:
                old_red = red_counts_4[quad_idx]
                if old_val == 0:
                    new_red = old_red - 1
                    if old_red == 4 and new_red != 4:
                        delta -= 1
                    elif old_red != 4 and new_red == 4:
                        delta += 1
                else:
                    new_red = old_red + 1
                    if old_red == 4 and new_red != 4:
                        delta -= 1
                    elif old_red != 4 and new_red == 4:
                        delta += 1
                        
            for quint_idx in triple_to_5sets[t_idx]:
                old_blue = blue_counts_5[quint_idx]
                if old_val == 1:
                    new_blue = old_blue - 1
                    if old_blue == 10 and new_blue != 10:
                        delta -= 1
                    elif old_blue != 10 and new_blue == 10:
                        delta += 1
                else:
                    new_blue = old_blue + 1
                    if old_blue == 10 and new_blue != 10:
                        delta -= 1
                    elif old_blue != 10 and new_blue == 10:
                        delta += 1
                        
            colors[t_idx] = new_val
            
            # Update counts
            for quad_idx in triple_to_4sets[t_idx]:
                if old_val == 0:
                    red_counts_4[quad_idx] -= 1
                else:
                    red_counts_4[quad_idx] += 1
                    
            for quint_idx in triple_to_5sets[t_idx]:
                if old_val == 1:
                    blue_counts_5[quint_idx] -= 1
                else:
                    blue_counts_5[quint_idx] += 1
            
            current_violations = get_current_violations()
            
            if delta < 0 or (T > min_T and np.random.random() < np.exp(-delta / T)):
                # Accept
                pass
            else:
                # Revert
                colors[t_idx] = old_val
                for quad_idx in triple_to_4sets[t_idx]:
                    if old_val == 0:
                        red_counts_4[quad_idx] += 1
                    else:
                        red_counts_4[quad_idx] -= 1
                        
                for quint_idx in triple_to_5sets[t_idx]:
                    if old_val == 1:
                        blue_counts_5[quint_idx] += 1
                    else:
                        blue_counts_5[quint_idx] -= 1
                
                current_violations = get_current_violations()
            
            if current_violations == 0:
                return colors.tolist()
                
            T *= cooling
            
        return None

    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try increasing n
    for n in range(10, 35):
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
            
        t_limit = min(8.0, remaining - 1.0)
        if t_limit <= 0:
            break
            
        res = search_for_n(n, t_limit)
        if res is not None:
            best_n = n
            best_colors = res
            
    return {'n': best_n, 'colors': best_colors}
