import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_n = 9
    best_colors = None
    
    def get_validity_info(n):
        """Precompute indices and counts structures for a given n."""
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        num_triples = len(triples)
        
        k4s = list(combinations(range(n), 4))
        k4_triple_indices = []
        for k4 in k4s:
            inds = [triple_to_idx[(a, b, c)] for a, b, c in combinations(k4, 3)]
            k4_triple_indices.append(inds)
            
        k5s = list(combinations(range(n), 5))
        k5_triple_indices = []
        for k5 in k5s:
            inds = [triple_to_idx[(a, b, c)] for a, b, c in combinations(k5, 3)]
            k5_triple_indices.append(inds)
            
        # Map triple index to list of k4 indices and k5 indices it belongs to
        triple_to_k4s = [[] for _ in range(num_triples)]
        for i, inds in enumerate(k4_triple_indices):
            for t_idx in inds:
                triple_to_k4s[t_idx].append(i)
                
        triple_to_k5s = [[] for _ in range(num_triples)]
        for i, inds in enumerate(k5_triple_indices):
            for t_idx in inds:
                triple_to_k5s[t_idx].append(i)
                
        return num_triples, k4_triple_indices, k5_triple_indices, triple_to_k4s, triple_to_k5s

    def find_valid_coloring(n, time_budget):
        """Run SA to find a valid coloring for n."""
        num_triples, k4_inds, k5_inds, t_to_k4s, t_to_k5s = get_validity_info(n)
        
        # We will try multiple restarts within the time budget
        end_time = time.time() + time_budget
        max_restarts = 100
        
        for restart in range(max_restarts):
            if time.time() > end_time:
                break
                
            # Initialize random coloring
            colors = np.random.randint(0, 2, num_triples)
            
            # Calculate initial counts
            red_count = np.zeros(len(k4_inds), dtype=int)
            blue_count = np.zeros(len(k5_inds), dtype=int)
            
            for j, inds in enumerate(k4_inds):
                red_count[j] = np.sum(colors[inds] == 0)
                
            for j, inds in enumerate(k5_inds):
                blue_count[j] = np.sum(colors[inds] == 1)
                
            E = np.sum(red_count == 4) + np.sum(blue_count == 10)
            
            if E == 0:
                return colors.tolist()
            
            T = 1.0
            cooling = 0.9995
            min_T = 0.01
            
            # Run SA for a fixed number of steps or until time slice ends
            # Let's do a time-sliced SA
            step_end_time = time.time() + (end_time - time.time()) / (max_restarts - restart)
            if step_end_time < time.time():
                 step_end_time = time.time() + 1.0 # At least 1 sec per restart? No, just use remaining.
            
            # Actually, just run until time is up for this restart or E=0
            # But if we spend all time on one restart, we might not explore other basins.
            # Let's limit each restart to a small fraction of time.
            # Say, 1 second per restart?
            # If time_budget is 100s, 100 restarts * 1s = 100s.
            
            while time.time() < step_end_time and T > min_T:
                t_idx = np.random.randint(num_triples)
                current_color = colors[t_idx]
                
                delta_E = 0
                
                for k4_idx in t_to_k4s[t_idx]:
                    if current_color == 0:
                        if red_count[k4_idx] == 4:
                            delta_E -= 1
                    else:
                        if red_count[k4_idx] == 3:
                            delta_E += 1
                            
                for k5_idx in t_to_k5s[t_idx]:
                    if current_color == 0:
                        if blue_count[k5_idx] == 9:
                            delta_E += 1
                    else:
                        if blue_count[k5_idx] == 10:
                            delta_E -= 1
                            
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    colors[t_idx] = 1 - current_color
                    E += delta_E
                    
                    for k4_idx in t_to_k4s[t_idx]:
                        if current_color == 0:
                            red_count[k4_idx] -= 1
                        else:
                            red_count[k4_idx] += 1
                            
                    for k5_idx in t_to_k5s[t_idx]:
                        if current_color == 0:
                            blue_count[k5_idx] += 1
                        else:
                            blue_count[k5_idx] -= 1
                            
                    if E == 0:
                        return colors.tolist()
                        
                T *= cooling
                
        return None

    # Try n values in descending order
    # Start with 15, then 14, then 13
    # If we find 15, we are done.
    
    target_ns = [15, 14, 13]
    
    for n in target_ns:
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        # Give this n a good chunk of time
        # If we find it, we stop.
        res = find_valid_coloring(n, remaining - 2)
        
        if res is not None:
            best_n = n
            best_colors = res
            break
            
    if best_n == 9:
        from itertools import combinations
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
        
    return {'n': best_n, 'colors': best_colors}
