import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    def try_find_coloring(n, max_time):
        """
        Try to find a valid coloring for n vertices using local search with incremental violation tracking.
        """
        if time.time() > deadline - 1:
            return None
            
        # Precompute triples
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        n_triples = len(triples)
        
        # Precompute K4s and K5s
        k4s = []
        for quad in combinations(range(n), 4):
            k4_triples = [triple_to_idx[t] for t in combinations(quad, 3)]
            k4s.append(k4_triples)
            
        k5s = []
        for quint in combinations(range(n), 5):
            k5_triples = [triple_to_idx[t] for t in combinations(quint, 3)]
            k5s.append(k5_triples)
            
        # Build reverse maps
        # triple -> list of k4 indices
        triple_to_k4s = [[] for _ in range(n_triples)]
        for i, k4_triples in enumerate(k4s):
            for t in k4_triples:
                triple_to_k4s[t].append(i)
                
        # triple -> list of k5 indices
        triple_to_k5s = [[] for _ in range(n_triples)]
        for j, k5_triples in enumerate(k5s):
            for t in k5_triples:
                triple_to_k5s[t].append(j)

        # Local search
        best_colors = None
        
        # Try a few restarts
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 1:
                break
                
            # Initialize
            colors = np.random.randint(0, 2, size=n_triples)
            
            # Initialize counts
            k4_red_counts = np.zeros(len(k4s), dtype=np.int8)
            for i, k4_triples in enumerate(k4s):
                k4_red_counts[i] = sum(1 - colors[t] for t in k4_triples)
                
            k5_blue_counts = np.zeros(len(k5s), dtype=np.int8)
            for j, k5_triples in enumerate(k5s):
                k5_blue_counts[j] = sum(colors[t] for t in k5_triples)
                
            # Active violations lists and position dicts for O(1) removal
            active_k4_list = []
            active_k4_pos = {}
            for i in range(len(k4s)):
                if k4_red_counts[i] == 4:
                    active_k4_pos[i] = len(active_k4_list)
                    active_k4_list.append(i)
                    
            active_k5_list = []
            active_k5_pos = {}
            for j in range(len(k5s)):
                if k5_blue_counts[j] == 10:
                    active_k5_pos[j] = len(active_k5_list)
                    active_k5_list.append(j)
            
            # Local search loop
            max_iters = 100000
            for it in range(max_iters):
                if time.time() > deadline - 1:
                    break
                    
                if not active_k4_list and not active_k5_list:
                    return colors.tolist()
                    
                # Pick a violation
                if active_k4_list:
                    idx_in_list = random.randint(0, len(active_k4_list) - 1)
                    k4_idx = active_k4_list[idx_in_list]
                    k4_triples = k4s[k4_idx]
                    t_idx = random.choice(k4_triples)
                else:
                    idx_in_list = random.randint(0, len(active_k5_list) - 1)
                    k5_idx = active_k5_list[idx_in_list]
                    k5_triples = k5s[k5_idx]
                    t_idx = random.choice(k5_triples)
                    
                # Flip the triple
                old_color = colors[t_idx]
                new_color = 1 - old_color
                colors[t_idx] = new_color
                
                # Update k4s containing t_idx
                for k4_i in triple_to_k4s[t_idx]:
                    old_count = k4_red_counts[k4_i]
                    if old_color == 0:
                        new_count = old_count - 1
                    else:
                        new_count = old_count + 1
                    k4_red_counts[k4_i] = new_count
                    
                    # Update active list
                    if old_count == 4 and new_count != 4:
                        # Remove from active list
                        if k4_i in active_k4_pos:
                            pos = active_k4_pos[k4_i]
                            last_elem = active_k4_list[-1]
                            active_k4_list[pos] = last_elem
                            active_k4_pos[last_elem] = pos
                            active_k4_list.pop()
                            del active_k4_pos[k4_i]
                    elif old_count != 4 and new_count == 4:
                        # Add to active list
                        active_k4_pos[k4_i] = len(active_k4_list)
                        active_k4_list.append(k4_i)
                            
                # Update k5s containing t_idx
                for k5_j in triple_to_k5s[t_idx]:
                    old_count = k5_blue_counts[k5_j]
                    if old_color == 0:
                        new_count = old_count + 1
                    else:
                        new_count = old_count - 1
                    k5_blue_counts[k5_j] = new_count
                    
                    # Update active list
                    if old_count == 10 and new_count != 10:
                        # Remove from active list
                        if k5_j in active_k5_pos:
                            pos = active_k5_pos[k5_j]
                            last_elem = active_k5_list[-1]
                            active_k5_list[pos] = last_elem
                            active_k5_pos[last_elem] = pos
                            active_k5_list.pop()
                            del active_k5_pos[k5_j]
                    elif old_count != 10 and new_count == 10:
                        # Add to active list
                        active_k5_pos[k5_j] = len(active_k5_list)
                        active_k5_list.append(k5_j)
                            
        return None

    # Try to find coloring for n=34, then 33, etc.
    time_per_n = 20
    for n in range(34, 4, -1):
        if time.time() > deadline - 2:
            break
        remaining = deadline - time.time() - 1
        t_budget = min(time_per_n, remaining)
        if t_budget < 1:
            break
        result = try_find_coloring(n, t_budget)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback to a small valid coloring
    # n=5: 10 triples. One red, 9 blue.
    # Red K4? No (max 1 red). Blue K5? No (9 blue).
    return {'n': 5, 'colors': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
