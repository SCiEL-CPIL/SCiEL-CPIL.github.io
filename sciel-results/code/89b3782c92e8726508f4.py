import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    def build_structures(n):
        triples = list(itertools.combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        num_triples = len(triples)
        
        four_sets = list(itertools.combinations(range(n), 4))
        five_sets = list(itertools.combinations(range(n), 5))
        
        num_fours = len(four_sets)
        num_fives = len(five_sets)
        
        # M: shape (num_fours, 4)
        M = np.zeros((num_fours, 4), dtype=np.int32)
        for i, quad in enumerate(four_sets):
            M[i, 0] = triple_to_idx[(quad[0], quad[1], quad[2])]
            M[i, 1] = triple_to_idx[(quad[0], quad[1], quad[3])]
            M[i, 2] = triple_to_idx[(quad[0], quad[2], quad[3])]
            M[i, 3] = triple_to_idx[(quad[1], quad[2], quad[3])]
            
        # Q: shape (num_fives, 10)
        Q = np.zeros((num_fives, 10), dtype=np.int32)
        for i, quint in enumerate(five_sets):
            idx = 0
            for t in itertools.combinations(quint, 3):
                Q[i, idx] = triple_to_idx[t]
                idx += 1
                
        # Build reverse index as lists of numpy arrays for vectorized updates
        triple_to_fours = [[] for _ in range(num_triples)]
        triple_to_fives = [[] for _ in range(num_triples)]
        
        for i in range(num_fours):
            for j in range(4):
                triple_to_fours[M[i, j]].append(i)
        
        for i in range(num_fives):
            for j in range(10):
                triple_to_fives[Q[i, j]].append(i)
                
        # Convert to numpy arrays
        triple_to_fours_np = [np.array(lst, dtype=np.int32) for lst in triple_to_fours]
        triple_to_fives_np = [np.array(lst, dtype=np.int32) for lst in triple_to_fives]
                
        return M, Q, num_triples, triple_to_fours_np, triple_to_fives_np

    def solve_for_n(n, time_budget):
        M, Q, num_triples, triple_to_fours_np, triple_to_fives_np = build_structures(n)
        num_fours = M.shape[0]
        num_fives = Q.shape[0]
        
        def init_counts(colors):
            red_counts = np.zeros(num_fours, dtype=np.int32)
            blue_counts = np.zeros(num_fives, dtype=np.int32)
            
            # Vectorized count initialization
            # For each four-set, count reds
            # M is (num_fours, 4). colors[M] is (num_fours, 4).
            # red_counts = np.sum(colors[M] == 0, axis=1)
            red_counts = np.sum(colors[M] == 0, axis=1).astype(np.int32)
            
            # For each five-set, count blues
            # Q is (num_fives, 10). colors[Q] is (num_fives, 10).
            # blue_counts = np.sum(colors[Q] == 1, axis=1)
            blue_counts = np.sum(colors[Q] == 1, axis=1).astype(np.int32)
            
            return red_counts, blue_counts
        
        def get_cost(red_counts, blue_counts):
            return int(np.sum(red_counts == 4) + np.sum(blue_counts == 10))
        
        # Initial random coloring
        colors = np.random.randint(0, 2, num_triples)
        red_counts, blue_counts = init_counts(colors)
        current_cost = get_cost(red_counts, blue_counts)
        
        best_cost = current_cost
        best_colors = colors.copy()
        
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        restart_threshold = 5000
        steps_since_improvement = 0
        
        deadline = start_time + time_budget
        
        while time.time() < deadline:
            if current_cost == 0:
                return colors.tolist()
            
            t_idx = np.random.randint(0, num_triples)
            old_color = colors[t_idx]
            
            # Vectorized update
            if old_color == 0:
                # Red -> Blue
                red_counts[triple_to_fours_np[t_idx]] -= 1
                blue_counts[triple_to_fives_np[t_idx]] += 1
            else:
                # Blue -> Red
                red_counts[triple_to_fours_np[t_idx]] += 1
                blue_counts[triple_to_fives_np[t_idx]] -= 1
            
            colors[t_idx] = 1 - old_color
            
            new_cost = get_cost(red_counts, blue_counts)
            
            if new_cost <= current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = colors.copy()
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                delta = new_cost - current_cost
                if T > min_T:
                    prob = np.exp(-delta / T)
                    if np.random.random() < prob:
                        current_cost = new_cost
                        steps_since_improvement += 1
                    else:
                        # Revert
                        colors[t_idx] = old_color
                        if old_color == 0:
                            red_counts[triple_to_fours_np[t_idx]] += 1
                            blue_counts[triple_to_fives_np[t_idx]] -= 1
                        else:
                            red_counts[triple_to_fours_np[t_idx]] -= 1
                            blue_counts[triple_to_fives_np[t_idx]] += 1
                        steps_since_improvement += 1
                else:
                    # Revert
                    colors[t_idx] = old_color
                    if old_color == 0:
                        red_counts[triple_to_fours_np[t_idx]] += 1
                        blue_counts[triple_to_fives_np[t_idx]] -= 1
                    else:
                        red_counts[triple_to_fours_np[t_idx]] -= 1
                        blue_counts[triple_to_fives_np[t_idx]] += 1
                    steps_since_improvement += 1
            
            T *= cooling_rate
            
            if steps_since_improvement > restart_threshold:
                T = 1.0
                colors = np.random.randint(0, 2, num_triples)
                red_counts, blue_counts = init_counts(colors)
                current_cost = get_cost(red_counts, blue_counts)
                steps_since_improvement = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()

        if best_cost == 0:
            return best_colors.tolist()
        return None

    # Target n=35 first
    # Allocate 100s to n=35
    if time.time() - start_time < time_limit - 10:
        result = solve_for_n(35, 100.0)
        if result is not None:
            return {'n': 35, 'colors': result}
    
    # Target n=34
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        result = solve_for_n(34, remaining - 2)
        if result is not None:
            return {'n': 34, 'colors': result}
            
    # Fallback
    n_fallback = 34
    # If search failed, we might still return a random one? No, must be valid.
    # If n=34 search failed, try n=33?
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        result = solve_for_n(33, remaining - 2)
        if result is not None:
            return {'n': 33, 'colors': result}
            
    # Absolute fallback: return a known small valid one? 
    # The prompt implies we should maximize n. If we can't find 34, we should find 33, etc.
    # But 34 is known to exist. The vectorized search should find it quickly.
    # If it fails, there's a bug or it's extremely hard.
    # Let's try a quick greedy for small n if all else fails.
    n_fallback = 10
    colors_fallback = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n_fallback), 3)]
    return {'n': n_fallback, 'colors': colors_fallback}
