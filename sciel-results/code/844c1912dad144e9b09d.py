import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin

    def get_seed():
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}
    
    best_result = get_seed()
    
    def solve_n(n, time_budget):
        """
        Attempt to find a valid 3-uniform hypergraph coloring for n vertices
        without red K4^(3) or blue K5^(3).
        """
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4 and K5 structures
        k4_indices = []
        for quad in itertools.combinations(range(n), 4):
            t_indices = [triple_idx[(quad[i], quad[j], quad[k])] for i in range(4) for j in range(i+1, 4) for k in range(j+1, 4)]
            k4_indices.append(t_indices)
            
        k5_indices = []
        for quint in itertools.combinations(range(n), 5):
            t_indices = [triple_idx[(quint[c0], quint[c1], quint[c2])] for c0, c1, c2 in itertools.combinations(range(5), 3)]
            k5_indices.append(t_indices)
            
        num_k4 = len(k4_indices)
        num_k5 = len(k5_indices)
        
        # Build incidence: which K4s/K5s contain each triple
        triple_to_k4 = [[] for _ in range(num_triples)]
        for i, inds in enumerate(k4_indices):
            for t in inds:
                triple_to_k4[t].append(i)
                
        triple_to_k5 = [[] for _ in range(num_triples)]
        for i, inds in enumerate(k5_indices):
            for t in inds:
                triple_to_k5[t].append(i)
                
        # Helper to compute initial violations
        def compute_initial_violations(colors):
            is_red_k4 = np.zeros(num_k4, dtype=bool)
            is_blue_k5 = np.zeros(num_k5, dtype=bool)
            
            # For each K4, check if all 4 triples are red (0)
            for i, inds in enumerate(k4_indices):
                if all(colors[t] == 0 for t in inds):
                    is_red_k4[i] = True
                    
            # For each K5, check if all 10 triples are blue (1)
            for i, inds in enumerate(k5_indices):
                if all(colors[t] == 1 for t in inds):
                    is_blue_k5[i] = True
                    
            return is_red_k4, is_blue_k5, is_red_k4.sum() + is_blue_k5.sum()

        # Main search loop
        while time.time() < deadline:
            # Random initialization
            colors = np.random.randint(0, 2, size=num_triples)
            is_red_k4, is_blue_k5, total_viol = compute_initial_violations(colors)
            
            if total_viol == 0:
                return colors.tolist()
                
            # Simulated Annealing / Local Search
            current_temp = 1.0
            cooling_rate = 0.999
            min_temp = 1e-6
            no_improve_count = 0
            max_no_improve = 2000
            
            while time.time() < deadline and no_improve_count < max_no_improve:
                if total_viol == 0:
                    return colors.tolist()
                    
                # Pick a random triple to flip
                t_idx = random.randint(0, num_triples - 1)
                old_color = colors[t_idx]
                new_color = 1 - old_color
                
                delta = 0
                affected_k4 = []
                affected_k5 = []
                
                # Check K4s containing this triple
                for k4_i in triple_to_k4[t_idx]:
                    affected_k4.append(k4_i)
                    if is_red_k4[k4_i]:
                        # Was red, flipping one triple to blue breaks it (unless we flip to red, but we flip)
                        # If old was 0 and we flip to 1, it's no longer red.
                        # Wait, if is_red_k4 is true, all 4 are 0.
                        # If we flip t_idx from 0 to 1, it becomes not red. Delta -= 1.
                        # If we flip t_idx from 1 to 0, it was already not red (since all 4 must be 0 for is_red_k4 to be true).
                        # So if is_red_k4[k4_i] is True, old_color MUST be 0.
                        delta -= 1
                    else:
                        # Not red. Can it become red?
                        # Only if new_color is 0 (i.e., old was 1) and all other 3 are 0.
                        if new_color == 0:
                            all_others_red = True
                            for t in k4_indices[k4_i]:
                                if t != t_idx and colors[t] != 0:
                                    all_others_red = False
                                    break
                            if all_others_red:
                                delta += 1
                                
                # Check K5s containing this triple
                for k5_i in triple_to_k5[t_idx]:
                    affected_k5.append(k5_i)
                    if is_blue_k5[k5_i]:
                        # Was blue, flipping one triple to red breaks it.
                        # If is_blue_k5 is true, all 10 are 1.
                        # So old_color MUST be 1.
                        delta -= 1
                    else:
                        # Not blue. Can it become blue?
                        # Only if new_color is 1 (i.e., old was 0) and all other 9 are 1.
                        if new_color == 1:
                            all_others_blue = True
                            for t in k5_indices[k5_i]:
                                if t != t_idx and colors[t] != 1:
                                    all_others_blue = False
                                    break
                            if all_others_blue:
                                delta += 1
                                
                # Acceptance criterion
                if delta <= 0 or random.random() < np.exp(-delta / current_temp):
                    # Flip
                    colors[t_idx] = new_color
                    total_viol += delta
                    
                    # Update flags
                    for k4_i in affected_k4:
                        if is_red_k4[k4_i]:
                            is_red_k4[k4_i] = False
                        else:
                            if new_color == 0:
                                all_others_red = True
                                for t in k4_indices[k4_i]:
                                    if t != t_idx and colors[t] != 0:
                                        all_others_red = False
                                        break
                                if all_others_red:
                                    is_red_k4[k4_i] = True
                                      
                    for k5_i in affected_k5:
                        if is_blue_k5[k5_i]:
                            is_blue_k5[k5_i] = False
                        else:
                            if new_color == 1:
                                all_others_blue = True
                                for t in k5_indices[k5_i]:
                                    if t != t_idx and colors[t] != 1:
                                        all_others_blue = False
                                        break
                                if all_others_blue:
                                    is_blue_k5[k5_i] = True
                                    
                    if delta < 0:
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        
                else:
                    no_improve_count += 1
                    
                current_temp *= cooling_rate
                if current_temp < min_temp:
                    current_temp = min_temp
                    
        return None

    # Try n=34 first
    res = solve_n(34, deadline - time.time())
    if res is not None:
        return {'n': 34, 'colors': res}
        
    # Fallback to seed
    return best_result
