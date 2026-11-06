# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 117  # Safety margin for 120s limit
    
    def build_index_and_maps(n):
        """Build mapping from triple to index and precompute adjacency maps."""
        idx = {}
        i = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx[(a,b,c)] = i
                    i += 1
        
        n_triples = len(idx)
        
        # Precompute which triples are in each quad and quint
        quad_triples = []
        for quad in combinations(range(n), 4):
            quad_triples.append([idx[tri] for tri in combinations(quad, 3)])
        
        quint_triples = []
        for quint in combinations(range(n), 5):
            quint_triples.append([idx[tri] for tri in combinations(quint, 3)])
        
        n_quads = len(quad_triples)
        n_quints = len(quint_triples)
        
        # Convert to numpy arrays
        quad_arr = np.array(quad_triples, dtype=np.int32) if n_quads > 0 else np.empty((0,4), dtype=np.int32)
        quint_arr = np.array(quint_triples, dtype=np.int32) if n_quints > 0 else np.empty((0,10), dtype=np.int32)
        
        # Build reverse maps: triple -> list of quad indices, triple -> list of quint indices
        triple_to_quads = [[] for _ in range(n_triples)]
        for q_idx, quads in enumerate(quad_triples):
            for t in quads:
                triple_to_quads[t].append(q_idx)
                
        triple_to_quints = [[] for _ in range(n_triples)]
        for q_idx, quints in enumerate(quint_triples):
            for t in quints:
                triple_to_quints[t].append(q_idx)
                
        return idx, quad_arr, quint_arr, triple_to_quads, triple_to_quints

    def simulated_annealing(n, p_red, time_limit):
        """Run simulated annealing with incremental energy updates."""
        idx, quad_arr, quint_arr, triple_to_quads, triple_to_quints = build_index_and_maps(n)
        n_triples = len(idx)
        n_quads = len(quad_arr)
        n_quints = len(quint_arr)
        
        if n_triples == 0:
            return []
            
        # Initial random coloring
        colors = np.random.choice([0, 1], size=n_triples, p=[p_red, 1-p_red])
        
        # Initialize violation flags
        quad_is_red = np.zeros(n_quads, dtype=bool)
        quint_is_blue = np.zeros(n_quints, dtype=bool)
        
        # Calculate initial energy and flags
        if n_quads > 0:
            quad_colors = colors[quad_arr]
            quad_is_red = np.all(quad_colors == 0, axis=1)
        if n_quints > 0:
            quint_colors = colors[quint_arr]
            quint_is_blue = np.all(quint_colors == 1, axis=1)
            
        current_energy = np.sum(quad_is_red) + np.sum(quint_is_blue)
        
        if current_energy == 0:
            return colors.tolist()

        T = 10.0  # Initial temperature
        T_min = 0.01
        cooling_rate = 0.9999  # Slow cooling
        max_time = time_limit
        
        start_sa = time.time()
        
        while T > T_min:
            if time.time() - start_sa > max_time:
                break
            
            # Pick a random triple to flip
            tri_idx = np.random.randint(n_triples)
            
            old_color = colors[tri_idx]
            new_color = 1 - old_color
            
            # Calculate delta energy
            delta_energy = 0
            
            # Check affected quads
            for q_idx in triple_to_quads[tri_idx]:
                was_red = quad_is_red[q_idx]
                
                # Determine if quad is red after flip
                # If new_color is 1, it's not red.
                # If new_color is 0, check if other 3 are 0.
                if new_color == 1:
                    is_red_now = False
                else:
                    # new_color == 0
                    # Check others
                    # quad_arr[q_idx] has 4 elements. One is tri_idx.
                    # We can just check all 4, treating tri_idx as new_color
                    # But colors[tri_idx] is still old_color.
                    # So we check:
                    # for t in quad_arr[q_idx]:
                    #   val = new_color if t == tri_idx else colors[t]
                    #   if val != 0: is_red_now = False; break
                    #   else: is_red_now = True
                    
                    is_red_now = True
                    for t in quad_arr[q_idx]:
                        val = new_color if t == tri_idx else colors[t]
                        if val != 0:
                            is_red_now = False
                            break
                    
                if is_red_now and not was_red:
                    delta_energy += 1
                elif not is_red_now and was_red:
                    delta_energy -= 1
            
            # Check affected quints
            for q_idx in triple_to_quints[tri_idx]:
                was_blue = quint_is_blue[q_idx]
                
                # Determine if quint is blue after flip
                if new_color == 0:
                    is_blue_now = False
                else:
                    # new_color == 1
                    is_blue_now = True
                    for t in quint_arr[q_idx]:
                        val = new_color if t == tri_idx else colors[t]
                        if val != 1:
                            is_blue_now = False
                            break

                if is_blue_now and not was_blue:
                    delta_energy += 1
                elif not is_blue_now and was_blue:
                    delta_energy -= 1
            
            # Accept or Reject
            if delta_energy < 0 or (delta_energy == 0 and random.random() < 0.5) or random.random() < np.exp(-delta_energy / T):
                # Accept
                colors[tri_idx] = new_color
                current_energy += delta_energy
                
                # Update flags
                # We need to update quad_is_red and quint_is_blue for the affected items
                # We can re-evaluate them or just apply the logic again.
                # Since we already computed is_red_now/is_blue_now, we could store them, but it's easier to just re-check or update based on delta.
                # Actually, we can just update the flags for the specific indices we checked.
                
                for q_idx in triple_to_quads[tri_idx]:
                    # Re-calculate is_red_now? No, we can just use the logic.
                    # But we didn't store it. Let's just re-calculate for safety or store in a dict.
                    # For speed, let's just re-calculate using the updated colors.
                    # It's cheap.
                    quad_is_red[q_idx] = np.all(colors[quad_arr[q_idx]] == 0)
                    
                for q_idx in triple_to_quints[tri_idx]:
                    quint_is_blue[q_idx] = np.all(colors[quint_arr[q_idx]] == 1)
                
                if current_energy == 0:
                    return colors.tolist()
            
            T *= cooling_rate
            
        return None

    # Try decreasing n from 34 down to 30 to maximize the score
    time_budgets = {
        34: 40,
        33: 30,
        32: 20,
        31: 10,
        30: 5
    }
    
    best_n = 1
    best_colors = [0]
    
    for n in [34, 33, 32, 31, 30]:
        if time.time() > deadline - 2:
            break
        
        remaining = deadline - time.time()
        budget = min(time_budgets[n], remaining - 2)
        
        if budget < 1:
            continue
            
        biases = [0.5, 0.4, 0.6]
        
        found = False
        for p in biases:
            if time.time() > deadline - 2:
                break
            sub_budget = budget / len(biases)
            
            result = simulated_annealing(n, p, sub_budget)
            if result is not None:
                best_n = n
                best_colors = result
                found = True
                break
        
        if found:
            break

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
