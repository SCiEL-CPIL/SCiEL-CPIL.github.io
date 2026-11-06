# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    def build_precomputed(n):
        """Build all precomputed structures for fast violation checking."""
        idx = {}
        i = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx[(a,b,c)] = i
                    i += 1
        n_triples = i
        
        # Precompute quad triple indices (for red K4)
        quad_triples = []
        for quad in combinations(range(n), 4):
            quad_triples.append([idx[tri] for tri in combinations(quad, 3)])
        quad_arr = np.array(quad_triples, dtype=np.int32) if quad_triples else np.empty((0,4), dtype=np.int32)
        
        # Precompute quint triple indices (for blue K5)
        quint_triples = []
        for quint in combinations(range(n), 5):
            quint_triples.append([idx[tri] for tri in combinations(quint, 3)])
        quint_arr = np.array(quint_triples, dtype=np.int32) if quint_triples else np.empty((0,10), dtype=np.int32)
        
        # For each triple, precompute which quads/quints contain it
        # This allows fast impact scoring when flipping a triple
        triple_to_quads = [[] for _ in range(n_triples)]
        for qi, q in enumerate(quad_arr):
            for t in q:
                triple_to_quads[t].append(qi)
        
        triple_to_quints = [[] for _ in range(n_triples)]
        for qi, q in enumerate(quint_arr):
            for t in q:
                triple_to_quints[t].append(qi)
        
        return n_triples, quad_arr, quint_arr, triple_to_quads, triple_to_quints
    
    def count_violations(colors, quad_arr, quint_arr):
        """Count total violations (red K4 + blue K5)."""
        total = 0
        if len(quad_arr) > 0:
            total += int(np.sum(np.all(colors[quad_arr] == 0, axis=1)))
        if len(quint_arr) > 0:
            total += int(np.sum(np.all(colors[quint_arr] == 1, axis=1)))
        return total
    
    def get_violating_triples(colors, quad_arr, quint_arr):
        """Get set of triples involved in any violation."""
        violating = set()
        if len(quad_arr) > 0:
            red_viol = np.all(colors[quad_arr] == 0, axis=1)
            if np.any(red_viol):
                violating.update(quad_arr[red_viol].flatten().tolist())
        if len(quint_arr) > 0:
            blue_viol = np.all(colors[quint_arr] == 1, axis=1)
            if np.any(blue_viol):
                violating.update(quint_arr[blue_viol].flatten().tolist())
        return violating
    
    def simulated_annealing(n, p_red, max_time, rng_seed=42):
        """Run simulated annealing to find valid coloring."""
        rng = np.random.RandomState(rng_seed)
        n_triples, quad_arr, quint_arr, triple_to_quads, triple_to_quints = build_precomputed(n)
        
        # Initial coloring
        colors = rng.choice([0, 1], size=n_triples, p=[p_red, 1-p_red])
        
        current_energy = count_violations(colors, quad_arr, quint_arr)
        best_energy = current_energy
        best_colors = colors.copy()
        
        # SA parameters
        T_start = 5.0
        T_end = 0.01
        # Number of steps: aim for ~0.5s per restart
        n_steps = max(500, int(max_time * 800))
        cooling = (T_end / T_start) ** (1.0 / max(1, n_steps))
        
        T = T_start
        step = 0
        while step < n_steps:
            if time.time() > deadline - 3:
                break
            
            # Pick a random triple to flip
            tri = rng.randint(0, n_triples)
            old_color = colors[tri]
            new_color = 1 - old_color
            colors[tri] = new_color
            
            new_energy = count_violations(colors, quad_arr, quint_arr)
            delta = new_energy - current_energy
            
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors.copy()
                    if best_energy == 0:
                        break
            else:
                colors[tri] = old_color
            
            T *= cooling
            step += 1
        
        if best_energy == 0:
            return best_colors.tolist()
        return None
    
    best_n = 1
    best_colors = [0]
    
    # Try n values, prioritizing larger n
    # Focus on n=34 (published construction) and nearby values
    target_ns = [34, 33, 32, 35, 31, 30, 28, 26, 24, 20, 15, 10]
    
    for n in target_ns:
        if time.time() > deadline - 10:
            break
        if n > 40:
            continue
            
        remaining = deadline - 10 - time.time()
        if remaining < 2:
            break
        
        # Allocate time: more time for larger n
        if n >= 30:
            time_per_restart = min(1.5, remaining / 3)
        elif n >= 20:
            time_per_restart = min(1.0, remaining / 5)
        else:
            time_per_restart = min(0.5, remaining / 8)
        
        max_restarts = max(2, int(remaining / time_per_restart))
        
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Vary the bias
            p_red = 0.3 + 0.4 * (restart % 5) / 4.0  # 0.3, 0.4, 0.5, 0.6, 0.7
            seed = restart * 7 + n
            
            result = simulated_annealing(n, p_red, time_per_restart, rng_seed=seed)
            if result is not None:
                best_n = n
                best_colors = result
                found = True
                break
        
        if found:
            # Try to go higher
            continue
        else:
            # If we couldn't find for this n, try smaller ones
            continue
    
    # Final validation
    if best_n > 1:
        # Quick sanity check: verify no violations
        n_triples, quad_arr, quint_arr, _, _ = build_precomputed(best_n)
        colors_arr = np.array(best_colors)
        if len(quad_arr) > 0:
            red_viol = np.all(colors_arr[quad_arr] == 0, axis=1)
            if np.any(red_viol):
                # Invalid, fall back to n=1
                best_n = 1
                best_colors = [0]
        if len(quint_arr) > 0:
            blue_viol = np.all(colors_arr[quint_arr] == 1, axis=1)
            if np.any(blue_viol):
                best_n = 1
                best_colors = [0]
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
