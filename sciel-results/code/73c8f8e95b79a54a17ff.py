import random
import time
import itertools
import numpy as np
import math

def construct():
    # Target n=34 is the known bound. We aim to find a valid coloring for n=34.
    # If we fail, we can fall back to smaller n, but 34 is the target for max score.
    
    def run_sa(n, time_limit=115):
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        quads = list(itertools.combinations(range(n), 4))
        num_quads = len(quads)
        
        quints = list(itertools.combinations(range(n), 5))
        num_quints = len(quints)
        
        # Precompute incidence: for each triple, which quads and quints does it belong to?
        # And for each quad/quint, which triples does it contain?
        quad_to_triples = []
        for q in quads:
            ts = [triple_idx[t] for t in itertools.combinations(q, 3)]
            quad_to_triples.append(ts)
            
        quint_to_triples = []
        for q in quints:
            ts = [triple_idx[t] for t in itertools.combinations(q, 3)]
            quint_to_triples.append(ts)
            
        triple_to_quads = [[] for _ in range(num_triples)]
        for i, ts in enumerate(quad_to_triples):
            for t in ts:
                triple_to_quads[t].append(i)
                
        triple_to_quints = [[] for _ in range(num_triples)]
        for i, ts in enumerate(quint_to_triples):
            for t in ts:
                triple_to_quints[t].append(i)
                
        # Initialize state
        # Random initialization
        colors = np.random.randint(0, 2, size=num_triples)
        
        # Initial energy calculation
        # red_count_4[q] = number of red triples in quad q
        # blue_count_5[q] = number of blue triples in quint q
        red_count_4 = np.zeros(num_quads, dtype=np.int32)
        blue_count_5 = np.zeros(num_quints, dtype=np.int32)
        
        for t_idx, c in enumerate(colors):
            if c == 0:
                for q in triple_to_quads[t_idx]:
                    red_count_4[q] += 1
            else:
                for q in triple_to_quints[t_idx]:
                    blue_count_5[q] += 1
                    
        def calc_energy():
            # Energy is number of quads with 4 reds + number of quints with 10 blues
            return np.sum(red_count_4 == 4) + np.sum(blue_count_5 == 10)
            
        current_energy = calc_energy()
        best_energy = current_energy
        best_colors = colors.copy()
        
        # Simulated Annealing parameters
        # Initial temperature high enough to accept bad moves
        T = 1.0
        cooling_rate = 0.999 # Tune this
        min_T = 1e-5
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Pick a random triple to flip
            t_idx = np.random.randint(0, num_triples)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            # Calculate delta energy
            # This is complex to do incrementally without full recalc if we don't store counts carefully.
            # We have counts. Let's update counts tentatively, calc new energy, and revert if rejected.
            
            # Tentative update
            if old_color == 0: # Red -> Blue
                # Decrease red counts for quads containing t_idx
                for q in triple_to_quads[t_idx]:
                    red_count_4[q] -= 1
                # Increase blue counts for quints containing t_idx
                for q in triple_to_quints[t_idx]:
                    blue_count_5[q] += 1
            else: # Blue -> Red
                # Increase red counts for quads containing t_idx
                for q in triple_to_quads[t_idx]:
                    red_count_4[q] += 1
                # Decrease blue counts for quints containing t_idx
                for q in triple_to_quints[t_idx]:
                    blue_count_5[q] -= 1
            
            colors[t_idx] = new_color
            new_energy = calc_energy()
            delta_e = new_energy - current_energy
            
            if delta_e <= 0 or random.random() < math.exp(-delta_e / T):
                # Accept move
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_colors = colors.copy()
                    if best_energy == 0:
                        return {'n': n, 'colors': best_colors.tolist()}
            else:
                # Reject move, revert
                if old_color == 0: # Was Red, tried Blue, revert to Red
                    for q in triple_to_quads[t_idx]:
                        red_count_4[q] += 1
                    for q in triple_to_quints[t_idx]:
                        blue_count_5[q] -= 1
                else: # Was Blue, tried Red, revert to Blue
                    for q in triple_to_quads[t_idx]:
                        red_count_4[q] -= 1
                    for q in triple_to_quints[t_idx]:
                        blue_count_5[q] += 1
                colors[t_idx] = old_color
                # current_energy remains unchanged
                
            T *= cooling_rate
            if T < min_T:
                break
                
        if best_energy == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        return None

    # Try n=34
    res = run_sa(34, time_limit=110)
    if res:
        return res
        
    # Fallback if 34 fails (unlikely with SA if tuned well, but good for safety)
    # Try n=33
    res = run_sa(33, time_limit=5)
    if res:
        return res
        
    # Fallback to a simple valid small n
    # n=9 is small enough to construct easily or just return a known small valid one.
    # Actually, any n<=7 is trivially valid? No, R(4,5;3) > 7 is known.
    # Let's just return a small n to be safe, though score will be low.
    n = 9
    triples = list(itertools.combinations(range(n), 3))
    # Simple random coloring for small n might work, or we can just return a fixed one.
    # For n=9, there are 84 triples.
    # A random coloring has a chance, but let's just return a dummy valid one if we can't find 34.
    # Actually, we can just return n=1 to be safe? No, score is low.
    # Let's try a quick local search for n=30 if 34 fails.
    
    # For the sake of the "new program" directive, the SA for 34 is the core.
    # If it fails, we return a smaller n.
    
    # Fallback: return n=1 (always valid)
    return {'n': 1, 'colors': []}
