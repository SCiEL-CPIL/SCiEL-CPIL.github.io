import time
import numpy as np
from itertools import combinations
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 112  # safety margin before 150s kill

    def search_for_n(n, time_limit, seed=0):
        M = comb(n, 3)
        triples = list(combinations(range(n), 3))
        tri_to_idx = {tri: i for i, tri in enumerate(triples)}

        # Precompute incidence
        quad_triples = []
        for quad in combinations(range(n), 4):
            idxs = [tri_to_idx[tuple(x for j, x in enumerate(quad) if j != i)] for i in range(4)]
            quad_triples.append(idxs)
        quint_triples = []
        for quint in combinations(range(n), 5):
            idxs = [tri_to_idx[t] for t in combinations(quint, 3)]
            quint_triples.append(idxs)

        n_quads = len(quad_triples)
        n_quints = len(quint_triples)

        # Inverse incidence: triple -> list of quads/quints it belongs to
        triple_to_quads = [[] for _ in range(M)]
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                triple_to_quads[t].append(qidx)
        
        triple_to_quints = [[] for _ in range(M)]
        for qidx, tidxs in enumerate(quint_triples):
            for t in tidxs:
                triple_to_quints[t].append(qidx)

        # Convert to numpy arrays for faster iteration if needed, or keep lists
        # Lists are fine for small n, but n=34 -> M=5984, n_quads=46376, n_quints=237336
        # Iterating over lists of indices is fast in Python if we don't do heavy work inside.
        
        rng = np.random.RandomState(seed)

        # Initialize colors
        # Use a heuristic initialization: random or modular
        # Modular: (a+b+c) % 3 == 0 -> Red (0), else Blue (1)
        colors = np.zeros(M, dtype=np.int8)
        for i, (a, b, c) in enumerate(triples):
            colors[i] = 0 if (a + b + c) % 3 == 0 else 1

        # Initialize counts
        red_counts = np.zeros(n_quads, dtype=np.int32)
        blue_counts = np.zeros(n_quints, dtype=np.int32)
        
        # Populate counts
        for t in range(M):
            if colors[t] == 0: # Red
                for q in triple_to_quads[t]:
                    red_counts[q] += 1
            else: # Blue
                for k in triple_to_quints[t]:
                    blue_counts[k] += 1
        
        # Count initial violations
        total_violations = np.sum(red_counts == 4) + np.sum(blue_counts == 10)

        def flip(t):
            nonlocal total_violations
            old_color = colors[t]
            
            if old_color == 0: # Red -> Blue
                # Update Red counts (decrease)
                for q in triple_to_quads[t]:
                    old_c = red_counts[q]
                    if old_c == 4:
                        total_violations -= 1
                    red_counts[q] = old_c - 1
                    # If it becomes 4? No, it decreases.
                
                # Update Blue counts (increase)
                for k in triple_to_quints[t]:
                    old_c = blue_counts[k]
                    blue_counts[k] = old_c + 1
                    if old_c + 1 == 10:
                        total_violations += 1
                
                colors[t] = 1

            else: # Blue -> Red
                # Update Blue counts (decrease)
                for k in triple_to_quints[t]:
                    old_c = blue_counts[k]
                    if old_c == 10:
                        total_violations -= 1
                    blue_counts[k] = old_c - 1
                
                # Update Red counts (increase)
                for q in triple_to_quads[t]:
                    old_c = red_counts[q]
                    red_counts[q] = old_c + 1
                    if old_c + 1 == 4:
                        total_violations += 1
                
                colors[t] = 0

        # Local Search
        # Greedy phase
        max_iterations = 10000
        for it in range(max_iterations):
            if time.time() > start_time + time_limit:
                break
            if total_violations == 0:
                return colors.tolist()
            
            # Find a violating quad or quin and pick a random triple from it to flip
            # Or better: pick a random triple and see if it helps?
            # Greedy: Find any violation, pick a triple from it, flip if it reduces violations.
            
            # To be fast, we can just pick random triples and flip if it reduces violations.
            # This is a stochastic local search.
            
            # Let's do: Pick 100 random triples, flip the one that reduces violations most.
            best_delta = 0
            best_t = -1
            
            # Sample candidates
            for _ in range(100):
                t = rng.randint(M)
                # Calculate delta
                old_color = colors[t]
                delta = 0
                if old_color == 0:
                    # Red -> Blue
                    for q in triple_to_quads[t]:
                        if red_counts[q] == 4: delta -= 1
                        # if red_counts[q] == 3, no change in violation count (4->3 is break, 3->2 is no)
                        # Wait. 
                        # Red->Blue: red_count decreases.
                        # If red_count was 4, violation removed (-1).
                        # If red_count was 3, violation NOT removed (0 change).
                        # So only if red_count[q] == 4 does delta change?
                        # Yes.
                    for k in triple_to_quints[t]:
                        if blue_counts[k] == 9: delta += 1 # Becomes 10
                        # If blue_count was 10, it becomes 9. No violation created/removed?
                        # Wait. Blue->Red is the other way.
                        # Here we are doing Red->Blue.
                        # Blue count increases.
                        # If blue_count was 9, it becomes 10. Violation created (+1).
                        # If blue_count was 10, it becomes 11. No change in violation status (still violation).
                        # So only if blue_count[k] == 9 does delta change?
                        # Yes.
                else:
                    # Blue -> Red
                    for k in triple_to_quints[t]:
                        if blue_counts[k] == 10: delta -= 1
                    for q in triple_to_quads[t]:
                        if red_counts[q] == 3: delta += 1 # Becomes 4
                
                if delta < best_delta:
                    best_delta = delta
                    best_t = t
            
            if best_t != -1:
                flip(best_t)
            else:
                # No improving move, do a random flip (SA style or just random walk)
                t = rng.randint(M)
                flip(t)
        
        # Simulated Annealing phase
        T = 5.0
        cooling = 0.999
        min_T = 0.1
        while time.time() < start_time + time_limit:
            if total_violations == 0:
                return colors.tolist()
            
            t = rng.randint(M)
            
            # Calculate delta
            old_color = colors[t]
            delta = 0
            if old_color == 0:
                for q in triple_to_quads[t]:
                    if red_counts[q] == 4: delta -= 1
                for k in triple_to_quints[t]:
                    if blue_counts[k] == 9: delta += 1
            else:
                for k in triple_to_quints[t]:
                    if blue_counts[k] == 10: delta -= 1
                for q in triple_to_quads[t]:
                    if red_counts[q] == 3: delta += 1
            
            flip(t)
            
            if delta > 0 or (T > min_T and rng.random() < np.exp(-delta / T)):
                pass # Accept
            else:
                flip(t) # Reject (flip back)
            
            T *= cooling
        
        return None

    best_n = 1
    best_colors = [0]

    # Try small n first to verify logic
    # Then jump to larger n
    for n in range(1, 35):
        if time.time() > deadline - 10:
            break
        # Use a time budget for each n
        # If n is large, give it more time
        t_limit = 10.0 if n < 20 else 20.0
        if n >= 30:
            t_limit = 30.0
            
        res = search_for_n(n, t_limit, seed=n)
        if res is not None:
            best_n = n
            best_colors = res
        else:
            # If we fail for a small n, it might be due to bad luck.
            # But for n=1..20, it should always succeed.
            # If it fails for n=20, we might need to restart.
            # Let's just break if we fail for n > 15, assuming we've done our best.
            if n > 15:
                break

    return {'n': best_n, 'colors': best_colors}
