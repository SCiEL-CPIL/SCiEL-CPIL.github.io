"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    """
    Find a valid 2-colouring of triples on n vertices with no red K4^(3) and no blue K5^(3).
    Maximise n. The known construction has n=34.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    target_n = 34
    
    # Precompute all triples in lexicographic order
    triples = list(combinations(range(target_n), 3))
    num_triples = len(triples)
    
    # Map triple -> index
    triple_index = {}
    for i, t in enumerate(triples):
        triple_index[t] = i
    
    # For each 4-set, the 4 triples that form K4^(3)
    k4_triple_indices = []
    for quad in combinations(range(target_n), 4):
        a, b, c, d = quad
        idxs = [
            triple_index[(a, b, c)],
            triple_index[(a, b, d)],
            triple_index[(a, c, d)],
            triple_index[(b, c, d)]
        ]
        k4_triple_indices.append(idxs)
    
    # For each 5-set, the 10 triples that form K5^(3)
    k5_triple_indices = []
    for quint in combinations(range(target_n), 5):
        a, b, c, d, e = quint
        idxs = [
            triple_index[(a, b, c)],
            triple_index[(a, b, d)],
            triple_index[(a, b, e)],
            triple_index[(a, c, d)],
            triple_index[(a, c, e)],
            triple_index[(a, d, e)],
            triple_index[(b, c, d)],
            triple_index[(b, c, e)],
            triple_index[(b, d, e)],
            triple_index[(c, d, e)]
        ]
        k5_triple_indices.append(idxs)
    
    num_k4 = len(k4_triple_indices)
    num_k5 = len(k5_triple_indices)
    
    # Build reverse maps: triple -> which K4s and K5s it belongs to
    triple_to_k4s = [[] for _ in range(num_triples)]
    for k4_i, idxs in enumerate(k4_triple_indices):
        for tri_idx in idxs:
            triple_to_k4s[tri_idx].append(k4_i)
    
    triple_to_k5s = [[] for _ in range(num_triples)]
    for k5_i, idxs in enumerate(k5_triple_indices):
        for tri_idx in idxs:
            triple_to_k5s[tri_idx].append(k5_i)
            
    # Precompute red_count and blue_count for each K4 and K5
    # This allows O(1) update per flip
    
    def simulate_annealing(colors, max_iters=100000):
        """
        Simulated annealing with efficient delta calculation.
        colors: list of 0s and 1s.
        """
        # Convert to numpy array for faster access if needed, but list is fine for small n
        # We maintain counts of reds in each K4 and blues in each K5
        red_counts_k4 = np.zeros(num_k4, dtype=np.int8)
        blue_counts_k5 = np.zeros(num_k5, dtype=np.int8)
        
        # Initialize counts
        for k4_i, idxs in enumerate(k4_triple_indices):
            red_counts_k4[k4_i] = sum(1 for idx in idxs if colors[idx] == 0)
        for k5_i, idxs in enumerate(k5_triple_indices):
            blue_counts_k5[k5_i] = sum(1 for idx in idxs if colors[idx] == 1)
            
        # Calculate initial penalty
        current_penalty = np.sum(red_counts_k4 == 4) + np.sum(blue_counts_k5 == 10)
        
        best_penalty = current_penalty
        best_colors = colors[:]
        
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick random triple to flip
            t = random.randrange(num_triples)
            old_color = colors[t]
            
            # Calculate delta penalty
            delta = 0
            
            if old_color == 0: # Red -> Blue
                # Update K4s: red count decreases by 1
                for k4_i in triple_to_k4s[t]:
                    if red_counts_k4[k4_i] == 4:
                        delta -= 1 # Violation removed
                    red_counts_k4[k4_i] -= 1
                # Update K5s: blue count increases by 1
                for k5_i in triple_to_k5s[t]:
                    if blue_counts_k5[k5_i] == 9:
                        delta += 1 # New violation
                    blue_counts_k5[k5_i] += 1
            else: # Blue -> Red
                # Update K4s: red count increases by 1
                for k4_i in triple_to_k4s[t]:
                    if red_counts_k4[k4_i] == 3:
                        delta += 1 # New violation
                    red_counts_k4[k4_i] += 1
                # Update K5s: blue count decreases by 1
                for k5_i in triple_to_k5s[t]:
                    if blue_counts_k5[k5_i] == 10:
                        delta -= 1 # Violation removed
                    blue_counts_k5[k5_i] -= 1
                    
            new_penalty = current_penalty + delta
            
            # Acceptance criterion
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < math.exp(-delta / T)):
                colors[t] = 1 - old_color
                current_penalty = new_penalty
                
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                    best_colors = colors[:]
                    
            else:
                # Revert counts
                if old_color == 0: # We didn't flip, so counts were changed, need to revert
                    for k4_i in triple_to_k4s[t]:
                        red_counts_k4[k4_i] += 1
                    for k5_i in triple_to_k5s[t]:
                        blue_counts_k5[k5_i] -= 1
                else:
                    for k4_i in triple_to_k4s[t]:
                        red_counts_k4[k4_i] -= 1
                    for k5_i in triple_to_k5s[t]:
                        blue_counts_k5[k5_i] += 1

            T *= cooling_rate
            if T < min_T:
                T = min_T
                
            if best_penalty == 0:
                break
                
        return best_penalty, best_colors

    best_overall_penalty = float('inf')
    best_overall_colors = None
    
    # Multiple restarts
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Bias towards red or blue? 
        # Random is usually fine, but maybe 0.5 is best.
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Run Simulated Annealing
        penalty, sa_colors = simulate_annealing(colors, max_iters=50000)
        
        if penalty < best_overall_penalty:
            best_overall_penalty = penalty
            best_overall_colors = sa_colors[:]
            
        if best_overall_penalty == 0:
            break
            
    if best_overall_penalty == 0 and best_overall_colors is not None:
        return {'n': target_n, 'colors': best_overall_colors}
        
    # If SA didn't find it, try a few more with higher temperature or different init
    # Or just return best found if valid? No, must be valid.
    # If we don't have a valid one, we must return a valid one for smaller n or fallback.
    
    # Fallback: try smaller n with greedy or SA
    # Given time constraints, if we didn't find n=34, we might run out of time.
    # But let's try to salvage with a quick greedy for n=33 or 32 if time permits.
    
    if time.time() < deadline - 10:
        for n in range(33, 20, -1):
            if time.time() > deadline - 5:
                break
                
            triples_n = list(combinations(range(n), 3))
            num_triples_n = len(triples_n)
            triple_index_n = {}
            for i, t in enumerate(triples_n):
                triple_index_n[t] = i
                
            k4_triple_indices_n = []
            for quad in combinations(range(n), 4):
                a, b, c, d = quad
                idxs = [
                    triple_index_n[(a, b, c)],
                    triple_index_n[(a, b, d)],
                    triple_index_n[(a, c, d)],
                    triple_index_n[(b, c, d)]
                ]
                k4_triple_indices_n.append(idxs)
                
            k5_triple_indices_n = []
            for quint in combinations(range(n), 5):
                a, b, c, d, e = quint
                idxs = [
                    triple_index_n[(a, b, c)],
                    triple_index_n[(a, b, d)],
                    triple_index_n[(a, b, e)],
                    triple_index_n[(a, c, d)],
                    triple_index_n[(a, c, e)],
                    triple_index_n[(a, d, e)],
                    triple_index_n[(b, c, d)],
                    triple_index_n[(b, c, e)],
                    triple_index_n[(b, d, e)],
                    triple_index_n[(c, d, e)]
                ]
                k5_triple_indices_n.append(idxs)
                
            # Simple Greedy
            colors_n = [0] * num_triples_n
            valid = True
            for tri_idx in range(num_triples_n):
                # Try red
                colors_n[tri_idx] = 0
                violation = False
                for k4_i, idxs in enumerate(k4_triple_indices_n):
                    if tri_idx in idxs:
                        red_count = sum(1 for idx in idxs if colors_n[idx] == 0)
                        if red_count == 4:
                            violation = True
                            break
                if not violation:
                    continue
                # Try blue
                colors_n[tri_idx] = 1
                violation = False
                for k5_i, idxs in enumerate(k5_triple_indices_n):
                    if tri_idx in idxs:
                        blue_count = sum(1 for idx in idxs if colors_n[idx] == 1)
                        if blue_count == 10:
                            violation = True
                            break
                if violation:
                    valid = False
                    break
                    
            if valid:
                # Verify
                is_valid = True
                for k4_i, idxs in enumerate(k4_triple_indices_n):
                    red_count = sum(1 for idx in idxs if colors_n[idx] == 0)
                    if red_count == 4:
                        is_valid = False
                        break
                if is_valid:
                    for k5_i, idxs in enumerate(k5_triple_indices_n):
                        blue_count = sum(1 for idx in idxs if colors_n[idx] == 1)
                        if blue_count == 10:
                            is_valid = False
                            break
                if is_valid:
                    return {'n': n, 'colors': colors_n}
                    
    # Final fallback: very small construction
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
# EVOLVE-BLOCK-END
