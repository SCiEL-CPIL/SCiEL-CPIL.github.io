"""Stronger program for the suite task; deterministic construction + local repair."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 130s kill

    def get_triples(n):
        return list(combinations(range(n), 3))

    def triple_index(triples, t):
        # Use a dictionary for O(1) lookup
        return triples[t]

    def build_index(n):
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        return triples, triple_to_idx

    def check_validity(n, colors, triple_to_idx=None):
        """Check validity: no red K4^(3) (all 4 triples red) and no blue K5^(3) (all 10 triples blue)."""
        if triple_to_idx is None:
            triples = list(combinations(range(n), 3))
            triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        colors_arr = np.array(colors)
        
        # Red K4 check: for each 4-set, check if all 4 triples are 0
        for quad in combinations(range(n), 4):
            idxs = [
                triple_to_idx[(quad[0], quad[1], quad[2])],
                triple_to_idx[(quad[0], quad[1], quad[3])],
                triple_to_idx[(quad[0], quad[2], quad[3])],
                triple_to_idx[(quad[1], quad[2], quad[3])]
            ]
            if all(colors_arr[i] == 0 for i in idxs):
                return False
        
        # Blue K5 check: for each 5-set, check if all 10 triples are 1
        for quint in combinations(range(n), 5):
            count = 0
            for tri in combinations(quint, 3):
                idx = triple_to_idx[tri]
                if colors_arr[idx] == 1:
                    count += 1
                else:
                    break
            if count == 10:
                return False
        
        return True

    # Strategy 1: Try a known algebraic construction
    # For 3-uniform hypergraphs, one common approach is to use finite fields.
    # Let vertices be elements of GF(q) or a subset.
    # Color triple {a,b,c} red if a+b+c = 0 (or some condition), else blue.
    # But we need to avoid red K4 and blue K5.
    
    # A simpler deterministic approach for small n:
    # Use a "cyclic" coloring based on indices modulo k.
    # However, the known bound is 34, so we likely need a more sophisticated construction
    # or a very efficient local search starting from a good heuristic.

    # Let's try a heuristic initialization:
    # For each triple (a,b,c), color it red (0) if (a+b+c) % 2 == 0, else blue (1).
    # This is just a parity check. Let's see if it works for small n.
    
    def heuristic_coloring(n):
        triples = list(combinations(range(n), 3))
        colors = []
        for a, b, c in triples:
            # Simple heuristic: based on sum modulo 3
            colors.append(0 if (a + b + c) % 3 == 0 else 1)
        return triples, colors

    def local_search(n, initial_colors, max_iters=5000, rng=None, deadline=None):
        """
        Local search to fix violations.
        Returns (colors, violation_count)
        """
        if rng is None:
            rng = random.Random()
            
        triples, triple_to_idx = build_index(n)
        m = len(triples)
        colors = np.array(initial_colors, dtype=np.int8)
        
        # Precompute red K4s and blue K5s as lists of triple indices
        red_k4s = []
        for quad in combinations(range(n), 4):
            idxs = [
                triple_to_idx[(quad[0], quad[1], quad[2])],
                triple_to_idx[(quad[0], quad[1], quad[3])],
                triple_to_idx[(quad[0], quad[2], quad[3])],
                triple_to_idx[(quad[1], quad[2], quad[3])]
            ]
            red_k4s.append(idxs)
            
        blue_k5s = []
        for quint in combinations(range(n), 5):
            idxs = [triple_to_idx[tri] for tri in combinations(quint, 3)]
            blue_k5s.append(idxs)
            
        # Count violations
        def count_violations():
            red_v = 0
            for k4 in red_k4s:
                if np.all(colors[k4] == 0):
                    red_v += 1
            blue_v = 0
            for k5 in blue_k5s:
                if np.all(colors[k5] == 1):
                    blue_v += 1
            return red_v + blue_v, red_v, blue_v
        
        total_v, red_v, blue_v = count_violations()
        best_colors = colors.copy()
        best_total = total_v
        
        for it in range(max_iters):
            if it % 500 == 0:
                if time.time() > deadline - 5:
                    break
                
            if total_v == 0:
                break
                
            # Pick a violation and flip a random triple in it
            if red_v > 0:
                # Pick a random red violation
                k4_idx = rng.randint(0, len(red_k4s) - 1)
                k4 = red_k4s[k4_idx]
                # Check if it's still a violation (colors might have changed)
                if np.all(colors[k4] == 0):
                    flip_i = k4[rng.randint(0, 3)]
                    colors[flip_i] = 1
                    total_v, red_v, blue_v = count_violations()
                    if total_v < best_total:
                        best_total = total_v
                        best_colors = colors.copy()
                else:
                    # If not a violation anymore, do nothing or pick another
                    pass
            else:
                # Pick a random blue violation
                k5_idx = rng.randint(0, len(blue_k5s) - 1)
                k5 = blue_k5s[k5_idx]
                if np.all(colors[k5] == 1):
                    flip_i = k5[rng.randint(0, 9)]
                    colors[flip_i] = 0
                    total_v, red_v, blue_v = count_violations()
                    if total_v < best_total:
                        best_total = total_v
                        best_colors = colors.copy()
                else:
                    pass
                    
        return best_colors, best_total

    best_n = 0
    best_colors = None

    # Try small n first to establish a baseline
    for n in range(9, 12):
        if time.time() > deadline - 10:
            break
        triples, colors = heuristic_coloring(n)
        if check_validity(n, colors):
            best_n = n
            best_colors = colors

    # Try larger n with local search
    # We want to maximize n. Let's try from high to low? Or low to high?
    # Local search is faster if we start close to a solution.
    # Let's try n=20, 30, 34...
    
    target_ns = [15, 20, 25, 30, 34, 35, 36, 37, 38, 39, 40]
    
    # Sort by difficulty? Or just try in increasing order and keep the best?
    # Since we want max n, let's try high n first. If we find a valid one, we can stop?
    # But maybe a lower n is easier to find?
    # Actually, if we find a valid n=34, that's the known bound.
    # Let's try to find a valid n=34.
    
    for n in [34, 35, 36, 37, 38, 39, 40, 33, 32, 30, 25, 20, 15]:
        if time.time() > deadline - 10:
            break
            
        if n <= best_n:
            continue
            
        # Heuristic init
        triples, colors = heuristic_coloring(n)
        
        # Check if already valid
        if check_validity(n, colors):
            best_n = n
            best_colors = colors
            continue
            
        # Local search
        # Use multiple restarts
        found = False
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            rng = random.Random(n * 100 + restart)
            # Add some noise to initial colors
            noisy_colors = colors.copy()
            # Flip 10% of colors
            flip_mask = np.random.default_rng(rng).random(len(noisy_colors)) < 0.1
            noisy_colors[flip_mask] = 1 - noisy_colors[flip_mask]
            
            res_colors, viol_count = local_search(n, noisy_colors, max_iters=2000, rng=rng, deadline=deadline)
            
            if viol_count == 0:
                best_n = n
                best_colors = res_colors.tolist()
                found = True
                break
                
    # Fallback
    if best_n == 0:
        n = 9
        triples, colors = heuristic_coloring(n)
        best_n = n
        best_colors = colors

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
