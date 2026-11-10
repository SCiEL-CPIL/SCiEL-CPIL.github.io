"""
Rewritten program using Cyclic Symmetry Reduction (Orbit-based Search).
The known solution for R(4,5;3) > 34 is highly symmetric.
We reduce the search space from ~6000 variables to ~176 variables (orbits).
This allows for much faster and deeper search.
"""
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    n = 34
    
    # 1. Precompute triples and their orbit representatives
    # We assume a cyclic symmetry C_n acting on vertices 0..n-1.
    # A triple (a,b,c) is equivalent to (a+k, b+k, c+k) mod n.
    # We define the orbit representative as the lexicographically smallest 
    # rotated version of the triple.
    
    triples = list(combinations(range(n), 3))
    M = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Map each triple to its orbit representative index
    # We pick the smallest rotation as the representative.
    triple_to_orbit = np.zeros(M, dtype=np.int32)
    orbit_reps = []
    rep_to_idx = {}
    
    for i, t in enumerate(triples):
        a, b, c = t
        # Generate all rotations
        rotations = []
        for k in range(n):
            ra = (a + k) % n
            rb = (b + k) % n
            rc = (c + k) % n
            # Sort to get canonical form
            rt = tuple(sorted((ra, rb, rc)))
            rotations.append(rt)
        
        # Find lexicographically smallest rotation
        rep = min(rotations)
        
        if rep not in rep_to_idx:
            rep_to_idx[rep] = len(orbit_reps)
            orbit_reps.append(rep)
            
        triple_to_orbit[i] = rep_to_idx[rep]
        
    N_orbits = len(orbit_reps)
    
    # 2. Precompute K4 and K5 edge indices
    k4_list = list(combinations(range(n), 4))
    k5_list = list(combinations(range(n), 5))
    
    N4 = len(k4_list)
    N5 = len(k5_list)
    
    k4_edges = np.zeros((N4, 4), dtype=np.int32)
    k5_edges = np.zeros((N5, 10), dtype=np.int32)
    
    # Precompute K4 edges
    for i, k4 in enumerate(k4_list):
        for j in range(4):
            t = tuple(sorted(k4[:j] + k4[j+1:]))
            idx = triple_to_idx[t]
            k4_edges[i, j] = idx
            
    # Precompute K5 edges
    for i, k5 in enumerate(k5_list):
        for j, combo in enumerate(combinations(range(5), 3)):
            t = tuple(sorted([k5[x] for x in combo]))
            idx = triple_to_idx[t]
            k5_edges[i, j] = idx

    def run_sa_orbits(n_orbits, time_budget):
        # Initialize orbit colors
        orbit_colors = np.random.randint(0, 2, size=n_orbits, dtype=np.int8)
        
        # Expand to full colors for vectorized violation counting
        # colors = orbit_colors[triple_to_orbit]
        
        def get_violations(orbit_colors):
            colors = orbit_colors[triple_to_orbit]
            # Red violations: K4 where all 4 are red (0)
            # Blue violations: K5 where all 10 are blue (1)
            
            # Check K4s
            # k4_edges contains indices into 'colors'
            # We need to check if colors[k4_edges] are all 0.
            # Sum of (1 - colors) should be 4.
            red_counts = np.sum(1 - colors[k4_edges], axis=1)
            red_violations = np.sum(red_counts == 4)
            
            # Check K5s
            # Sum of colors should be 10.
            blue_counts = np.sum(colors[k5_edges], axis=1)
            blue_violations = np.sum(blue_counts == 10)
            
            return int(red_violations + blue_violations), colors

        cur_violations, cur_colors = get_violations(orbit_colors)
        best_violations = cur_violations
        best_orbit_colors = orbit_colors.copy()
        
        T = 50.0
        T_min = 0.1
        cooling = 0.9999
        no_improve = 0
        max_no_improve = 5000
        kicks = 0
        max_kicks = 10
        
        while time.time() < start_time + time_budget:
            if cur_violations == 0:
                return orbit_colors.tolist()
            
            if cur_violations < best_violations:
                best_violations = cur_violations
                best_orbit_colors = orbit_colors.copy()
                no_improve = 0
                T = max(T * 1.05, 10.0)
            else:
                no_improve += 1
                
            if no_improve > max_no_improve:
                kicks += 1
                if kicks > max_kicks:
                    return None
                
                # Kick: flip 20% of orbit bits
                num_flips = max(1, n_orbits // 5)
                flip_indices = np.random.choice(n_orbits, num_flips, replace=False)
                orbit_colors[flip_indices] = 1 - orbit_colors[flip_indices]
                
                cur_violations, _ = get_violations(orbit_colors)
                T = 50.0
                no_improve = 0
                continue
            
            # Single Flip in orbit space
            idx = random.randint(0, n_orbits - 1)
            old_val = orbit_colors[idx]
            new_val = 1 - old_val
            
            # Calculate delta
            # This is expensive to do incrementally in Python loops, 
            # but since N_orbits is small, we can just recalculate full violations.
            # Optimization: Only recalculate if the flip might change violations.
            # But for simplicity and robustness, we recalculate.
            # To speed up, we can check if the orbit is "critical".
            
            orbit_colors[idx] = new_val
            new_violations, _ = get_violations(orbit_colors)
            
            if new_violations < cur_violations:
                cur_violations = new_violations
            else:
                if T > T_min:
                    delta = new_violations - cur_violations
                    if delta > 0:
                        if delta < 100 * T:
                            prob = np.exp(-delta / T)
                            if random.random() < prob:
                                cur_violations = new_violations
                            else:
                                orbit_colors[idx] = old_val # Revert
                        else:
                            orbit_colors[idx] = old_val # Revert
                    else:
                        cur_violations = new_violations # Should not happen if we check <
                else:
                    orbit_colors[idx] = old_val # Revert
                    
            T *= cooling
            
        if best_violations == 0:
            return best_orbit_colors.tolist()
        return None

    # Run SA on orbits
    res = run_sa_orbits(N_orbits, deadline - start_time - 10)
    
    if res is not None:
        # Expand to full colors
        # res is list of orbit colors
        orbit_colors_arr = np.array(res, dtype=np.int8)
        colors = orbit_colors_arr[triple_to_orbit]
        return {'n': 34, 'colors': colors.tolist()}
    
    # Fallback: Try n=33 with same method? Or just return a valid smaller one.
    # For now, return a simple valid coloring for n=9 if n=34 fails.
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

