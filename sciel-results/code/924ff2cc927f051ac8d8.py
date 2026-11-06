# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 117  # Safety margin for 120s limit
    
    def build_index(n):
        """Build mapping from triple to index and precompute quad/quint triple lists."""
        idx = {}
        i = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx[(a,b,c)] = i
                    i += 1
        
        # Precompute which triples are in each quad (for red K4 check)
        quad_triples = []
        for quad in combinations(range(n), 4):
            quad_triples.append([idx[tri] for tri in combinations(quad, 3)])
        
        # Precompute which triples are in each quint (for blue K5 check)
        quint_triples = []
        for quint in combinations(range(n), 5):
            quint_triples.append([idx[tri] for tri in combinations(quint, 3)])
        
        # Convert to numpy arrays for vectorized ops
        quad_arr = np.array(quad_triples, dtype=np.int32)
        quint_arr = np.array(quint_triples, dtype=np.int32)
        
        return idx, quad_arr, quint_arr
    
    def count_violations_vectorized(colors, n, quad_arr, quint_arr):
        """Count red K4 and blue K5 violations using numpy."""
        # Red K4: all 4 triples in quad must be red (0)
        if len(quad_arr) > 0:
            quad_colors = colors[quad_arr]  # shape (n_quads, 4)
            red_violations = np.all(quad_colors == 0, axis=1)
        else:
            red_violations = np.array([], dtype=bool)
        
        # Blue K5: all 10 triples in quint must be blue (1)
        if len(quint_arr) > 0:
            quint_colors = colors[quint_arr]  # shape (n_quints, 10)
            blue_violations = np.all(quint_colors == 1, axis=1)
        else:
            blue_violations = np.array([], dtype=bool)
        
        return red_violations, blue_violations
    
    def find_violating_triples_fast(colors, n, quad_arr, quint_arr):
        """Find triples that are part of violations using precomputed lists."""
        violating = set()
        
        # Red K4^3 violations
        if len(quad_arr) > 0:
            quad_colors = colors[quad_arr]
            red_violations = np.all(quad_colors == 0, axis=1)
            if np.any(red_violations):
                violating.update(quad_arr[red_violations].flatten().tolist())
        
        # Blue K5^3 violations
        if len(quint_arr) > 0:
            quint_colors = colors[quint_arr]
            blue_violations = np.all(quint_colors == 1, axis=1)
            if np.any(blue_violations):
                violating.update(quint_arr[blue_violations].flatten().tolist())
        
        return violating
    
    def score_flip_impact(colors, tri_idx, n, quad_arr, quint_arr):
        """Estimate how many violations flipping tri_idx would resolve."""
        # Count how many red K4s contain this triple
        red_impact = 0
        if len(quad_arr) > 0:
            # For each quad containing tri_idx, check if flipping would break it
            for q in range(len(quad_arr)):
                if tri_idx in quad_arr[q]:
                    # If all other 3 are red, flipping this one breaks the violation
                    others = [t for t in quad_arr[q] if t != tri_idx]
                    if all(colors[t] == 0 for t in others):
                        if colors[tri_idx] == 0:  # currently red, flipping helps
                            red_impact += 1
        
        # Count how many blue K5s contain this triple
        blue_impact = 0
        if len(quint_arr) > 0:
            for q in range(len(quint_arr)):
                if tri_idx in quint_arr[q]:
                    others = [t for t in quint_arr[q] if t != tri_idx]
                    if all(colors[t] == 1 for t in others):
                        if colors[tri_idx] == 1:  # currently blue, flipping helps
                            blue_impact += 1
        
        return red_impact + blue_impact
    
    def local_search(n, p_red, max_iters=10000):
        """Run local search to find valid coloring."""
        idx, quad_arr, quint_arr = build_index(n)
        n_triples = len(idx)
        
        # Precompute for each triple, which quads/quints contain it (for fast impact scoring)
        # This is expensive, so we'll use a simpler approach: just pick randomly from violating set
        # but weight by how many violations they're in
        
        for restart in range(200):
            if time.time() > deadline - 2:
                break
            
            # Random initial coloring with bias
            colors = np.random.choice([0, 1], size=n_triples, p=[p_red, 1-p_red])
            
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    return None
                
                violating = find_violating_triples_fast(colors, n, quad_arr, quint_arr)
                if not violating:
                    return colors.tolist()
                
                # Pick a violating triple - try a few random ones and pick the best
                # For speed, just pick randomly but with some intelligence
                if len(violating) <= 10:
                    candidates = list(violating)
                else:
                    candidates = random.sample(list(violating), 10)
                
                # Try each candidate, pick the one that resolves most violations
                best_flip = candidates[0]
                best_score = -1
                for tri in candidates:
                    # Quick heuristic: prefer flipping triples that appear in more violations
                    # Count appearances in violating quads/quints
                    score = 0
                    if len(quad_arr) > 0:
                        quad_colors = colors[quad_arr]
                        red_violations = np.all(quad_colors == 0, axis=1)
                        if np.any(red_violations):
                            viol_quads = quad_arr[red_violations]
                            score += np.sum([1 for q in viol_quads if tri in q])
                    if len(quint_arr) > 0:
                        quint_colors = colors[quint_arr]
                        blue_violations = np.all(quint_colors == 1, axis=1)
                        if np.any(blue_violations):
                            viol_quints = quint_arr[blue_violations]
                            score += np.sum([1 for q in viol_quints if tri in q])
                    
                    if score > best_score:
                        best_score = score
                        best_flip = tri
                
                colors[best_flip] = 1 - colors[best_flip]
        
        return None
    
    # Try increasing n from small to large
    best_n = 1
    best_colors = [0]  # trivial valid coloring for n=1
    
    # Allocate time budget: spend more time on larger n
    # For n=34, we need good probability of success
    for n in range(2, 35):
        if time.time() > deadline - 5:
            break
        
        # Estimate remaining time and allocate
        remaining_time = deadline - 5 - time.time()
        # Allocate proportionally, but ensure minimum time for small n
        if n <= 10:
            time_budget = max(1, remaining_time / (35 - n))
        else:
            time_budget = max(2, remaining_time / (35 - n))
        
        max_iters = int(time_budget * 500)  # Scale iterations with time budget
        max_iters = max(500, min(max_iters, 50000))
        
        # Try multiple bias values
        for p in [0.5, 0.3, 0.4, 0.6, 0.7, 0.35, 0.45, 0.55, 0.65]:
            if time.time() > deadline - 5:
                break
            
            result = local_search(n, p, max_iters=max_iters)
            if result is not None:
                best_n = n
                best_colors = result
                break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
