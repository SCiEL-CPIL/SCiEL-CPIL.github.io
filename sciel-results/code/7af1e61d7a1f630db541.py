# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    def build_data(n):
        """Build index mapping and precompute quad/quint triple lists."""
        idx = {}
        i = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx[(a,b,c)] = i
                    i += 1
        n_triples = i
        
        # Precompute all quads and their triple indices
        quads = []
        for quad in combinations(range(n), 4):
            tri_indices = [idx[t] for t in combinations(quad, 3)]
            quads.append(tri_indices)
        
        # Precompute all quints and their triple indices
        quints = []
        for quint in combinations(range(n), 5):
            tri_indices = [idx[t] for t in combinations(quint, 3)]
            quints.append(tri_indices)
        
        # Build reverse index: for each triple, which quads/quints contain it
        triple_to_quads = [[] for _ in range(n_triples)]
        for qi, tri_indices in enumerate(quads):
            for ti in tri_indices:
                triple_to_quads[ti].append(qi)
        
        triple_to_quints = [[] for _ in range(n_triples)]
        for qi, tri_indices in enumerate(quints):
            for ti in tri_indices:
                triple_to_quints[ti].append(qi)
        
        return idx, n_triples, quads, quints, triple_to_quads, triple_to_quints
    
    def count_violations_full(colors, quads, quints):
        """Count all violations from scratch."""
        red_violations = 0
        for tri_indices in quads:
            if all(colors[i] == 0 for i in tri_indices):
                red_violations += 1
        
        blue_violations = 0
        for tri_indices in quints:
            if all(colors[i] == 1 for i in tri_indices):
                blue_violations += 1
        
        return red_violations, blue_violations
    
    def local_search_sa(n, p_red, max_iters=2000):
        """Run simulated annealing local search."""
        idx, n_triples, quads, quints, triple_to_quads, triple_to_quints = build_data(n)
        n_quads = len(quads)
        n_quints = len(quints)
        
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random initial coloring with bias toward red (reduces blue K5 violations)
            colors = np.random.choice([0, 1], size=n_triples, p=[p_red, 1-p_red])
            
            # Track violation status of each quad and quint
            red_quad_viol = np.zeros(n_quads, dtype=bool)
            blue_quint_viol = np.zeros(n_quints, dtype=bool)
            
            # Initialize violation tracking
            for qi, tri_indices in enumerate(quads):
                if all(colors[i] == 0 for i in tri_indices):
                    red_quad_viol[qi] = True
            
            for qi, tri_indices in enumerate(quints):
                if all(colors[i] == 1 for i in tri_indices):
                    blue_quint_viol[qi] = True
            
            red_count = int(np.sum(red_quad_viol))
            blue_count = int(np.sum(blue_quint_viol))
            total_viol = red_count + blue_count
            
            if total_viol == 0:
                return colors.tolist()
            
            best_colors = colors.copy()
            best_total = total_viol
            
            # Simulated annealing parameters
            T = 1.0
            cooling = 0.995  # Faster cooling for more effective search
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    if best_total == 0:
                        return best_colors.tolist()
                    break
                
                # Pick a random triple to flip
                ti = random.randint(0, n_triples - 1)
                old_color = colors[ti]
                new_color = 1 - old_color
                
                # Compute delta violation count
                delta = 0
                if old_color == 0:
                    # Red -> Blue: may break red violations, may create blue violations
                    for qi in triple_to_quads[ti]:
                        if red_quad_viol[qi]:
                            tri_idx = quads[qi]
                            if not all(colors[j] == 0 for j in tri_idx if j != ti):
                                delta -= 1
                    for qi in triple_to_quints[ti]:
                        if not blue_quint_viol[qi]:
                            tri_idx = quints[qi]
                            if all(colors[j] == 1 for j in tri_idx if j != ti):
                                delta += 1
                else:
                    # Blue -> Red: may break blue violations, may create red violations
                    for qi in triple_to_quints[ti]:
                        if blue_quint_viol[qi]:
                            tri_idx = quints[qi]
                            if not all(colors[j] == 1 for j in tri_idx if j != ti):
                                delta -= 1
                    for qi in triple_to_quads[ti]:
                        if not red_quad_viol[qi]:
                            tri_idx = quads[qi]
                            if all(colors[j] == 0 for j in tri_idx if j != ti):
                                delta += 1
                
                # Accept if improvement or with probability exp(-delta/T)
                if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                    # Apply the flip
                    colors[ti] = new_color
                    
                    # Update violation tracking
                    if old_color == 0:
                        # Red -> Blue
                        for qi in triple_to_quads[ti]:
                            if red_quad_viol[qi]:
                                tri_idx = quads[qi]
                                if not all(colors[j] == 0 for j in tri_idx):
                                    red_quad_viol[qi] = False
                                    red_count -= 1
                        for qi in triple_to_quints[ti]:
                            if not blue_quint_viol[qi]:
                                tri_idx = quints[qi]
                                if all(colors[j] == 1 for j in tri_idx):
                                    blue_quint_viol[qi] = True
                                    blue_count += 1
                    else:
                        # Blue -> Red
                        for qi in triple_to_quints[ti]:
                            if blue_quint_viol[qi]:
                                tri_idx = quints[qi]
                                if not all(colors[j] == 1 for j in tri_idx):
                                    blue_quint_viol[qi] = False
                                    blue_count -= 1
                        for qi in triple_to_quads[ti]:
                            if not red_quad_viol[qi]:
                                tri_idx = quads[qi]
                                if all(colors[j] == 0 for j in tri_idx):
                                    red_quad_viol[qi] = True
                                    red_count += 1
                    
                    total_viol = red_count + blue_count
                    
                    if total_viol < best_total:
                        best_total = total_viol
                        best_colors = colors.copy()
                        if best_total == 0:
                            return best_colors.tolist()
                
                # Cool down
                T *= cooling
        
        return None
    
    # Try increasing n
    best_n = 1
    best_colors = [0]  # trivial valid coloring for n=1
    
    for n in range(2, 35):
        if time.time() > deadline - 10:
            break
        
        found = False
        for p in [0.6, 0.5, 0.4]:
            if time.time() > deadline - 10:
                break
            
            result = local_search_sa(n, p, max_iters=1500)
            if result is not None:
                best_n = n
                best_colors = result
                found = True
                break
        
        if not found:
            # Try once more with different parameters
            if time.time() <= deadline - 10:
                result = local_search_sa(n, 0.5, max_iters=800)
                if result is not None:
                    best_n = n
                    best_colors = result
                else:
                    # Give up on this n
                    break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
