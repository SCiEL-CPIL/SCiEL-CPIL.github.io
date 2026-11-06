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
    
    def local_search(n, p_red, max_iters=1000):
        """Run local search with incremental violation tracking."""
        idx, n_triples, quads, quints, triple_to_quads, triple_to_quints = build_data(n)
        
        for restart in range(30):
            if time.time() > deadline - 5:
                break
            
            # Random initial coloring with bias
            colors = np.random.choice([0, 1], size=n_triples, p=[p_red, 1-p_red])
            
            # Track violation status of each quad and quint
            # red_quad_viol[qi] = True if quad qi is a red violation
            # blue_quint_viol[qi] = True if quint qi is a blue violation
            red_quad_viol = np.zeros(len(quads), dtype=bool)
            blue_quint_viol = np.zeros(len(quints), dtype=bool)
            
            # Initialize
            for qi, tri_indices in enumerate(quads):
                if all(colors[i] == 0 for i in tri_indices):
                    red_quad_viol[qi] = True
            
            for qi, tri_indices in enumerate(quints):
                if all(colors[i] == 1 for i in tri_indices):
                    blue_quint_viol[qi] = True
            
            red_count = np.sum(red_quad_viol)
            blue_count = np.sum(blue_quint_viol)
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                if red_count == 0 and blue_count == 0:
                    return colors.tolist()
                
                # Find the triple that appears in the most violations
                # and flipping it would help most
                best_idx = -1
                best_gain = -1
                
                # Sample a subset of triples to check (for speed)
                sample_size = min(n_triples, 200)
                candidates = random.sample(range(n_triples), sample_size)
                
                for ti in candidates:
                    current_color = colors[ti]
                    
                    if current_color == 0:
                        # Currently red, consider flipping to blue
                        # Gain: number of red violations this triple is in
                        gain = sum(red_quad_viol[qi] for qi in triple_to_quads[ti])
                    else:
                        # Currently blue, consider flipping to red
                        gain = sum(blue_quint_viol[qi] for qi in triple_to_quints[ti])
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = ti
                
                if best_idx == -1 or best_gain <= 0:
                    # No good flip found, try random
                    best_idx = random.randint(0, n_triples - 1)
                    best_gain = 0
                
                # Flip the triple
                old_color = colors[best_idx]
                new_color = 1 - old_color
                colors[best_idx] = new_color
                
                # Update violation status for affected quads and quints
                if old_color == 0:
                    # Was red, now blue
                    # Check all quads containing this triple
                    for qi in triple_to_quads[best_idx]:
                        if red_quad_viol[qi]:
                            # Was a violation, check if still is
                            tri_indices = quads[qi]
                            still_red = all(colors[i] == 0 for i in tri_indices)
                            if not still_red:
                                red_quad_viol[qi] = False
                                red_count -= 1
                else:
                    # Was blue, now red
                    # Check all quints containing this triple
                    for qi in triple_to_quints[best_idx]:
                        if blue_quint_viol[qi]:
                            # Was a violation, check if still is
                            tri_indices = quints[qi]
                            still_blue = all(colors[i] == 1 for i in tri_indices)
                            if not still_blue:
                                blue_quint_viol[qi] = False
                                blue_count -= 1
                    
                    # Also check if flipping to red creates new red violations
                    for qi in triple_to_quads[best_idx]:
                        if not red_quad_viol[qi]:
                            tri_indices = quads[qi]
                            all_red = all(colors[i] == 0 for i in tri_indices)
                            if all_red:
                                red_quad_viol[qi] = True
                                red_count += 1
                
                if old_color == 1:
                    # Was blue, now red - also check for new blue violations? No, we removed a blue triple
                    pass
                else:
                    # Was red, now blue - check if this creates new blue violations
                    for qi in triple_to_quints[best_idx]:
                        if not blue_quint_viol[qi]:
                            tri_indices = quints[qi]
                            all_blue = all(colors[i] == 1 for i in tri_indices)
                            if all_blue:
                                blue_quint_viol[qi] = True
                                blue_count += 1
        
        return None
    
    # Try increasing n
    best_n = 1
    best_colors = [0]  # trivial valid coloring for n=1
    
    for n in range(2, 35):
        if time.time() > deadline - 10:
            break
        
        found = False
        for p in [0.5, 0.4, 0.6]:
            if time.time() > deadline - 10:
                break
            
            result = local_search(n, p, max_iters=500)
            if result is not None:
                best_n = n
                best_colors = result
                found = True
                break
        
        if not found:
            # Try once more with different parameters
            if time.time() <= deadline - 10:
                result = local_search(n, 0.5, max_iters=300)
                if result is not None:
                    best_n = n
                    best_colors = result
                else:
                    # Give up on this n
                    break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
