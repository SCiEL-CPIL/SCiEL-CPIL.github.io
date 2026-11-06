"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from math import comb

def construct():
    """Construct a valid 2-coloring of triples maximizing n.
    
    Returns dict with 'n' and 'colors' (list of 0/1 for each triple in lex order).
    """
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit
    
    def fast_local_search(n, deadline, max_restarts=10):
        """
        Fast local search using incremental violation counts.
        """
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        
        # Map triple tuple to index
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute quads and quints
        quads = list(combinations(range(n), 4))
        quints = list(combinations(range(n), 5))
        
        # For each triple, find which quads and quints it belongs to
        triple_to_quads = [[] for _ in range(num_triples)]
        triple_to_quints = [[] for _ in range(num_triples)]
        
        # For each quad, find its triples
        quad_triples = [[] for _ in range(len(quads))]
        # For each quint, find its triples
        quint_triples = [[] for _ in range(len(quints))]
        
        # Build mappings
        # Iterate through all quads
        for qi, quad in enumerate(quads):
            a, b, c, d = quad
            t1 = (a, b, c); t2 = (a, b, d); t3 = (a, c, d); t4 = (b, c, d)
            idxs = [triple_to_idx[t1], triple_to_idx[t2], triple_to_idx[t3], triple_to_idx[t4]]
            quad_triples[qi] = idxs
            for idx in idxs:
                triple_to_quads[idx].append(qi)
                
        # Iterate through all quints
        for qi, quint in enumerate(quints):
            trs = list(combinations(quint, 3))
            idxs = [triple_to_idx[t] for t in trs]
            quint_triples[qi] = idxs
            for idx in idxs:
                triple_to_quints[idx].append(qi)

        best_colors = None
        best_violations = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            # Random initial coloring
            colors = np.random.randint(0, 2, num_triples)
            
            # Initialize counts
            quad_red = np.zeros(len(quads), dtype=int)
            quint_blue = np.zeros(len(quints), dtype=int)
            
            # Calculate initial counts
            for t_idx in range(num_triples):
                if colors[t_idx] == 0: # Red
                    for qi in triple_to_quads[t_idx]:
                        quad_red[qi] += 1
                else: # Blue
                    for qi in triple_to_quints[t_idx]:
                        quint_blue[qi] += 1
            
            # Identify violations
            red_violations = set(np.where(quad_red == 4)[0].tolist())
            blue_violations = set(np.where(quint_blue == 10)[0].tolist())
            
            current_violations = len(red_violations) + len(blue_violations)
            if current_violations == 0:
                return colors.tolist(), True
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors.tolist()
            
            # Local search loop
            stagnant = 0
            while time.time() < deadline - 1:
                if not red_violations and not blue_violations:
                    return colors.tolist(), True
                    
                # Pick a random violation
                all_violations = list(red_violations) + list(blue_violations)
                if not all_violations:
                    break
                    
                v_idx = random.choice(all_violations)
                is_red = v_idx in red_violations
                
                # Get the triples associated with this violation
                if is_red:
                    trs = quad_triples[v_idx]
                    # Only consider flipping red triples to blue
                    candidates = [t for t in trs if colors[t] == 0]
                else:
                    trs = quint_triples[v_idx]
                    # Only consider flipping blue triples to red
                    candidates = [t for t in trs if colors[t] == 1]
                
                if not candidates:
                    stagnant += 1
                    if stagnant > 100: break
                    continue
                
                best_delta = 0
                best_candidate = None
                
                for t in candidates:
                    # Calculate delta violations if we flip t
                    delta = 0
                    if colors[t] == 0: # Red -> Blue
                        # Red quads decrease
                        for qi in triple_to_quads[t]:
                            if quad_red[qi] == 4:
                                delta -= 1 # Violation removed
                        # Blue quints increase
                        for qi in triple_to_quints[t]:
                            if quint_blue[qi] == 9:
                                delta += 1 # Violation created
                    else: # Blue -> Red
                        # Red quads increase
                        for qi in triple_to_quads[t]:
                            if quad_red[qi] == 3:
                                delta += 1 # Violation created
                        # Blue quints decrease
                        for qi in triple_to_quints[t]:
                            if quint_blue[qi] == 10:
                                delta -= 1 # Violation removed
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_candidate = t
                        
                if best_candidate is not None:
                    # Apply flip
                    t = best_candidate
                    old_color = colors[t]
                    colors[t] = 1 - old_color
                    
                    # Update counts
                    if old_color == 0: # Was Red, now Blue
                        for qi in triple_to_quads[t]:
                            quad_red[qi] -= 1
                            if quad_red[qi] == 3 and qi in red_violations:
                                red_violations.remove(qi)
                        for qi in triple_to_quints[t]:
                            quint_blue[qi] += 1
                            if quint_blue[qi] == 10 and qi not in blue_violations:
                                blue_violations.add(qi)
                    else: # Was Blue, now Red
                        for qi in triple_to_quads[t]:
                            quad_red[qi] += 1
                            if quad_red[qi] == 4 and qi not in red_violations:
                                red_violations.add(qi)
                        for qi in triple_to_quints[t]:
                            quint_blue[qi] -= 1
                            if quint_blue[qi] == 9 and qi in blue_violations:
                                blue_violations.remove(qi)
                    
                    stagnant = 0
                else:
                    stagnant += 1
                    if stagnant > 500:
                        break
            
            # Update best
            current_violations = len(red_violations) + len(blue_violations)
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors.tolist()
                
        return best_colors, False

    # Try to find the maximum n with a valid coloring
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Prioritize higher n
    # We know n=34 is possible. Try 34 first.
    for n in [34, 33, 32, 30, 28, 25]:
        if time.time() > deadline - 5:
            break
        
        colors, valid = fast_local_search(n, deadline, max_restarts=20)
        if valid:
            best_n = n
            best_colors = colors
            break # Found a valid one for this n, stop trying lower n? 
            # Actually, we want MAX n. So if we find 34, we are done.
            # If we fail 34, we try 33.
    
    # If we didn't find a valid one for 34, we might have a partial result (min violations)
    # But the task requires a VALID coloring.
    # If we return an invalid coloring, it scores 0.
    # So we must only return if valid.
    
    # If we didn't find any valid coloring for the tried n's, we should return a small valid one
    # e.g. n=9 is always valid?
    # Let's ensure we have a fallback valid coloring.
    # n=9: C(9,3)=84.
    # We can just return the n=9 coloring if nothing else works.
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
