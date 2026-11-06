import time
import random
import numpy as np
from itertools import combinations
from math import comb

def construct():
    """Construct a valid 2-coloring of triples maximizing n."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def try_find_coloring(n, max_time):
        """
        Try to find a valid coloring for n vertices using incremental local search.
        """
        if time.time() > deadline - 5:
            return None
        
        # Precompute triples
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        
        # Map triple to index
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute for each triple, the 4-sets it belongs to
        # A triple (a,b,c) is in 4-sets (a,b,c,d) for d not in {a,b,c}
        # We need the index of the 4-set to update counts
        # Let's index 4-sets lexicographically
        quads = list(combinations(range(n), 4))
        num_quads = len(quads)
        quad_to_idx = {q: i for i, q in enumerate(quads)}
        
        # For each triple, list of quad indices it belongs to
        tri_to_quads = [[] for _ in range(num_triples)]
        for q_idx, q in enumerate(quads):
            a, b, c, d = q
            # The 4 triples in this quad
            t1 = (a, b, c)
            t2 = (a, b, d)
            t3 = (a, c, d)
            t4 = (b, c, d)
            
            i1 = triple_to_idx[t1]
            i2 = triple_to_idx[t2]
            i3 = triple_to_idx[t3]
            i4 = triple_to_idx[t4]
            
            tri_to_quads[i1].append(q_idx)
            tri_to_quads[i2].append(q_idx)
            tri_to_quads[i3].append(q_idx)
            tri_to_quads[i4].append(q_idx)

        # Precompute for each triple, the 5-sets it belongs to
        quints = list(combinations(range(n), 5))
        num_quints = len(quints)
        quint_to_idx = {q: i for i, q in enumerate(quints)}
        
        tri_to_quints = [[] for _ in range(num_triples)]
        for q_idx, q in enumerate(quints):
            # 10 triples in a 5-set
            for t in combinations(q, 3):
                i_t = triple_to_idx[t]
                tri_to_quints[i_t].append(q_idx)

        # Initialize colors
        # Heuristic: random or modular
        # Let's try a few different initializations
        best_valid_colors = None
        
        num_restarts = 0
        max_restarts = 20
        
        while time.time() < deadline - 5 and num_restarts < max_restarts:
            num_restarts += 1
            
            # Random initialization
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            
            # Calculate initial red counts for quads
            # red_count[q] = number of red triples in quad q
            red_counts = [0] * num_quads
            for t_idx, t in enumerate(triples):
                if colors[t_idx] == 0: # Red
                    for q_idx in tri_to_quads[t_idx]:
                        red_counts[q_idx] += 1
            
            # Calculate initial blue counts for quints
            # blue_count[q] = number of blue triples in quint q
            blue_counts = [0] * num_quints
            for t_idx, t in enumerate(triples):
                if colors[t_idx] == 1: # Blue
                    for q_idx in tri_to_quints[t_idx]:
                        blue_counts[q_idx] += 1
            
            # Count violations
            red_violations = sum(1 for c in red_counts if c == 4)
            blue_violations = sum(1 for c in blue_counts if c == 10)
            
            if red_violations == 0 and blue_violations == 0:
                return colors
            
            # Local Search
            no_improve_count = 0
            max_no_improve = 5000
            
            while time.time() < deadline - 5 and no_improve_count < max_no_improve:
                if red_violations == 0 and blue_violations == 0:
                    return colors
                
                # Pick a random violation
                if red_violations > 0:
                    # Find a random red violation
                    # We can iterate until we find one
                    # Optimization: keep a list of violating quads?
                    # For now, just sample
                    while True:
                        q_idx = random.randint(0, num_quads - 1)
                        if red_counts[q_idx] == 4:
                            break
                    quad = quads[q_idx]
                    # Pick a random triple from this quad
                    # The 4 triples in this quad
                    a, b, c, d = quad
                    candidates = [
                        triple_to_idx[(a,b,c)],
                        triple_to_idx[(a,b,d)],
                        triple_to_idx[(a,c,d)],
                        triple_to_idx[(b,c,d)]
                    ]
                    t_idx = random.choice(candidates)
                    # Flip red to blue
                    colors[t_idx] = 1
                    
                    # Update red_counts
                    for q_i in tri_to_quads[t_idx]:
                        red_counts[q_i] -= 1
                        if red_counts[q_i] == 3: # Was 4, now 3, violation resolved
                            red_violations -= 1
                        elif red_counts[q_i] == 4: # Was 5? No, max is 4.
                            pass
                        # If it was 4 and became 3, red_violations decreases.
                        # If it was <4, no change in violation count.
                        # Wait, if red_counts[q_i] was 4, it became 3. So red_violations -= 1.
                        # If red_counts[q_i] was 3, it became 2. No change.
                        # So: if red_counts[q_i] + 1 == 4 (i.e. was 4), red_violations -= 1.
                        # Let's re-evaluate:
                        # Before flip: c_old. After: c_new = c_old - 1.
                        # If c_old == 4, violation removed. red_violations -= 1.
                        # If c_old != 4, no change in violation count.
                        # So we need to check the OLD value.
                        # My code above: red_counts[q_i] -= 1. Then checks if red_counts[q_i] == 3.
                        # If it is 3, it was 4. Correct.
                    
                    # Update blue_counts
                    for q_i in tri_to_quints[t_idx]:
                        old_blue = blue_counts[q_i]
                        blue_counts[q_i] += 1
                        if old_blue == 9: # Was 9, now 10, new violation
                            blue_violations += 1
                        # If old_blue < 9, no change in violation count.
                    
                    # Check if total violations decreased
                    # We don't track total explicitly, we track red_violations and blue_violations.
                    # We need to compare (red_violations + blue_violations) before and after?
                    # It's easier to just accept if we reduced red_violations or blue_violations.
                    # But we might increase one while decreasing the other.
                    # Let's store old total.
                    # This is getting complicated with in-place updates.
                    # Let's try a simpler strategy: 
                    # 1. Pick a violation.
                    # 2. Flip a triple in it.
                    # 3. Recalculate local impact.
                    # 4. If total violations decreased, keep. Else revert.
                    
                    # To do this efficiently, we need to know the change in total violations.
                    # Change in red_violations: 
                    #   For each quad in tri_to_quads[t_idx]:
                    #     if old_red == 4: red_violations -= 1
                    #     if new_red == 4: red_violations += 1 (impossible here, we decreased red count)
                    #   So delta_red = - (number of quads where old_red == 4)
                    
                    # Change in blue_violations:
                    #   For each quint in tri_to_quints[t_idx]:
                    #     if old_blue == 9: blue_violations += 1
                    #     if new_blue == 10: blue_violations += 1 (same)
                    #   So delta_blue = + (number of quints where old_blue == 9)
                    
                    # Let's redo the update logic to calculate delta first.
                    
                else: # blue_violations > 0
                    while True:
                        q_idx = random.randint(0, num_quints - 1)
                        if blue_counts[q_idx] == 10:
                            break
                    quint = quints[q_idx]
                    # Pick a random triple from this quint
                    # There are 10 triples. We can pick one randomly from the 10.
                    # We have the list of triples in this quint? 
                    # We can just iterate combinations of the quint.
                    t_candidates = list(combinations(quint, 3))
                    t = random.choice(t_candidates)
                    t_idx = triple_to_idx[t]
                    # Flip blue to red
                    colors[t_idx] = 0
                    
                    # Update logic similar to above.
                
                # The logic above is messy. Let's implement a clean "try flip" function.
                
                # Break out of this restart if it's taking too long without improvement
                # This is a placeholder, the real logic needs to be tighter.
                no_improve_count += 1
                if no_improve_count % 100 == 0:
                    # Check time
                    if time.time() > deadline - 5:
                        break
            
        return None

    # Try n=34
    # If that fails, try 33, 32, etc.
    for n in [34, 33, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 9]:
        if time.time() > deadline - 10:
            break
        
        # The local search implementation above is incomplete/messy.
        # Let's use a simpler, robust approach for the final code.
        pass
    
    # Fallback: Return a known small valid coloring
    # n=9 coloring: (a+b+c)%3==0 -> Red (0), else Blue (1)
    # Check validity for n=9:
    # Red K4^3? 
    # Blue K5^3?
    # This is the seed. Score 1.0.
    
    # We need a better fallback or a working local search.
    
    # Let's implement a proper local search for n=34.
    
    def robust_local_search(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        quads = list(combinations(range(n), 4))
        num_quads = len(quads)
        quad_to_idx = {q: i for i, q in enumerate(quads)}
        
        quints = list(combinations(range(n), 5))
        num_quints = len(quints)
        quint_to_idx = {q: i for i, q in enumerate(quints)}
        
        # Precompute incidences
        tri_to_quads = [[] for _ in range(num_triples)]
        for q_idx, q in enumerate(quads):
            a,b,c,d = q
            for t in combinations(q, 3):
                tri_to_quads[triple_to_idx[t]].append(q_idx)
                
        tri_to_quints = [[] for _ in range(num_triples)]
        for q_idx, q in enumerate(quints):
            for t in combinations(q, 3):
                tri_to_quints[triple_to_idx[t]].append(q_idx)
        
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        red_counts = [0] * num_quads
        for i, c in enumerate(colors):
            if c == 0:
                for q_idx in tri_to_quads[i]:
                    red_counts[q_idx] += 1
                    
        blue_counts = [0] * num_quints
        for i, c in enumerate(colors):
            if c == 1:
                for q_idx in tri_to_quints[i]:
                    blue_counts[q_idx] += 1
                    
        red_violations = sum(1 for c in red_counts if c == 4)
        blue_violations = sum(1 for c in blue_counts if c == 10)
        
        if red_violations == 0 and blue_violations == 0:
            return colors
            
        # Local Search
        # We will try to reduce violations.
        # Strategy: Pick a random violation, flip a random triple in it, 
        # accept if total violations decrease.
        
        last_improve = time.time()
        
        while time.time() < deadline - 5:
            if red_violations == 0 and blue_violations == 0:
                return colors
                
            if time.time() - last_improve > 5:
                # Restart or give up on this n
                break
                
            # Pick a violation
            if red_violations > 0:
                # Find a random red violation
                # Optimization: maintain a list of violating quads?
                # For simplicity, sample.
                while True:
                    q_idx = random.randint(0, num_quads - 1)
                    if red_counts[q_idx] == 4:
                        break
                quad = quads[q_idx]
                # Pick a random triple from this quad
                a,b,c,d = quad
                t_idx = random.choice([
                    triple_to_idx[(a,b,c)],
                    triple_to_idx[(a,b,d)],
                    triple_to_idx[(a,c,d)],
                    triple_to_idx[(b,c,d)]
                ])
                # Flip Red (0) to Blue (1)
                colors[t_idx] = 1
                
                # Calculate delta
                delta_red = 0
                for q_i in tri_to_quads[t_idx]:
                    old = red_counts[q_i]
                    red_counts[q_i] = old - 1
                    if old == 4:
                        delta_red -= 1
                    elif red_counts[q_i] == 4: # Impossible, we decreased
                        delta_red += 1
                        
                delta_blue = 0
                for q_i in tri_to_quints[t_idx]:
                    old = blue_counts[q_i]
                    blue_counts[q_i] = old + 1
                    if old == 9:
                        delta_blue += 1
                    elif blue_counts[q_i] == 10: # Same
                        pass
                        
                if delta_red + delta_blue < 0:
                    red_violations += delta_red
                    blue_violations += delta_blue
                    last_improve = time.time()
                else:
                    # Revert
                    colors[t_idx] = 0
                    for q_i in tri_to_quads[t_idx]:
                        red_counts[q_i] += 1
                    for q_i in tri_to_quints[t_idx]:
                        blue_counts[q_i] -= 1
                        
            else: # blue_violations > 0
                while True:
                    q_idx = random.randint(0, num_quints - 1)
                    if blue_counts[q_idx] == 10:
                        break
                quint = quints[q_idx]
                # Pick a random triple from this quint
                t_candidates = list(combinations(quint, 3))
                t = random.choice(t_candidates)
                t_idx = triple_to_idx[t]
                # Flip Blue (1) to Red (0)
                colors[t_idx] = 0
                
                delta_red = 0
                for q_i in tri_to_quads[t_idx]:
                    old = red_counts[q_i]
                    red_counts[q_i] = old + 1
                    if old == 3:
                        delta_red += 1
                        
                delta_blue = 0
                for q_i in tri_to_quints[t_idx]:
                    old = blue_counts[q_i]
                    blue_counts[q_i] = old - 1
                    if old == 10:
                        delta_blue -= 1
                        
                if delta_red + delta_blue < 0:
                    red_violations += delta_red
                    blue_violations += delta_blue
                    last_improve = time.time()
                else:
                    # Revert
                    colors[t_idx] = 1
                    for q_i in tri_to_quads[t_idx]:
                        red_counts[q_i] -= 1
                    for q_i in tri_to_quints[t_idx]:
                        blue_counts[q_i] += 1
                        
        return None

    # Try to find a valid coloring for decreasing n
    for n in [34, 33, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 9]:
        if time.time() > deadline - 10:
            break
            
        # Try a few restarts for each n
        found = None
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            found = robust_local_search(n)
            if found:
                break
                
        if found:
            return {'n': n, 'colors': found}
            
    # Fallback to n=9
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
