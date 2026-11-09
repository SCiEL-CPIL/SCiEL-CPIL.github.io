"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _validate_fast(n, colors):
    """
    Check if a coloring of triples on n vertices is valid using NumPy.
    colors: list or array of 0/1 in lexicographic order of triples (a,b,c), a<b<c.
    Returns True if no red K4^(3) and no blue K5^(3).
    """
    n_triples = len(colors)
    expected = n * (n - 1) * (n - 2) // 6
    if n_triples != expected:
        return False
    
    colors = np.array(colors, dtype=np.uint8)
    
    # Build 3D array: color[a,b,c] for a<b<c
    color3d = np.zeros((n, n, n), dtype=np.uint8)
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                color3d[a, b, c] = colors[idx]
                idx += 1
    
    # Check for red K4^(3): any 4 vertices where all 4 triples are red
    # For each 4-set {a,b,c,d}, check if all 4 triples are color 0
    for a in range(n - 3):
        for b in range(a + 1, n - 2):
            for c in range(b + 1, n - 1):
                for d in range(c + 1, n):
                    if (color3d[a, b, c] == 0 and color3d[a, b, d] == 0 and
                        color3d[a, c, d] == 0 and color3d[b, c, d] == 0):
                        return False
    
    # Check for blue K5^(3): any 5 vertices where all 10 triples are blue
    # For each 5-set, check if all 10 triples are color 1
    for a in range(n - 4):
        for b in range(a + 1, n - 3):
            for c in range(b + 1, n - 2):
                for d in range(c + 1, n - 1):
                    for e in range(d + 1, n):
                        # 10 triples: (a,b,c),(a,b,d),(a,b,e),(a,c,d),(a,c,e),(a,d,e),(b,c,d),(b,c,e),(b,d,e),(c,d,e)
                        if (color3d[a, b, c] == 1 and color3d[a, b, d] == 1 and
                            color3d[a, b, e] == 1 and color3d[a, c, d] == 1 and
                            color3d[a, c, e] == 1 and color3d[a, d, e] == 1 and
                            color3d[b, c, d] == 1 and color3d[b, c, e] == 1 and
                            color3d[b, d, e] == 1 and color3d[c, d, e] == 1):
                            return False
    
    return True

def _incremental_search(n, initial_colors, deadline, max_iters=500000):
    """
    Local search with incremental violation counting.
    Tries to eliminate all violations by flipping individual triples.
    """
    colors = np.array(initial_colors, dtype=np.uint8).copy()
    n_triples = len(colors)
    
    # Precompute triple index mapping
    triple_to_idx = {}
    triples = []
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triple_to_idx[(a, b, c)] = idx
                triples.append((a, b, c))
                idx += 1
    
    # Build color3d for quick lookup
    color3d = np.zeros((n, n, n), dtype=np.uint8)
    for i, (a, b, c) in enumerate(triples):
        color3d[a, b, c] = colors[i]
    
    def count_violations():
        """Count total violations (red K4s + blue K5s)."""
        red_k4 = 0
        blue_k5 = 0
        for a in range(n - 3):
            for b in range(a + 1, n - 2):
                for c in range(b + 1, n - 1):
                    for d in range(c + 1, n):
                        if (color3d[a, b, c] == 0 and color3d[a, b, d] == 0 and
                            color3d[a, c, d] == 0 and color3d[b, c, d] == 0):
                            red_k4 += 1
        for a in range(n - 4):
            for b in range(a + 1, n - 3):
                for c in range(b + 1, n - 2):
                    for d in range(c + 1, n - 1):
                        for e in range(d + 1, n):
                            if (color3d[a, b, c] == 1 and color3d[a, b, d] == 1 and
                                color3d[a, b, e] == 1 and color3d[a, c, d] == 1 and
                                color3d[a, c, e] == 1 and color3d[a, d, e] == 1 and
                                color3d[b, c, d] == 1 and color3d[b, c, e] == 1 and
                                color3d[b, d, e] == 1 and color3d[c, d, e] == 1):
                                blue_k5 += 1
        return red_k4 + blue_k5
    
    def delta_red_k4(triple_idx, new_color):
        """
        Compute change in red K4 count if we set triple[triple_idx] to new_color.
        A red K4 is destroyed if the triple becomes blue (was red, all 4 were red).
        A red K4 is created if the triple becomes red (was blue, other 3 are red).
        """
        a, b, c = triples[triple_idx]
        old_color = colors[triple_idx]
        if old_color == new_color:
            return 0
        
        delta = 0
        # For each vertex d not in {a,b,c}, check if {a,b,c,d} forms a K4
        for d in range(n):
            if d == a or d == b or d == c:
                continue
            # The 4 triples in K4 {a,b,c,d}: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
            # Need to handle ordering: a<b<c<d or some permutation
            # Let's get the sorted 4-tuple
            verts = sorted([a, b, c, d])
            v0, v1, v2, v3 = verts
            
            # The 4 triples: (v0,v1,v2), (v0,v1,v3), (v0,v2,v3), (v1,v2,v3)
            t1 = (v0, v1, v2)
            t2 = (v0, v1, v3)
            t3 = (v0, v2, v3)
            t4 = (v1, v2, v3)
            
            # Get colors of the other 3 triples (not our triple)
            other_triples = [t for t in [t1, t2, t3, t4] if t != (a, b, c)]
            # Wait, (a,b,c) might not be in the sorted form. Let me re-think.
            # Actually, the triple (a,b,c) with a<b<c is one specific triple.
            # In the K4 {v0,v1,v2,v3}, the 4 triples are:
            # (v0,v1,v2), (v0,v1,v3), (v0,v2,v3), (v1,v2,v3)
            # Our triple (a,b,c) is one of these. Let's identify which.
            
            our_triple = (a, b, c)
            if our_triple == t1:
                others = [t2, t3, t4]
            elif our_triple == t2:
                others = [t1, t3, t4]
            elif our_triple == t3:
                others = [t1, t2, t4]
            elif our_triple == t4:
                others = [t1, t2, t3]
            else:
                continue  # Should not happen
            
            other_colors = [color3d[o[0], o[1], o[2]] for o in others]
            
            if new_color == 0:  # Making red
                # Creates a red K4 if all other 3 are red
                if all(col == 0 for col in other_colors):
                    delta += 1
            else:  # Making blue (new_color == 1)
                # Destroys a red K4 if it was red and all other 3 are red
                if old_color == 0 and all(col == 0 for col in other_colors):
                    delta -= 1
        
        return delta
    
    def delta_blue_k5(triple_idx, new_color):
        """
        Compute change in blue K5 count if we set triple[triple_idx] to new_color.
        A blue K5 is destroyed if the triple becomes red (was blue, all 10 were blue).
        A blue K5 is created if the triple becomes blue (was red, other 9 are blue).
        """
        a, b, c = triples[triple_idx]
        old_color = colors[triple_idx]
        if old_color == new_color:
            return 0
        
        delta = 0
        # For each 5-set containing {a,b,c}, check if it forms a blue K5
        # The 5-set is {a,b,c,x,y} where x,y are two other vertices
        # There are C(n-3, 2) such 5-sets
        remaining = [v for v in range(n) if v != a and v != b and v != c]
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                x = remaining[i]
                y = remaining[j]
                verts = sorted([a, b, c, x, y])
                v0, v1, v2, v3, v4 = verts
                
                # The 10 triples in K5
                all_triples = []
                for p in range(5):
                    for q in range(p + 1, 5):
                        for r in range(q + 1, 5):
                            all_triples.append((verts[p], verts[q], verts[r]))
                
                our_triple = (a, b, c)
                if our_triple not in all_triples:
                    continue
                
                others = [t for t in all_triples if t != our_triple]
                other_colors = [color3d[o[0], o[1], o[2]] for o in others]
                
                if new_color == 1:  # Making blue
                    # Creates a blue K5 if all other 9 are blue
                    if all(col == 1 for col in other_colors):
                        delta += 1
                else:  # Making red (new_color == 0)
                    # Destroys a blue K5 if it was blue and all other 9 are blue
                    if old_color == 1 and all(col == 1 for col in other_colors):
                        delta -= 1
        
        return delta
    
    # Initial violation count
    total_violations = count_violations()
    best_violations = total_violations
    best_colors = colors.copy()
    
    iter_count = 0
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline and iter_count < max_iters:
        iter_count += 1
        
        if total_violations == 0:
            break
        
        # Pick a random triple to flip
        idx = random.randint(0, n_triples - 1)
        new_color = 1 - colors[idx]
        
        # Compute delta in violations
        d_red = delta_red_k4(idx, new_color)
        d_blue = delta_blue_k5(idx, new_color)
        total_delta = d_red + d_blue
        
        # Accept if it improves or with some probability (simulated annealing style)
        if total_delta <= 0:
            colors[idx] = new_color
            color3d[triples[idx][0], triples[idx][1], triples[idx][2]] = new_color
            total_violations += total_delta
            no_improve_count = 0
        else:
            # Occasionally accept worsening moves to escape local minima
            if random.random() < 0.01:
                colors[idx] = new_color
                color3d[triples[idx][0], triples[idx][1], triples[idx][2]] = new_color
                total_violations += total_delta
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        if total_violations < best_violations:
            best_violations = total_violations
            best_colors = colors.copy()
            no_improve_count = 0
        
        if no_improve_count > max_no_improve:
            # Reset to best and try again with more aggressive randomization
            colors = best_colors.copy()
            for i, (a, b, c) in enumerate(triples):
                color3d[a, b, c] = colors[i]
            total_violations = best_violations
            no_improve_count = 0
            max_no_improve = max(1000, max_no_improve // 2)
    
    return best_colors, best_violations

def construct():
    """
    Construct a valid two-coloring of triples on n vertices with no red K4^(3)
    and no blue K5^(3), maximizing n.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_n = 0
    best_colors = None
    
    # Strategy: Try various algebraic initializations, then use local search
    # to eliminate violations.
    
    # First, try simple modular constructions directly
    for n in range(34, 8, -1):
        if time.time() > deadline - 30:
            break
        
        # Try sum mod m
        for m in range(2, n + 5):
            if time.time() > deadline - 30:
                break
            
            colors = []
            for a in range(n):
                for b in range(a + 1, n):
                    for c in range(b + 1, n):
                        if (a + b + c) % m == 0:
                            colors.append(0)
                        else:
                            colors.append(1)
            
            if _validate_fast(n, colors):
                best_n = n
                best_colors = colors[:]
                break
        
        if best_n == n:
            break
    
    if best_n > 0 and best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    
    # Now try local search from various initializations
    for target_n in range(34, 8, -1):
        if time.time() > deadline - 10:
            break
        
        # Try several initialization strategies
        inits = []
        
        # Sum mod m
        for m in [3, 5, 7, target_n, target_n + 1, target_n + 2]:
            if m < 2:
                continue
            init = [0 if (a + b + c) % m == 0 else 1 
                    for a in range(target_n) for b in range(a+1, target_n) for c in range(b+1, target_n)]
            inits.append(init)
        
        # Product-based
        for m in [3, 5, 7, target_n]:
            if m < 2:
                continue
            init = [0 if (a * b + b * c + c * a) % m == 0 else 1 
                    for a in range(target_n) for b in range(a+1, target_n) for c in range(b+1, target_n)]
            inits.append(init)
        
        # Random initialization
        init = [random.randint(0, 1) for _ in range(target_n * (target_n-1) * (target_n-2) // 6)]
        inits.append(init)
        
        # Run local search from each initialization
        search_deadline = min(deadline - 5, start_time + 25)
        for init in inits:
            if time.time() > search_deadline:
                break
            
            result_colors, violations = _incremental_search(target_n, init, search_deadline)
            
            if violations == 0:
                best_n = target_n
                best_colors = result_colors.tolist()
                break
        
        if best_n == target_n:
            break
    
    if best_n > 0 and best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    
    # Fallback: return a small valid coloring
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 
              for a in range(n) for b in range(a+1, n) for c in range(b+1, n)]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
