# EVOLVE-BLOCK-START

import time
import numpy as np
from itertools import combinations

def _build_constraint_indices(n):
    """Build index arrays for quad constraints (4 triples each) and quint constraints (10 triples each)."""
    triples = []
    triple_to_idx = {}
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triples.append((a, b, c))
                triple_to_idx[(a, b, c)] = idx
                idx += 1
                
    num_triples = idx
    
    # Quad constraints: each 4-set has 4 triples
    quad_constraints = []
    for quad in combinations(range(n), 4):
        ts = [
            (quad[0], quad[1], quad[2]),
            (quad[0], quad[1], quad[3]),
            (quad[0], quad[2], quad[3]),
            (quad[1], quad[2], quad[3]),
        ]
        quad_constraints.append([triple_to_idx[t] for t in ts])
    
    # Quint constraints: each 5-set has 10 triples
    quint_constraints = []
    for quint in combinations(range(n), 5):
        ts = []
        for i in range(5):
            for j in range(i + 1, 5):
                for k in range(j + 1, 5):
                    ts.append((quint[i], quint[j], quint[k]))
        quint_constraints.append([triple_to_idx[t] for t in ts])
    
    quad_idx = np.array(quad_constraints, dtype=np.int32)
    quint_idx = np.array(quint_constraints, dtype=np.int32)
    
    return quad_idx, quint_idx

def _fast_local_search(n, deadline, seed=42):
    """
    Fast local search using incremental violation counts.
    Returns colors list if valid, else None.
    """
    rng = np.random.RandomState(seed)
    num_triples = n * (n - 1) * (n - 2) // 6
    
    if num_triples == 0:
        return []
        
    quad_idx, quint_idx = _build_constraint_indices(n)
    
    # Precompute which quads/quints each triple belongs to
    triple_to_quads = [[] for _ in range(num_triples)]
    for qi in range(len(quad_idx)):
        for ti in quad_idx[qi]:
            triple_to_quads[ti].append(qi)
            
    triple_to_quints = [[] for _ in range(num_triples)]
    for ui in range(len(quint_idx)):
        for ti in quint_idx[ui]:
            triple_to_quints[ti].append(ui)
            
    # Initialize colors randomly
    colors = (rng.random(size=num_triples) < 0.5).astype(np.int8)
    
    # Initialize counts
    quad_red_counts = np.zeros(len(quad_idx), dtype=np.int32)
    quint_blue_counts = np.zeros(len(quint_idx), dtype=np.int32)
    
    for qi in range(len(quad_idx)):
        for ti in quad_idx[qi]:
            if colors[ti] == 0:
                quad_red_counts[qi] += 1
                
    for ui in range(len(quint_idx)):
        for ti in quint_idx[ui]:
            if colors[ti] == 1:
                quint_blue_counts[ui] += 1
                
    cur_viol = np.sum(quad_red_counts == 4) + np.sum(quint_blue_counts == 10)
    
    best_colors = colors.copy()
    best_viol = cur_viol
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() < deadline:
        if cur_viol == 0:
            break
            
        # Pick random triple
        t = rng.randint(num_triples)
        old_c = colors[t]
        new_c = 1 - old_c
        
        # Calculate delta
        delta = 0
        
        # Update quad counts and check violations
        old_quad_viol = 0
        new_quad_viol = 0
        for qi in triple_to_quads[t]:
            if old_c == 0:
                # Red -> Blue: red count decreases
                quad_red_counts[qi] -= 1
            else:
                # Blue -> Red: red count increases
                quad_red_counts[qi] += 1
                
            if quad_red_counts[qi] == 4:
                new_quad_viol += 1
                
            # Revert to calculate old_viol? No, we need old_viol before update.
            # Let's restructure: store old state, calculate new, revert if needed?
            # Or calculate delta directly.
            
        # This approach is messy with in-place updates. Let's do it cleanly.
        
        # --- Clean Delta Calculation ---
        delta = 0
        
        # Quads
        for qi in triple_to_quads[t]:
            old_count = quad_red_counts[qi]
            old_viol = 1 if old_count == 4 else 0
            
            if old_c == 0:
                new_count = old_count - 1
            else:
                new_count = old_count + 1
                
            new_viol = 1 if new_count == 4 else 0
            
            if old_viol != new_viol:
                delta += new_viol - old_viol
                
            # We will update the counts after we decide to accept
            # But we need to know the counts for the next step.
            # So we must update them if we accept.
            # Let's store the changes in a temp list or just update and revert.
            # Given the small size of triple_to_quads[t], updating and reverting is fast.
            
            # Let's just update, calculate total, and revert if rejected.
        
        # Let's try a simpler approach:
        # 1. Save old counts for affected constraints.
        # 2. Update counts.
        # 3. Calculate new total violations for affected constraints.
        # 4. Compare with old total violations for affected constraints.
        # 5. If accept, keep updates. If reject, revert.
        
        affected_quads = triple_to_quads[t]
        affected_quints = triple_to_quints[t]
        
        old_q_viol = 0
        old_u_viol = 0
        for qi in affected_quads:
            if quad_red_counts[qi] == 4:
                old_q_viol += 1
        for ui in affected_quints:
            if quint_blue_counts[ui] == 10:
                old_u_viol += 1
        
        # Apply flip
        for qi in affected_quads:
            if old_c == 0:
                quad_red_counts[qi] -= 1
            else:
                quad_red_counts[qi] += 1
                
        for ui in affected_quints:
            if old_c == 1:
                quint_blue_counts[ui] -= 1
            else:
                quint_blue_counts[ui] += 1
                
        new_q_viol = 0
        new_u_viol = 0
        for qi in affected_quads:
            if quad_red_counts[qi] == 4:
                new_q_viol += 1
        for ui in affected_quints:
            if quint_blue_counts[ui] == 10:
                new_u_viol += 1
                
        delta = (new_q_viol - old_q_viol) + (new_u_viol - old_u_viol)
        
        accept = False
        if delta < 0:
            accept = True
        else:
            if delta == 0:
                accept = True # Neutral moves help escape local minima
            else:
                if rng.random() < np.exp(-delta / max(T, 1e-10)):
                    accept = True
        
        if accept:
            colors[t] = new_c
            cur_viol += delta
            if cur_viol < best_viol:
                best_viol = cur_viol
                best_colors = colors.copy()
        else:
            # Revert counts
            for qi in affected_quads:
                if old_c == 0:
                    quad_red_counts[qi] += 1
                else:
                    quad_red_counts[qi] -= 1
                    
            for ui in affected_quints:
                if old_c == 1:
                    quint_blue_counts[ui] += 1
                else:
                    quint_blue_counts[ui] -= 1
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    if best_viol == 0:
        return best_colors.tolist()
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try to find a valid coloring for n starting from 20 down to 10
    # 20 is a reasonable target that might be solvable quickly.
    # If we can't find 20, we try 19, 18, etc.
    
    best_n = 10
    best_colors = None
    
    # Trivial valid coloring for n=10 (or smaller) to guarantee return
    # For n=10, random search usually finds it very fast.
    # Let's just run the search for a few n values.
    
    for n in range(20, 9, -1):
        if time.time() > deadline - 5:
            break
            
        # Allocate time based on n
        if n >= 18:
            t_limit = 5.0
        elif n >= 15:
            t_limit = 3.0
        else:
            t_limit = 1.0
            
        sub_deadline = time.time() + t_limit
        
        # Try a few seeds
        found = False
        for seed in range(5):
            if time.time() > sub_deadline:
                break
            colors = _fast_local_search(n, sub_deadline, seed=seed)
            if colors is not None:
                best_n = n
                best_colors = colors
                found = True
                break
                
        if found:
            # Found a valid coloring for this n
            # We could try to go higher, but we are iterating downwards.
            # So the first one we find is the highest n we found in this pass.
            # However, we started from 20. If 20 fails, we try 19.
            # So if we find one at 18, we return it?
            # The loop goes 20, 19, 18...
            # If 20 fails, 19 fails, 18 succeeds -> return 18.
            # This is correct.
            return {'n': best_n, 'colors': best_colors}
            
    # If we didn't find anything for n >= 10, return a trivial one for n=10
    # Actually, the loop covers n=10. If it fails, we need a fallback.
    # Let's ensure n=10 is handled.
    if best_colors is None:
        # Fallback: n=10, random init usually works, but if not, use a simple pattern
        # For n=10, number of triples is 120.
        # Let's just do a quick random init and check.
        n = 10
        triples = list(combinations(range(n), 3))
        # Use a simple heuristic: color 0 if (a+b+c)%3==0 else 1
        colors = [0 if (a+b+c)%3==0 else 1 for a,b,c in triples]
        # Verify
        quad_idx, quint_idx = _build_constraint_indices(n)
        # Quick check
        q_viol = 0
        for qi in range(len(quad_idx)):
            if all(colors[t] == 0 for t in quad_idx[qi]):
                q_viol += 1
        u_viol = 0
        for ui in range(len(quint_idx)):
            if all(colors[t] == 1 for t in quint_idx[ui]):
                u_viol += 1
        if q_viol == 0 and u_viol == 0:
            best_n = n
            best_colors = colors
        else:
            # Last resort: n=5
            n = 5
            triples = list(combinations(range(n), 3))
            colors = [0]*len(triples) # All red. No blue K5 (size 5) possible if all red? 
            # Wait, if all red, then any 4 vertices form a red K4. 
            # So all-red is INVALID for n>=4.
            # We need a valid coloring for n=5.
            # Let's just return n=1 with empty colors? No, n>=1.
            # For n=1, no triples. Valid.
            # For n=2, no triples. Valid.
            # For n=3, 1 triple. Valid.
            # For n=4, 4 triples. We need no red K4. So at least one blue.
            # If we set one blue, no red K4. No blue K5 (needs 5 vertices).
            # So for n=4, colors=[1,0,0,0] is valid.
            # Let's just return n=4.
            n = 4
            triples = list(combinations(range(n), 3))
            # (0,1,2), (0,1,3), (0,2,3), (1,2,3)
            # Set (0,1,2) to blue (1), others red (0).
            colors = [1, 0, 0, 0]
            best_n = n
            best_colors = colors

    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
