"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_n = 0
    best_colors = None
    
    # Try increasing n
    for n in range(5, 35):
        if time.time() > deadline - 10:
            break
        
        num_triples = n * (n-1) * (n-2) // 6
        triple_list = list(combinations(range(n), 3))
        
        # Build index map
        idx_map = {}
        for i, (a, b, c) in enumerate(triple_list):
            idx_map[(a, b, c)] = i
        
        # Precompute: for each 4-set, which 4 triples it contains
        # For each 5-set, which 10 triples it contains
        # Use numpy for fast violation counting
        
        # Precompute 4-set triple indices
        quad_triples = []  # list of arrays of 4 indices each
        for quad in combinations(range(n), 4):
            i, j, k, l = quad
            idxs = [idx_map[(i,j,k)], idx_map[(i,j,l)], idx_map[(i,k,l)], idx_map[(j,k,l)]]
            quad_triples.append(idxs)
        quad_triples = np.array(quad_triples, dtype=np.int32)
        num_quads = len(quad_triples)
        
        # Precompute 5-set triple indices
        quint_triples = []  # list of arrays of 10 indices each
        for quint in combinations(range(n), 5):
            i, j, k, l, m = quint
            idxs = [idx_map[(i,j,k)], idx_map[(i,j,l)], idx_map[(i,j,m)],
                    idx_map[(i,k,l)], idx_map[(i,k,m)], idx_map[(i,l,m)],
                    idx_map[(j,k,l)], idx_map[(j,k,m)], idx_map[(j,l,m)],
                    idx_map[(k,l,m)]]
            quint_triples.append(idxs)
        quint_triples = np.array(quint_triples, dtype=np.int32)
        num_quints = len(quint_triples)
        
        # Precompute for each triple, which quads and quints it belongs to
        # triple_to_quads[t] = list of quad indices containing triple t
        # triple_to_quints[t] = list of quint indices containing triple t
        triple_to_quads = [[] for _ in range(num_triples)]
        for qi, qidxs in enumerate(quad_triples):
            for t_idx in qidxs:
                triple_to_quads[t_idx].append(qi)
        triple_to_quads = [np.array(x, dtype=np.int32) if x else np.array([], dtype=np.int32) for x in triple_to_quads]
        
        triple_to_quints = [[] for _ in range(num_triples)]
        for qi, qidxs in enumerate(quint_triples):
            for t_idx in qidxs:
                triple_to_quints[t_idx].append(qi)
        triple_to_quints = [np.array(x, dtype=np.int32) if x else np.array([], dtype=np.int32) for x in triple_to_quints]
        
        # Try multiple initializations
        found_valid = False
        max_restarts = 300
        restart = 0
        
        while not found_valid and restart < max_restarts and time.time() < deadline - 5:
            restart += 1
            
            # Various initialization strategies
            if restart == 1:
                colors = np.random.randint(0, 2, size=num_triples)
            elif restart == 2:
                # Based on vertex sum mod 3
                colors = np.array([0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triple_list], dtype=np.int8)
            elif restart == 3:
                # Based on vertex sum mod 4
                colors = np.array([0 if (a + b + c) % 4 < 2 else 1 for a, b, c in triple_list], dtype=np.int8)
            elif restart == 4:
                # Based on max vertex mod 3
                colors = np.array([0 if max(a, b, c) % 3 == 0 else 1 for a, b, c in triple_list], dtype=np.int8)
            elif restart == 5:
                # Based on min vertex mod 2 and sum mod 3
                colors = np.array([0 if (min(a,b,c) % 2 == 0 and (a+b+c) % 3 == 0) or (min(a,b,c) % 2 == 1 and (a+b+c) % 3 != 0) else 1 for a, b, c in triple_list], dtype=np.int8)
            else:
                # Random with slight bias
                bias = 0.45 + 0.1 * np.random.random()
                colors = (np.random.random(num_triples) < bias).astype(np.int8)
            
            # Fast violation count using numpy
            # Red K4: quad has all 4 triples red (color 0)
            # Blue K5: quint has all 10 triples blue (color 1)
            
            def count_violations_fast(col):
                # col is int8 array of 0s and 1s
                # Red violations: quads where all 4 triples are 0
                if num_quads > 0:
                    quad_colors = col[quad_triples]  # shape (num_quads, 4)
                    red_viol = np.all(quad_colors == 0, axis=1).sum()
                else:
                    red_viol = 0
                # Blue violations: quints where all 10 triples are 1
                if num_quints > 0:
                    quint_colors = col[quint_triples]  # shape (num_quints, 10)
                    blue_viol = np.all(quint_colors == 1, axis=1).sum()
                else:
                    blue_viol = 0
                return int(red_viol + blue_viol)
            
            total_viol = count_violations_fast(colors)
            
            if total_viol == 0:
                found_valid = True
                best_n = n
                best_colors = colors.tolist()
                break
            
            # Local search with incremental updates
            # Track red count per quad and blue count per quint
            if num_quads > 0:
                red_quad_counts = np.sum(quad_triples and (colors[quad_triples] == 0), axis=1).astype(np.int32)
            else:
                red_quad_counts = np.array([], dtype=np.int32)
            
            if num_quints > 0:
                blue_quint_counts = np.sum(quint_triples and (colors[quint_triples] == 1), axis=1).astype(np.int32)
            else:
                blue_quint_counts = np.array([], dtype=np.int32)
            
            red_viol_count = int(np.sum(red_quad_counts == 4))
            blue_viol_count = int(np.sum(blue_quint_counts == 10))
            total_viol = red_viol_count + blue_viol_count
            
            max_iterations = 200000
            iter_count = 0
            improved = True
            
            while improved and iter_count < max_iterations and time.time() < deadline - 3:
                improved = False
                iter_count += 1
                
                # Pick a triple to flip - prioritize those in violating structures
                # Find a violating quad or quint and pick a triple from it
                candidates = []
                
                if red_viol_count > 0 and (blue_viol_count == 0 or np.random.random() < 0.5):
                    # Pick a violating quad
                    viol_quads = np.where(red_quad_counts == 4)[0]
                    if len(viol_quads) > 0:
                        qidx = viol_quads[np.random.randint(len(viol_quads))]
                        t_indices = quad_triples[qidx]
                        # Pick a random triple from this quad to flip
                        t_idx = int(t_indices[np.random.randint(4)])
                        candidates.append(t_idx)
                
                if blue_viol_count > 0:
                    viol_quints = np.where(blue_quint_counts == 10)[0]
                    if len(viol_quints) > 0:
                        qidx = viol_quints[np.random.randint(len(viol_quints))]
                        t_indices = quint_triples[qidx]
                        t_idx = int(t_indices[np.random.randint(10)])
                        candidates.append(t_idx)
                
                if not candidates:
                    # No violations - shouldn't happen if total_viol > 0
                    # Pick random
                    t_idx = np.random.randint(num_triples)
                    candidates.append(int(t_idx))
                
                t_idx = candidates[0]
                old_color = int(colors[t_idx])
                new_color = 1 - old_color
                
                # Compute delta in violations
                delta_red = 0
                delta_blue = 0
                
                # Update red quad counts
                if old_color == 0:  # was red, becoming blue
                    for qi in triple_to_quads[t_idx]:
                        qi = int(qi)
                        if red_quad_counts[qi] == 4:
                            delta_red -= 1  # was violation, now 3
                        red_quad_counts[qi] -= 1
                else:  # was blue, becoming red
                    for qi in triple_to_quads[t_idx]:
                        qi = int(qi)
                        if red_quad_counts[qi] == 3:
                            delta_red += 1  # becomes violation
                        red_quad_counts[qi] += 1
                
                # Update blue quint counts
                if old_color == 1:  # was blue, becoming red
                    for qi in triple_to_quints[t_idx]:
                        qi = int(qi)
                        if blue_quint_counts[qi] == 10:
                            delta_blue -= 1
                        blue_quint_counts[qi] -= 1
                else:  # was red, becoming blue
                    for qi in triple_to_quints[t_idx]:
                        qi = int(qi)
                        if blue_quint_counts[qi] == 9:
                            delta_blue += 1
                        blue_quint_counts[qi] += 1
                
                total_delta = delta_red + delta_blue
                
                if total_delta < 0:
                    colors[t_idx] = new_color
                    red_viol_count += delta_red
                    blue_viol_count += delta_blue
                    total_viol += total_delta
                    improved = True
                else:
                    # Revert
                    if old_color == 0:
                        for qi in triple_to_quads[t_idx]:
                            red_quad_counts[int(qi)] += 1
                    else:
                        for qi in triple_to_quads[t_idx]:
                            red_quad_counts[int(qi)] -= 1
                    if old_color == 1:
                        for qi in triple_to_quints[t_idx]:
                            blue_quint_counts[int(qi)] += 1
                    else:
                        for qi in triple_to_quints[t_idx]:
                            blue_quint_counts[int(qi)] -= 1
                
                # If stuck, try random flips
                if not improved and iter_count % 100 == 0:
                    for _ in range(50):
                        if time.time() > deadline - 3:
                            break
                        t_idx2 = np.random.randint(num_triples)
                        old_c = int(colors[t_idx2])
                        new_c = 1 - old_c
                        
                        dr = 0
                        db = 0
                        
                        if old_c == 0:
                            for qi in triple_to_quads[t_idx2]:
                                qi = int(qi)
                                if red_quad_counts[qi] == 4:
                                    dr -= 1
                                red_quad_counts[qi] -= 1
                        else:
                            for qi in triple_to_quads[t_idx2]:
                                qi = int(qi)
                                if red_quad_counts[qi] == 3:
                                    dr += 1
                                red_quad_counts[qi] += 1
                        
                        if old_c == 1:
                            for qi in triple_to_quints[t_idx2]:
                                qi = int(qi)
                                if blue_quint_counts[qi] == 10:
                                    db -= 1
                                blue_quint_counts[qi] -= 1
                        else:
                            for qi in triple_to_quints[t_idx2]:
                                qi = int(qi)
                                if blue_quint_counts[qi] == 9:
                                    db += 1
                                blue_quint_counts[qi] += 1
                        
                        if dr + db < 0:
                            colors[t_idx2] = new_c
                            red_viol_count += dr
                            blue_viol_count += db
                            total_viol += dr + db
                            improved = True
                            break
                        else:
                            # Revert
                            if old_c == 0:
                                for qi in triple_to_quads[t_idx2]:
                                    red_quad_counts[int(qi)] += 1
                            else:
                                for qi in triple_to_quads[t_idx2]:
                                    red_quad_counts[int(qi)] -= 1
                            if old_c == 1:
                                for qi in triple_to_quints[t_idx2]:
                                    blue_quint_counts[int(qi)] += 1
                            else:
                                for qi in triple_to_quints[t_idx2]:
                                    blue_quint_counts[int(qi)] -= 1
            
            # Final check
            total_viol = count_violations_fast(colors)
            if total_viol == 0:
                found_valid = True
                best_n = n
                best_colors = colors.tolist()
                break
        
        if found_valid:
            # Try to increase n
            continue
        else:
            # Keep best so far
            if n > best_n:
                best_n = n
                # Use the coloring with fewest violations as fallback
                best_colors = colors.tolist()
    
    if best_colors is None:
        # Fallback
        n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        best_n = n
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
