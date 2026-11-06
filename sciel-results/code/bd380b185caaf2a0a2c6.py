import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 115
    
    def is_valid_coloring(n, colors):
        """Check if the coloring of K_n has no monochromatic K2,2,2."""
        # This is a slow check, used only for final verification
        edge_to_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_to_idx[(i, j)] = idx
                idx += 1
        
        partitions = [
            [(0,1),(2,3),(4,5)], [(0,1),(2,4),(3,5)], [(0,1),(2,5),(3,4)],
            [(0,2),(1,3),(4,5)], [(0,2),(1,4),(3,5)], [(0,2),(1,5),(3,4)],
            [(0,3),(1,2),(4,5)], [(0,3),(1,4),(2,5)], [(0,3),(1,5),(2,4)],
            [(0,4),(1,2),(3,5)], [(0,4),(1,3),(2,5)], [(0,4),(1,5),(2,3)],
            [(0,5),(1,2),(3,4)], [(0,5),(1,3),(2,4)], [(0,5),(1,4),(2,3)]
        ]
        
        colors_arr = np.array(colors, dtype=np.int8)
        
        for vertices in combinations(range(n), 6):
            v = vertices
            for part in partitions:
                p0 = [v[part[0][0]], v[part[0][1]]]
                p1 = [v[part[1][0]], v[part[1][1]]]
                p2 = [v[part[2][0]], v[part[2][1]]]
                
                c0 = 0
                c1 = 0
                for u in p0:
                    for w in p1:
                        idx_e = edge_to_idx[(min(u,w), max(u,w))]
                        if colors_arr[idx_e] == 0: c0 += 1
                        else: c1 += 1
                for u in p0:
                    for w in p2:
                        idx_e = edge_to_idx[(min(u,w), max(u,w))]
                        if colors_arr[idx_e] == 0: c0 += 1
                        else: c1 += 1
                for u in p1:
                    for w in p2:
                        idx_e = edge_to_idx[(min(u,w), max(u,w))]
                        if colors_arr[idx_e] == 0: c0 += 1
                        else: c1 += 1
                
                if c0 == 12 or c1 == 12:
                    return False
        return True

    def find_n29():
        """Find a valid coloring for n=29 using local search."""
        n = 29
        num_edges = n * (n - 1) // 2
        
        # Precompute instances for n=29
        edge_to_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_to_idx[(i, j)] = idx
                idx += 1
                
        partitions = [
            [(0,1),(2,3),(4,5)], [(0,1),(2,4),(3,5)], [(0,1),(2,5),(3,4)],
            [(0,2),(1,3),(4,5)], [(0,2),(1,4),(3,5)], [(0,2),(1,5),(3,4)],
            [(0,3),(1,2),(4,5)], [(0,3),(1,4),(2,5)], [(0,3),(1,5),(2,4)],
            [(0,4),(1,2),(3,5)], [(0,4),(1,3),(2,5)], [(0,4),(1,5),(2,3)],
            [(0,5),(1,2),(3,4)], [(0,5),(1,3),(2,4)], [(0,5),(1,4),(2,3)]
        ]
        
        instances = []
        edge_to_instances = [[] for _ in range(num_edges)]
        
        for vertices in combinations(range(n), 6):
            v = vertices
            for part in partitions:
                p0 = [v[part[0][0]], v[part[0][1]]]
                p1 = [v[part[1][0]], v[part[1][1]]]
                p2 = [v[part[2][0]], v[part[2][1]]]
                
                edge_indices = []
                for u in p0:
                    for w in p1:
                        edge_indices.append(edge_to_idx[(min(u,w), max(u,w))])
                for u in p0:
                    for w in p2:
                        edge_indices.append(edge_to_idx[(min(u,w), max(u,w))])
                for u in p1:
                    for w in p2:
                        edge_indices.append(edge_to_idx[(min(u,w), max(u,w))])
                        
                inst_id = len(instances)
                instances.append(edge_indices)
                for e in edge_indices:
                    edge_to_instances[e].append(inst_id)
                    
        inst_arrays = [np.array(inst, dtype=np.int32) for inst in instances]
        
        # Local Search
        best_colors = None
        best_viol = float('inf')
        
        for restart in range(20):
            if time.time() > deadline - 30:
                break
                
            colors = np.random.randint(0, 2, size=num_edges, dtype=np.int8)
            count_0 = np.zeros(len(instances), dtype=np.int8)
            for i in range(len(instances)):
                count_0[i] = np.sum(colors[inst_arrays[i]] == 0)
                
            is_viol = (count_0 == 12) | (count_0 == 0)
            viol_set = set(np.where(is_viol)[0].tolist())
            
            if len(viol_set) == 0:
                return colors.tolist()
                
            if len(viol_set) < best_viol:
                best_viol = len(viol_set)
                best_colors = colors.copy()
                
            # Greedy improvement
            for _ in range(5000):
                if time.time() > deadline - 30:
                    break
                if not viol_set:
                    return colors.tolist()
                    
                # Pick random violating instance
                inst_id = random.choice(list(viol_set))
                inst_edges = inst_arrays[inst_id]
                
                # Try flipping each edge in this instance
                best_score = 0
                best_edge = None
                for e_idx in inst_edges:
                    old_color = colors[e_idx]
                    delta = -1 if old_color == 0 else 1
                    score = 0
                    for inst_id_2 in edge_to_instances[e_idx]:
                        c0 = count_0[inst_id_2]
                        is_viol_before = (c0 == 12) or (c0 == 0)
                        c0_after = c0 + delta
                        is_viol_after = (c0_after == 12) or (c0_after == 0)
                        if is_viol_before and not is_viol_after:
                            score += 1
                        elif not is_viol_before and is_viol_after:
                            score -= 1
                    if score > best_score:
                        best_score = score
                        best_edge = e_idx
                        
                if best_edge is None:
                    # Random flip to escape
                    e_idx = random.choice(inst_edges.tolist())
                else:
                    e_idx = best_edge
                    
                old_color = colors[e_idx]
                colors[e_idx] = 1 - old_color
                delta = -1 if old_color == 0 else 1
                
                new_viol_set = set()
                for inst_id_2 in edge_to_instances[e_idx]:
                    count_0[inst_id_2] += delta
                    c0 = count_0[inst_id_2]
                    if c0 == 12 or c0 == 0:
                        new_viol_set.add(inst_id_2)
                        
                viol_set -= set(edge_to_instances[e_idx])
                viol_set |= new_viol_set
                
                if len(viol_set) < best_viol:
                    best_viol = len(viol_set)
                    best_colors = colors.copy()
                    
                if len(viol_set) == 0:
                    return colors.tolist()
                    
        if best_colors is not None:
            return best_colors.tolist()
        return None

    def try_extend_to_n30(colors_29):
        """Try to extend a valid n=29 coloring to n=30."""
        n_old = 29
        n_new = 30
        num_new_edges = n_new - 1 # Edges incident to vertex 29
        
        # Precompute constraints involving vertex 29
        # Vertex 29 is index 29.
        # Edges (29, j) for j in 0..28 are indices 406..434 in the full n=30 graph.
        # But we treat them as local variables 0..28.
        
        # We need to check K2,2,2 subgraphs that include vertex 29.
        # The other 5 vertices must be chosen from 0..28.
        # There are C(29, 5) = 118755 ways to choose the other 5 vertices.
        # For each set of 5 vertices, there are 15 partitions.
        # Total constraints to check: 118755 * 15 = 1,781,325.
        
        # For each constraint, we determine:
        # 1. Which variable edges (0..28) are involved.
        # 2. The colors of the fixed edges (from colors_29).
        # 3. Whether the constraint is already satisfied (mixed fixed colors).
        # 4. If not, which monochromatic color it could potentially be (0 or 1).
        
        # Precompute edge indices for n=29
        edge_to_idx_29 = {}
        idx = 0
        for i in range(n_old):
            for j in range(i + 1, n_old):
                edge_to_idx_29[(i, j)] = idx
                idx += 1
                
        partitions = [
            [(0,1),(2,3),(4,5)], [(0,1),(2,4),(3,5)], [(0,1),(2,5),(3,4)],
            [(0,2),(1,3),(4,5)], [(0,2),(1,4),(3,5)], [(0,2),(1,5),(3,4)],
            [(0,3),(1,2),(4,5)], [(0,3),(1,4),(2,5)], [(0,3),(1,5),(2,4)],
            [(0,4),(1,2),(3,5)], [(0,4),(1,3),(2,5)], [(0,4),(1,5),(2,3)],
            [(0,5),(1,2),(3,4)], [(0,5),(1,3),(2,4)], [(0,5),(1,4),(2,3)]
        ]
        
        # List of (var_edges, fixed_colors, possible_colors)
        # possible_colors: 0 (none), 1 (can be 0), 2 (can be 1), 3 (can be 0 or 1)
        constraints = []
        
        for vertices in combinations(range(n_old), 5):
            v = list(vertices)
            # Add vertex 29 to the set
            all_v = v + [29]
            
            for part in partitions:
                p0 = [all_v[part[0][0]], all_v[part[0][1]]]
                p1 = [all_v[part[1][0]], all_v[part[1][1]]]
                p2 = [all_v[part[2][0]], all_v[part[2][1]]]
                
                var_edges = set()
                fixed_colors = []
                
                for u in p0:
                    for w in p1:
                        if u == 29 or w == 29:
                            # Variable edge
                            other = w if u == 29 else u
                            var_edges.add(other)
                        else:
                            # Fixed edge
                            c = colors_29[edge_to_idx_29[(min(u,w), max(u,w))]]
                            fixed_colors.append(c)
                            
                for u in p0:
                    for w in p2:
                        if u == 29 or w == 29:
                            other = w if u == 29 else u
                            var_edges.add(other)
                        else:
                            c = colors_29[edge_to_idx_29[(min(u,w), max(u,w))]]
                            fixed_colors.append(c)
                            
                for u in p1:
                    for w in p2:
                        if u == 29 or w == 29:
                            other = w if u == 29 else u
                            var_edges.add(other)
                        else:
                            c = colors_29[edge_to_idx_29[(min(u,w), max(u,w))]]
                            fixed_colors.append(c)
                            
                if not fixed_colors:
                    continue # Should not happen
                
                has_0 = 0 in fixed_colors
                has_1 = 1 in fixed_colors
                
                if has_0 and has_1:
                    continue # Already satisfied
                
                if has_0:
                    possible = 1 # Can only be 0
                elif has_1:
                    possible = 2 # Can only be 1
                else:
                    possible = 3 # Can be 0 or 1 (if no fixed edges? Impossible)
                    
                constraints.append((list(var_edges), possible))
                
        if time.time() > deadline - 20:
            return None
            
        # Local search on the 29 variable edges
        best_colors_new = None
        best_viol = float('inf')
        
        for restart in range(50):
            if time.time() > deadline - 10:
                break
                
            var_colors = np.random.randint(0, 2, size=num_new_edges, dtype=np.int8)
            
            # Count violations
            # For each constraint, check if it becomes monochromatic.
            # Violation if:
            # possible == 1 and all var_edges are 0
            # possible == 2 and all var_edges are 1
            # possible == 3 and (all var_edges are 0 or all var_edges are 1)
            
            def count_violations(vc):
                viol = 0
                for var_edges, possible in constraints:
                    if not var_edges:
                        # All fixed. If possible is 1 or 2, it's a violation?
                        # No, if all fixed and mixed, we skipped.
                        # If all fixed and same color, it's a violation.
                        # But we skipped if mixed.
                        # If all fixed and 0, possible=1. Violation.
                        # If all fixed and 1, possible=2. Violation.
                        # But we established that if fixed colors are mixed, we skip.
                        # So if we are here, fixed colors are uniform.
                        # So it is a violation.
                        viol += 1
                        continue
                        
                    # Check if all var_edges have the "dangerous" color
                    # Dangerous color for possible=1 is 0.
                    # Dangerous color for possible=2 is 1.
                    # Dangerous color for possible=3 is 0 or 1.
                    
                    # Optimization: precompute sum of var_colors for each constraint?
                    # No, var_edges are different for each constraint.
                    
                    # Simple check
                    target_0 = True
                    target_1 = True
                    for e in var_edges:
                        if vc[e] == 0:
                            target_1 = False
                        else:
                            target_0 = False
                        if not target_0 and not target_1:
                            break
                            
                    if possible == 1 and target_0:
                        viol += 1
                    elif possible == 2 and target_1:
                        viol += 1
                    elif possible == 3 and (target_0 or target_1):
                        viol += 1
                            
                return viol
                
            viol = count_violations(var_colors)
            
            if viol == 0:
                # Combine and return
                full_colors = colors_29 + var_colors.tolist()
                return full_colors
                
            if viol < best_viol:
                best_viol = viol
                best_colors_new = var_colors.copy()
                
            # Local search step
            for _ in range(2000):
                if time.time() > deadline - 5:
                    break
                if viol == 0:
                    full_colors = colors_29 + var_colors.tolist()
                    return full_colors
                    
                # Pick a random variable edge
                e_idx = random.randint(0, num_new_edges - 1)
                old_color = var_colors[e_idx]
                var_colors[e_idx] = 1 - old_color
                
                new_viol = count_violations(var_colors)
                
                if new_viol <= viol:
                    viol = new_viol
                    if viol < best_viol:
                        best_viol = viol
                        best_colors_new = var_colors.copy()
                else:
                    # Revert
                    var_colors[e_idx] = old_color
                    # Accept with small probability? No, keep greedy for now.
                    
        return None

    # Main logic
    best_result = {'n': 17, 'colors': [0] * (17 * 16 // 2)} # Fallback
    
    # 1. Find n=29
    colors_29 = find_n29()
    if colors_29 is not None:
        best_result = {'n': 29, 'colors': colors_29}
        
        # 2. Try to extend to n=30
        if time.time() < deadline - 15:
            colors_30 = try_extend_to_n30(colors_29)
            if colors_30 is not None:
                # Verify
                if is_valid_coloring(30, colors_30):
                    best_result = {'n': 30, 'colors': colors_30}
                    
    return best_result

