import random
import time
import numpy as np

def construct():
    N = 128
    deadline = time.time() + 85.0  # 5s margin for safety
    
    # Precompute all 4-APs (0-indexed)
    aps = []
    ap_indices = [[] for _ in range(N)]
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            ap_id = len(aps)
            aps.append((i0, i1, i2, i3))
            ap_indices[i0].append(ap_id)
            ap_indices[i1].append(ap_id)
            ap_indices[i2].append(ap_id)
            ap_indices[i3].append(ap_id)

    def count_violations(coloring):
        violations = 0
        for ap in aps:
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                violations += 1
        return violations

    def count_distinct_colors(coloring):
        return len(set(coloring))

    # Try to find a valid coloring with k colors, starting from k=3
    for k in range(3, 129):
        if time.time() > deadline - 10.0:
            break
            
        # Multiple restarts for better exploration
        best_found = None
        best_energy = float('inf')
        
        for restart in range(3):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize with random colors
            coloring = [random.randint(0, k - 1) for _ in range(N)]
            
            current_violations = count_violations(coloring)
            current_distinct = count_distinct_colors(coloring)
            current_energy = current_violations * 1000 + current_distinct
            
            best_coloring = coloring[:]
            best_violations = current_violations
            best_distinct = current_distinct
            best_energy_local = current_energy
            
            # Simulated Annealing with penalty-based objective
            T = 10.0
            cooling = 0.9995
            steps = 0
            max_steps = 50000
            
            while steps < max_steps and time.time() < deadline - 3.0:
                steps += 1
                idx = random.randint(0, N - 1)
                old_color = coloring[idx]
                new_color = random.randint(0, k - 1)
                
                if old_color == new_color:
                    continue
                    
                # Calculate change in violations
                delta_violations = 0
                for ap_id in ap_indices[idx]:
                    ap = aps[ap_id]
                    c0 = coloring[ap[0]]
                    c1 = coloring[ap[1]]
                    c2 = coloring[ap[2]]
                    c3 = coloring[ap[3]]
                    
                    old_set_size = len({c0, c1, c2, c3})
                    
                    new_c0 = new_color if ap[0] == idx else c0
                    new_c1 = new_color if ap[1] == idx else c1
                    new_c2 = new_color if ap[2] == idx else c2
                    new_c3 = new_color if ap[3] == idx else c3
                    
                    new_set_size = len({new_c0, new_c1, new_c2, new_c3})
                    
                    if old_set_size < 3 and new_set_size >= 3:
                        delta_violations -= 1
                    elif old_set_size >= 3 and new_set_size < 3:
                        delta_violations += 1
                
                # Calculate change in distinct colors
                old_distinct = current_distinct
                # Check if new_color is already used
                color_used = [False] * k
                for c in coloring:
                    color_used[c] = True
                color_used[old_color] = False
                # Recount distinct after removing old_color
                new_distinct = sum(1 for i in range(k) if color_used[i])
                if not color_used[new_color]:
                    new_distinct += 1
                
                delta_distinct = new_distinct - old_distinct
                delta_energy = delta_violations * 1000 + delta_distinct
                
                if delta_energy < 0:
                    # Always accept improvements
                    coloring[idx] = new_color
                    current_violations += delta_violations
                    current_distinct = new_distinct
                    current_energy += delta_energy
                    if current_energy < best_energy_local:
                        best_energy_local = current_energy
                        best_violations = current_violations
                        best_distinct = current_distinct
                        best_coloring = coloring[:]
                else:
                    # Accept worse solutions with some probability
                    if delta_energy == 0:
                        accept = random.random() < 0.5
                    else:
                        if T > 0:
                            accept = random.random() < np.exp(-delta_energy / T)
                        else:
                            accept = False
                    
                    if accept:
                        coloring[idx] = new_color
                        current_violations += delta_violations
                        current_distinct = new_distinct
                        current_energy += delta_energy
            
            if best_violations == 0:
                return best_coloring
            
            if best_energy_local < best_energy:
                best_energy = best_energy_local
                best_found = best_coloring[:]

        if best_found is not None and count_violations(best_found) == 0:
            return best_found

    # Fallback: Use N distinct colors. 
    # This guarantees validity because every 4-AP has 4 distinct positions, 
    # thus 4 distinct colors (since all labels are unique).
    return list(range(N))
