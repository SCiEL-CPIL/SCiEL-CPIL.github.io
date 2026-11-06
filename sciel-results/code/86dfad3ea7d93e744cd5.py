import random
import time
import math

def construct():
    N = 128
    deadline = time.time() + 55.0  # Safety margin for 60s limit
    
    # 1. Precompute all 4-APs
    # AP is (a, a+d, a+2d, a+3d) with 0 <= a, d >= 1, a+3d < N
    aps = []
    ap_indices = [[] for _ in range(N)]
    for a in range(N):
        # Max d such that a + 3d < N => 3d < N - a => d < (N - a) / 3
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            i0 = a
            i1 = a + d
            i2 = a + 2 * d
            i3 = a + 3 * d
            ap_id = len(aps)
            aps.append((i0, i1, i2, i3))
            ap_indices[i0].append(ap_id)
            ap_indices[i1].append(ap_id)
            ap_indices[i2].append(ap_id)
            ap_indices[i3].append(ap_id)
            
    # 2. Fast distinct count for 4 elements
    def count_distinct_4(c0, c1, c2, c3):
        # This unrolled logic is faster than set() for small integers
        if c0 == c1:
            if c2 == c3:
                if c0 == c2:
                    return 1
                else:
                    return 2
            else:
                if c0 == c2:
                    return 2
                elif c0 == c3:
                    return 2
                else:
                    return 3
        else:
            if c2 == c3:
                if c2 == c0:
                    return 2
                elif c2 == c1:
                    return 2
                else:
                    return 3
            else:
                if c2 == c0:
                    if c3 == c0:
                        return 2
                    elif c3 == c1:
                        return 3
                    else:
                        return 4
                elif c2 == c1:
                    if c3 == c0:
                        return 3
                    elif c3 == c1:
                        return 2
                    else:
                        return 4
                else:
                    if c3 == c0:
                        return 3
                    elif c3 == c1:
                        return 3
                    elif c3 == c2:
                        return 4
                    else:
                        return 4

    best_result = None
    best_energy = float('inf')
    
    # Try K from 3 upwards. 
    # The problem likely has a solution with 3 or 4 colors.
    for K in range(3, 129):
        if time.time() > deadline - 15.0:
            break
            
        # Multiple restarts for this K
        # Increase restarts if K is small (harder to satisfy? or easier?)
        # Generally, fewer colors means more constraints per color, so harder to find valid.
        # But we are minimizing K, so we start with the hardest.
        for restart in range(10):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize coloring
            coloring = [random.randint(0, K - 1) for _ in range(N)]
            color_counts = [0] * K
            for c in coloring:
                color_counts[c] += 1
            
            current_distinct = sum(1 for c in color_counts if c > 0)
            
            # Calculate initial violations
            current_violations = 0
            for ap in aps:
                c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
                if count_distinct_4(c0, c1, c2, c3) < 3:
                    current_violations += 1
                    
            current_energy = current_violations * 10000 + current_distinct
            
            best_coloring_local = coloring[:]
            best_violations_local = current_violations
            best_distinct_local = current_distinct
            best_energy_local = current_energy
            
            if current_violations == 0:
                return coloring

            # Simulated Annealing
            T = 5.0
            cooling = 0.9998
            steps = 0
            max_steps = 20000
            
            while steps < max_steps and time.time() < deadline - 2.0:
                steps += 1
                idx = random.randint(0, N - 1)
                old_color = coloring[idx]
                new_color = random.randint(0, K - 1)
                
                if old_color == new_color:
                    continue
                    
                # Delta Distinct Colors
                old_count = color_counts[old_color]
                new_count = color_counts[new_color]
                
                delta_distinct = 0
                if old_count == 1:
                    delta_distinct -= 1
                if new_count == 0:
                    delta_distinct += 1
                
                # Delta Violations
                delta_violations = 0
                for ap_id in ap_indices[idx]:
                    ap = aps[ap_id]
                    c0 = coloring[ap[0]]
                    c1 = coloring[ap[1]]
                    c2 = coloring[ap[2]]
                    c3 = coloring[ap[3]]
                    
                    old_dist = count_distinct_4(c0, c1, c2, c3)
                    
                    # Construct new colors for this AP
                    n0 = new_color if ap[0] == idx else c0
                    n1 = new_color if ap[1] == idx else c1
                    n2 = new_color if ap[2] == idx else c2
                    n3 = new_color if ap[3] == idx else c3
                    new_dist = count_distinct_4(n0, n1, n2, n3)
                    
                    if old_dist < 3 and new_dist >= 3:
                        delta_violations -= 1
                    elif old_dist >= 3 and new_dist < 3:
                        delta_violations += 1
                
                delta_energy = delta_violations * 10000 + delta_distinct
                
                if delta_energy < 0:
                    # Accept improvement
                    coloring[idx] = new_color
                    color_counts[old_color] -= 1
                    color_counts[new_color] += 1
                    current_violations += delta_violations
                    current_distinct += delta_distinct
                    current_energy += delta_energy
                    
                    if current_energy < best_energy_local:
                        best_energy_local = current_energy
                        best_violations_local = current_violations
                        best_distinct_local = current_distinct
                        best_coloring_local = coloring[:]
                else:
                    # Accept worse with probability
                    if T > 0:
                        # Use math.exp for speed
                        prob = math.exp(-delta_energy / T)
                        if random.random() < prob:
                            coloring[idx] = new_color
                            color_counts[old_color] -= 1
                            color_counts[new_color] += 1
                            current_violations += delta_violations
                            current_distinct += delta_distinct
                            current_energy += delta_energy
                            # We don't update best_local here because it's worse,
                            # unless we want to track the best "current" state.
                            # But since we accept worse, we might move away from best_local.
                            # This is correct SA behavior.
                            
                T *= cooling
                
            # End of SA run for this restart
            if best_violations_local == 0:
                return best_coloring_local
            
            if best_energy_local < best_energy:
                best_energy = best_energy_local
                best_result = best_coloring_local[:]

    # Fallback: If no valid coloring found with K < 128, return identity (128 colors)
    # This is always valid.
    if best_result is not None:
        return best_result
    return list(range(N))
