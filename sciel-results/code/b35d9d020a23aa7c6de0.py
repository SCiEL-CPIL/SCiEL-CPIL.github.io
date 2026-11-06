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

    # Try to find a valid coloring with k colors, starting from k=3
    for k in range(3, 129):
        if time.time() > deadline - 10.0:
            # If we are running out of time, we might not find a valid one for this k.
            # But we must return a valid one.
            # We break and fall through to the fallback.
            break
            
        # Initialize with random colors
        coloring = [random.randint(0, k - 1) for _ in range(N)]
        
        current_violations = count_violations(coloring)
        best_coloring = coloring[:]
        best_violations = current_violations
        
        # Simulated Annealing
        T = 5.0
        cooling = 0.999
        steps = 0
        max_steps = 100000
        
        while current_violations > 0 and steps < max_steps and time.time() < deadline - 2.0:
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
            
            if delta_violations < 0:
                # Always accept improvements
                coloring[idx] = new_color
                current_violations += delta_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_coloring = coloring[:]
            else:
                # Accept worse solutions with some probability
                if delta_violations == 0:
                    accept = random.random() < 0.5
                else:
                    # Avoid division by zero or exp overflow
                    if T > 0:
                        accept = random.random() < np.exp(-delta_violations / T)
                    else:
                        accept = False
                
                if accept:
                    coloring[idx] = new_color
                    current_violations += delta_violations
            
            T *= cooling

        if current_violations == 0:
            return coloring
        if best_violations == 0:
            return best_coloring

    # Fallback: Use N distinct colors. 
    # This guarantees validity because every 4-AP has 4 distinct positions, 
    # thus 4 distinct colors (since all labels are unique).
    return list(range(N))
