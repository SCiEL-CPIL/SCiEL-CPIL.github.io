import random
import math

def construct():
    N = 256
    # Generate all 4-APs
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    # Pre-compute for each index, which APs it belongs to and what the other indices are
    aps_by_index = [[] for _ in range(N)]
    for ap in aps:
        i1, i2, i3, i4 = ap
        aps_by_index[i1].append((i2, i3, i4))
        aps_by_index[i2].append((i1, i3, i4))
        aps_by_index[i3].append((i1, i2, i4))
        aps_by_index[i4].append((i1, i2, i3))

    # Initialize with random coloring
    random.seed(42)
    
    def calculate_score(colors):
        count = 0
        for ap in aps:
            c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c1 == c2 == c3 == c4:
                count += 1
        return count

    best_score = float('inf')
    best_colors = None

    # Simulated Annealing parameters
    T_start = 5.0
    T_end = 0.01
    num_restarts = 25
    steps_per_restart = 25000
    
    for r in range(num_restarts):
        colors = [random.randint(0, 1) for _ in range(N)]
        current_score = calculate_score(colors)
        
        T = T_start
        cooling_factor = (T_end / T_start) ** (1.0 / steps_per_restart)
        
        for s in range(steps_per_restart):
            i = random.randint(0, N - 1)
            
            # Calculate delta E
            delta_e = 0
            for others in aps_by_index[i]:
                o1, o2, o3 = others
                c_o1, c_o2, c_o3 = colors[o1], colors[o2], colors[o3]
                
                if c_o1 == 0 and c_o2 == 0 and c_o3 == 0:
                    if colors[i] == 0:
                        delta_e -= 1
                    else:
                        delta_e += 1
                elif c_o1 == 1 and c_o2 == 1 and c_o3 == 1:
                    if colors[i] == 1:
                        delta_e -= 1
                    else:
                        delta_e += 1
            
            if delta_e < 0:
                colors[i] = 1 - colors[i]
                current_score += delta_e
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
            else:
                if T > 0:
                    prob = math.exp(-delta_e / T)
                    if random.random() < prob:
                        colors[i] = 1 - colors[i]
                        current_score += delta_e
                        if current_score < best_score:
                            best_score = current_score
                            best_colors = colors[:]
            
            T *= cooling_factor

    return best_colors
