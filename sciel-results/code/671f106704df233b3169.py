import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    index_to_aps = [[] for _ in range(N)]
    
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            idx1 = a - 1
            idx2 = a + d - 1
            idx3 = a + 2 * d - 1
            idx4 = a + 3 * d - 1
            
            ap = (idx1, idx2, idx3, idx4)
            ap_id = len(aps)
            aps.append(ap)
            
            index_to_aps[idx1].append((ap_id, 0))
            index_to_aps[idx2].append((ap_id, 1))
            index_to_aps[idx3].append((ap_id, 2))
            index_to_aps[idx4].append((ap_id, 3))
            
    # Initial random coloring
    colors = [random.randint(0, 1) for _ in range(N)]
    
    # Calculate initial score
    current_score = 0
    for ap in aps:
        if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
            current_score += 1
            
    best_score = current_score
    best_colors = colors[:]
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.9999
    min_T = 0.001
    
    while time.time() < deadline:
        if T < min_T:
            # Restart with new random coloring if time allows
            if time.time() < deadline - 5.0:
                colors = [random.randint(0, 1) for _ in range(N)]
                current_score = 0
                for ap in aps:
                    if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                        current_score += 1
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
                T = 1.0
            else:
                break
                
        k = random.randint(0, N - 1)
        delta = 0
        c_k = colors[k]
        
        for ap_id, pos in index_to_aps[k]:
            ap = aps[ap_id]
            # Determine the other 3 indices
            if pos == 0:
                o1, o2, o3 = ap[1], ap[2], ap[3]
            elif pos == 1:
                o1, o2, o3 = ap[0], ap[2], ap[3]
            elif pos == 2:
                o1, o2, o3 = ap[0], ap[1], ap[3]
            else:
                o1, o2, o3 = ap[0], ap[1], ap[2]
                
            c_o1 = colors[o1]
            c_o2 = colors[o2]
            c_o3 = colors[o3]
            
            # Check if the other 3 are monochromatic
            if c_o1 == c_o2 == c_o3:
                if c_k == c_o1:
                    # Currently mono, will become non-mono
                    delta -= 1
                else:
                    # Currently non-mono, will become mono
                    delta += 1
                    
        # Acceptance criterion
        if delta < 0:
            colors[k] = 1 - c_k
            current_score += delta
            if current_score < best_score:
                best_score = current_score
                best_colors = colors[:]
        else:
            # delta >= 0
            # Accept with probability exp(-delta / T)
            # If T is very small, exp is ~0
            if T > 0.0:
                # Optimization: if delta is large relative to T, skip exp
                if delta < 100 * T: # heuristic to avoid underflow/slow exp
                    if random.random() < math.exp(-delta / T):
                        colors[k] = 1 - c_k
                        current_score += delta
                        if current_score < best_score:
                            best_score = current_score
                            best_colors = colors[:]
                        
        T *= cooling_rate
        
    return best_colors
