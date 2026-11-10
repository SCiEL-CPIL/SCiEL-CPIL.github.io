import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
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
            
    total_aps = len(aps)
    
    # Helper to count monochromatic APs from scratch
    def count_mono(colors):
        count = 0
        for ap in aps:
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count
    
    # Try several different initializations to find a good starting point
    initializations = []
    
    # 1. Random
    initializations.append([random.randint(0, 1) for _ in range(N)])
    
    # 2. Block pattern (i // 2) % 2
    initializations.append([(i // 2) % 2 for i in range(N)])
    
    # 3. Thue-Morse like: bit count mod 2
    initializations.append([bin(i).count('1') % 2 for i in range(N)])
    
    # 4. Another random
    initializations.append([random.randint(0, 1) for _ in range(N)])
    
    # 5. Periodic pattern
    initializations.append([i % 2 for i in range(N)])
    
    # 6. Block pattern (i // 3) % 2
    initializations.append([(i // 3) % 2 for i in range(N)])
    
    # 7. Block pattern (i // 4) % 2
    initializations.append([(i // 4) % 2 for i in range(N)])
    
    # 8. Another Thue-Morse variant
    initializations.append([bin(i + 1).count('1') % 2 for i in range(N)])
    
    # Evaluate all initializations and pick the best
    best_init = None
    best_init_score = float('inf')
    
    for init_colors in initializations:
        score = count_mono(init_colors)
        if score < best_init_score:
            best_init_score = score
            best_init = init_colors[:]
            
    colors = best_init
    current_score = best_init_score
    best_score = current_score
    best_colors = colors[:]
    
    # Simulated Annealing with adaptive temperature
    T = 2.0
    cooling_rate = 0.99995
    min_T = 0.0005
    
    while time.time() < deadline:
        if T < min_T:
            # Restart with new random coloring if time allows
            if time.time() < deadline - 5.0:
                # Try a few random restarts and pick the best
                restart_colors = [random.randint(0, 1) for _ in range(N)]
                restart_score = count_mono(restart_colors)
                
                # Also try the block pattern
                block_colors = [(i // 2) % 2 for i in range(N)]
                block_score = count_mono(block_colors)
                
                # Try Thue-Morse
                tm_colors = [bin(i).count('1') % 2 for i in range(N)]
                tm_score = count_mono(tm_colors)
                
                # Pick the best restart
                candidates = [(restart_score, restart_colors), (block_score, block_colors), (tm_score, tm_colors)]
                candidates.sort(key=lambda x: x[0])
                restart_score, restart_colors = candidates[0]
                
                if restart_score < best_score:
                    best_score = restart_score
                    best_colors = restart_colors[:]
                    colors = restart_colors
                    current_score = restart_score
                else:
                    colors = restart_colors
                    current_score = restart_score
                    
                T = 2.0
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
                if delta < 100 * T:  # heuristic to avoid underflow/slow exp
                    if random.random() < math.exp(-delta / T):
                        colors[k] = 1 - c_k
                        current_score += delta
                        if current_score < best_score:
                            best_score = current_score
                            best_colors = colors[:]
                        
        T *= cooling_rate
        
    return best_colors
