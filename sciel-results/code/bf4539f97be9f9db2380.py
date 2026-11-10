import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
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
    
    # Convert to numpy for faster operations
    aps_arr = np.array(aps, dtype=np.int32)
    
    def count_mono_numpy(colors_np):
        """Count monochromatic APs using numpy vectorization"""
        c0 = colors_np[aps_arr[:, 0]]
        c1 = colors_np[aps_arr[:, 1]]
        c2 = colors_np[aps_arr[:, 2]]
        c3 = colors_np[aps_arr[:, 3]]
        # All four must be equal: (c0==c1) & (c0==c2) & (c0==c3)
        return int(np.sum((c0 == c1) & (c0 == c2) & (c0 == c3)))
    
    def count_mono_python(colors):
        """Count monochromatic APs using pure Python (for reference)"""
        count = 0
        for ap in aps:
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count
    
    # Generate diverse initializations
    def generate_initializations():
        inits = []
        
        # 1. Pure random
        inits.append([random.randint(0, 1) for _ in range(N)])
        
        # 2. Block pattern with period 2
        inits.append([(i // 2) % 2 for i in range(N)])
        
        # 3. Block pattern with period 4
        inits.append([(i // 4) % 2 for i in range(N)])
        
        # 4. Striped pattern
        inits.append([i % 2 for i in range(N)])
        
        # 5. Thue-Morse-like pattern
        tm = [0]
        for i in range(1, N):
            tm.append(bin(i).count('1') % 2)
        inits.append(tm[:N])
        
        # 6. Random with bias
        inits.append([random.choice([0, 0, 1]) for _ in range(N)])
        
        # 7. Another random
        inits.append([random.randint(0, 1) for _ in range(N)])
        
        return inits
    
    # Evaluate initializations and pick the best few
    init_scores = []
    for init in generate_initializations():
        init_np = np.array(init, dtype=np.int32)
        score = count_mono_numpy(init_np)
        init_scores.append((score, init))
    
    # Sort by score, take top 3 for multi-start
    init_scores.sort(key=lambda x: x[0])
    best_inits = [x[1] for x in init_scores[:3]]
    
    best_score = min(x[0] for x in init_scores)
    best_colors = best_inits[0][:]
    
    # Multi-start simulated annealing
    for start_idx, initial_colors in enumerate(best_inits):
        if time.time() > deadline - 5.0:
            break
        
        colors = initial_colors[:]
        colors_np = np.array(colors, dtype=np.int32)
        current_score = count_mono_numpy(colors_np)
        
        if current_score < best_score:
            best_score = current_score
            best_colors = colors[:]
        
        # Simulated annealing parameters
        T = 1.0
        cooling_rate = 0.99995
        min_T = 0.005
        
        steps_since_improvement = 0
        max_steps_no_improve = 5000
        
        while time.time() < deadline - 5.0:
            if T < min_T:
                # Restart with perturbation
                if time.time() > deadline - 10.0:
                    break
                # Perturb current best
                colors = best_colors[:]
                for _ in range(10):
                    k = random.randint(0, N - 1)
                    colors[k] = 1 - colors[k]
                colors_np = np.array(colors, dtype=np.int32)
                current_score = count_mono_numpy(colors_np)
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
                T = 0.5
                steps_since_improvement = 0
                continue
            
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
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                # delta >= 0
                # Accept with probability exp(-delta / T)
                if T > 0.0:
                    if delta < 50 * T:
                        if random.random() < math.exp(-delta / T):
                            colors[k] = 1 - c_k
                            current_score += delta
                            if current_score < best_score:
                                best_score = current_score
                                best_colors = colors[:]
                                steps_since_improvement = 0
                            else:
                                steps_since_improvement += 1
                else:
                    steps_since_improvement += 1
            
            # Early termination if stuck
            if steps_since_improvement > max_steps_no_improve:
                break
                
            T *= cooling_rate
    
    # Final polish: try single-bit flips on best solution
    if time.time() < deadline - 3.0:
        colors = best_colors[:]
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for k in range(N):
                delta = 0
                c_k = colors[k]
                for ap_id, pos in index_to_aps[k]:
                    ap = aps[ap_id]
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
                    
                    if c_o1 == c_o2 == c_o3:
                        if c_k == c_o1:
                            delta -= 1
                        else:
                            delta += 1
                            
                if delta < 0:
                    colors[k] = 1 - c_k
                    current_score += delta
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = colors[:]
                        improved = True
    
    return best_colors
