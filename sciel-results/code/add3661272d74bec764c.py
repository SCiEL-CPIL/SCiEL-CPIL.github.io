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
    
    # Helper to count monochromatic APs
    def count_mono(colors):
        count = 0
        for ap in aps:
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count

    # Helper to get delta for flipping a single bit
    def get_delta_1bit(colors, k):
        delta = 0
        c_k = colors[k]
        for ap_id, pos in index_to_aps[k]:
            ap = aps[ap_id]
            # Get the other 3 indices
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
        return delta

    # Helper to get delta for flipping two bits i and j
    def get_delta_2bit(colors, i, j):
        # We need to consider all APs containing i or j
        # To avoid double counting or complex logic, we can just evaluate the "before" and "after" 
        # for the union of APs.
        # However, a full recount of the union is O(degree). 
        # Let's optimize: 
        # Delta = (New Mono Count in Union) - (Old Mono Count in Union)
        
        ap_ids_to_check = set()
        for ap_id, _ in index_to_aps[i]:
            ap_ids_to_check.add(ap_id)
        for ap_id, _ in index_to_aps[j]:
            ap_ids_to_check.add(ap_id)
            
        old_mono = 0
        new_mono = 0
        
        # Save old colors
        c_i = colors[i]
        c_j = colors[j]
        
        for ap_id in ap_ids_to_check:
            ap = aps[ap_id]
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            
            if c0 == c1 == c2 == c3:
                old_mono += 1
                
            # Calculate new colors for this AP
            # If ap contains i, flip it. If it contains j, flip it.
            # We can do this by checking indices
            n0 = c0
            n1 = c1
            n2 = c2
            n3 = c3
            
            # Faster: just check if i or j is in the ap
            # Since we precomputed index_to_aps, we know which pos i and j are in, 
            # but here we just have ap_id. 
            # It's faster to just flip in a temp array? No, N is small.
            # Let's just do the check.
            if ap[0] == i: n0 = 1 - n0
            if ap[1] == i: n1 = 1 - n1
            if ap[2] == i: n2 = 1 - n2
            if ap[3] == i: n3 = 1 - n3
            
            if ap[0] == j: n0 = 1 - n0
            if ap[1] == j: n1 = 1 - n1
            if ap[2] == j: n2 = 1 - n2
            if ap[3] == j: n3 = 1 - n3
            
            if n0 == n1 == n2 == n3:
                new_mono += 1
                
        return new_mono - old_mono

    # Generate diverse initializations
    def generate_initializations():
        inits = []
        # 1. Block pattern (i // 2) % 2
        inits.append([(i // 2) % 2 for i in range(N)])
        # 2. Thue-Morse like
        inits.append([bin(i).count('1') % 2 for i in range(N)])
        # 3. Random
        inits.append([random.randint(0, 1) for _ in range(N)])
        # 4. Random
        inits.append([random.randint(0, 1) for _ in range(N)])
        # 5. Periodic 4
        inits.append([(i // 4) % 2 for i in range(N)])
        
        return inits

    # Evaluate initializations
    best_init = None
    best_init_score = float('inf')
    
    for init_colors in generate_initializations():
        score = count_mono(init_colors)
        if score < best_init_score:
            best_init_score = score
            best_init = init_colors[:]
            
    colors = best_init
    current_score = best_init_score
    best_score = current_score
    best_colors = colors[:]
    
    # Simulated Annealing with 1-bit and 2-bit flips
    T = 1.0
    cooling_rate = 0.99995
    min_T = 0.01
    
    # Track stagnation
    steps_since_improvement = 0
    max_steps_no_improve = 5000
    
    while time.time() < deadline:
        if T < min_T:
            # Restart with perturbation of the best solution
            if time.time() > deadline - 10.0:
                break
                
            # Perturb best_colors
            colors = best_colors[:]
            num_flips = random.randint(10, 20)
            for _ in range(num_flips):
                k = random.randint(0, N - 1)
                colors[k] = 1 - colors[k]
                
            current_score = count_mono(colors)
            if current_score < best_score:
                best_score = current_score
                best_colors = colors[:]
                
            T = 1.0
            steps_since_improvement = 0
            continue
            
        # Decide move type
        use_2bit = random.random() < 0.2
        
        if use_2bit:
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                # Fall back to 1-bit if same
                use_2bit = False
                # Re-select i? Or just do 1-bit
                # Let's just do 1-bit
                delta = get_delta_1bit(colors, i)
                if delta < 0 or random.random() < math.exp(-delta / T):
                    colors[i] = 1 - colors[i]
                    current_score += delta
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = colors[:]
                        steps_since_improvement = 0
                    else:
                        steps_since_improvement += 1
            else:
                delta = get_delta_2bit(colors, i, j)
                if delta < 0 or random.random() < math.exp(-delta / T):
                    colors[i] = 1 - colors[i]
                    colors[j] = 1 - colors[j]
                    current_score += delta
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = colors[:]
                        steps_since_improvement = 0
                    else:
                        steps_since_improvement += 1
        else:
            k = random.randint(0, N - 1)
            delta = get_delta_1bit(colors, k)
            if delta < 0 or random.random() < math.exp(-delta / T):
                colors[k] = 1 - colors[k]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1

        # Force restart if stuck too long
        if steps_since_improvement > max_steps_no_improve:
            T = 1.0
            steps_since_improvement = 0
            # Perturb
            colors = best_colors[:]
            for _ in range(15):
                k = random.randint(0, N - 1)
                colors[k] = 1 - colors[k]
            current_score = count_mono(colors)
            if current_score < best_score:
                best_score = current_score
                best_colors = colors[:]
                
        T *= cooling_rate
        
    return best_colors
