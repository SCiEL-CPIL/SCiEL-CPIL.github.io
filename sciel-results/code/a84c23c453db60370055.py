import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Precompute all 4-term arithmetic progressions
    # a, a+d, a+2d, a+3d <= N
    # d >= 1
    aps = []
    index_to_aps = [[] for _ in range(N)]
    
    for d in range(1, (N - 3) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
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
            
    # Helper to count monochromatic 4-APs
    def count_mono(colors):
        count = 0
        for ap in aps:
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count

    best_score = float('inf')
    best_colors = None
    
    # Run multiple restarts or one long SA?
    # Given the speed, we can do a long SA with restarts.
    
    # Parameters for SA
    T_initial = 1.0
    T_final = 0.001
    cooling_rate = 0.99995  # Slow cooling to explore more
    
    # We will run for a set amount of time, restarting if stuck
    # Or just one long run with periodic restarts if no improvement
    
    while time.time() < deadline:
        # Initialize coloring
        if best_colors is None:
            colors = [random.randint(0, 1) for _ in range(N)]
        else:
            # Perturbed restart: start from best solution and flip some bits
            colors = best_colors[:]
            num_flips = random.randint(5, 20)
            indices_to_flip = random.sample(range(N), num_flips)
            for idx in indices_to_flip:
                colors[idx] = 1 - colors[idx]
                
        current_score = count_mono(colors)
        
        if current_score < best_score:
            best_score = current_score
            best_colors = colors[:]
            
        T = T_initial
        no_improve_steps = 0
        max_no_improve = 10000  # Restart if stuck
        
        while time.time() < deadline - 1.0:
            if T < T_final or no_improve_steps > max_no_improve:
                break
                
            # Choose move: 1-bit flip or 2-bit flip
            # 50% chance of 1-bit, 50% chance of 2-bit
            if random.random() < 0.5:
                # 1-bit flip
                k = random.randint(0, N - 1)
                c_k = colors[k]
                new_c_k = 1 - c_k
                
                delta = 0
                # Check all APs containing k
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
                    
                    # Check if other 3 are monochromatic
                    if c_o1 == c_o2 == c_o3:
                        if c_o1 == c_k:
                            # Currently mono, will become non-mono
                            delta -= 1
                        else:
                            # Currently non-mono, will become mono
                            delta += 1
                            
                if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                    colors[k] = new_c_k
                    current_score += delta
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = colors[:]
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    no_improve_steps += 1
                    
            else:
                # 2-bit flip
                k1 = random.randint(0, N - 1)
                k2 = random.randint(0, N - 1)
                while k2 == k1:
                    k2 = random.randint(0, N - 1)
                    
                # Calculate delta for flipping k1 and k2
                # We need to check all APs containing k1 or k2
                # To avoid double counting or missing interactions, we can
                # just evaluate the cost change for the union of APs.
                
                # Get unique AP ids
                affected_aps = set()
                for ap_id, _ in index_to_aps[k1]:
                    affected_aps.add(ap_id)
                for ap_id, _ in index_to_aps[k2]:
                    affected_aps.add(ap_id)
                    
                # Calculate old cost for these APs
                old_cost = 0
                for ap_id in affected_aps:
                    ap = aps[ap_id]
                    if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                        old_cost += 1
                        
                # Simulate flip
                c1_old = colors[k1]
                c2_old = colors[k2]
                colors[k1] = 1 - c1_old
                colors[k2] = 1 - c2_old
                
                # Calculate new cost for these APs
                new_cost = 0
                for ap_id in affected_aps:
                    ap = aps[ap_id]
                    if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                        new_cost += 1
                        
                delta = new_cost - old_cost
                
                # Acceptance
                if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                    current_score += delta
                    # Colors are already updated
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = colors[:]
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    # Revert
                    colors[k1] = c1_old
                    colors[k2] = c2_old
                    no_improve_steps += 1
                    
            T *= cooling_rate
            
    return best_colors
