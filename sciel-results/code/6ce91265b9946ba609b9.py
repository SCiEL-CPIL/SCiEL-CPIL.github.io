import time
import random
import numpy as np

def construct():
    n = 256
    # Deadline: 110s to be safe within 120s limit
    deadline = time.time() + 110.0
    
    # Precompute all 4-term arithmetic progressions
    # a, a+d, a+2d, a+3d with a >= 1, a+3d <= 256
    # Indices in 0..255 correspond to 1..256
    aps = []
    max_d = (n - 1) // 3
    for d in range(1, max_d + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            # Store 0-indexed
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    num_aps = len(aps)
    
    # Precompute incidence for fast delta calculation
    # ap_indices[i] = list of (other1, other2, other3) for each AP containing i
    ap_indices = [[] for _ in range(n)]
    for ap in aps:
        for pos in range(4):
            i = ap[pos]
            others = [ap[p] for p in range(4) if p != pos]
            ap_indices[i].append((others[0], others[1], others[2]))

    # Numpy arrays for fast bulk cost calculation
    ap_arr = np.array(aps, dtype=np.int32)
    a0 = ap_arr[:, 0]
    a1 = ap_arr[:, 1]
    a2 = ap_arr[:, 2]
    a3 = ap_arr[:, 3]

    def count_mono_numpy(c):
        """Calculate total monochromatic APs using NumPy. c is a numpy array."""
        c0 = c[a0]
        c1 = c[a1]
        c2 = c[a2]
        c3 = c[a3]
        mono_mask = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(mono_mask))

    def delta_cost(c_list, i):
        """Calculate change in cost if we flip bit i. c_list is a Python list."""
        old_val = c_list[i]
        new_val = 1 - old_val
        delta = 0
        for o1, o2, o3 in ap_indices[i]:
            # Check if AP was monochromatic before flip
            # It was mono if c[o1]==c[o2]==c[o3]==old_val
            if c_list[o1] == old_val and c_list[o2] == old_val and c_list[o3] == old_val:
                # It was mono. After flip, c[i] becomes new_val, so it's not mono anymore.
                # delta -= 1
                delta -= 1
            else:
                # It was not mono.
                # Check if it becomes mono after flip.
                # It becomes mono if c[o1]==c[o2]==c[o3]==new_val
                if c_list[o1] == new_val and c_list[o2] == new_val and c_list[o3] == new_val:
                    # It becomes mono.
                    # delta += 1
                    delta += 1
        return delta

    best_cost = float('inf')
    best_c = None
    
    # Main Search Loop: Multiple Restarts with Simulated Annealing / Hill Climbing
    # We will run several restarts. Each restart does local search.
    
    while time.time() < deadline:
        # Initialize
        # Use a random start, but maybe bias towards balanced colors?
        # Random is fine.
        c = [random.randint(0, 1) for _ in range(n)]
        c_np = np.array(c, dtype=np.int8)
        current_cost = count_mono_numpy(c_np)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_c = c[:]
            
        # Local Search Phase
        # We use a combination of steepest descent (single flip) and kicks (multi-flip)
        temperature = 5.0
        min_temperature = 0.1
        
        while time.time() < deadline:
            # 1. Try to find a single flip that improves cost
            best_delta = 0
            best_i = -1
            
            # Calculate delta for all i
            # We can break early if we find a very good one? No, find best.
            for i in range(n):
                d = delta_cost(c, i)
                if d < best_delta:
                    best_delta = d
                    best_i = i
                    
            if best_i != -1 and best_delta < 0:
                # Accept the flip
                c[best_i] = 1 - c[best_i]
                c_np[best_i] = c[best_i]
                current_cost += best_delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_c = c[:]
                    
                # Continue local search immediately
                continue
            else:
                # No single flip improves cost. We are at a local minimum.
                # Perform a kick: flip k random bits
                kick_size = random.randint(2, 5)
                kick_indices = random.sample(range(n), kick_size)
                
                # Apply kick
                for i in kick_indices:
                    c[i] = 1 - c[i]
                    c_np[i] = c[i]
                    
                new_cost = count_mono_numpy(c_np)
                delta_kick = new_cost - current_cost
                
                # Metropolis acceptance
                if delta_kick < 0:
                    accept = True
                else:
                    # Accept with probability exp(-delta / T)
                    # If T is very small, reject.
                    if temperature > min_temperature:
                        prob = np.exp(-delta_kick / temperature)
                        accept = (random.random() < prob)
                    else:
                        accept = False
                
                if accept:
                    current_cost = new_cost
                    # Cool down
                    temperature *= 0.95
                    if temperature < min_temperature:
                        temperature = min_temperature
                else:
                    # Revert kick
                    for i in kick_indices:
                        c[i] = 1 - c[i]
                        c_np[i] = c[i]
                    # Reset temperature or restart?
                    # If we reject, we might be stuck. Let's restart the local search loop
                    # with a new random state if we've been stuck for a while?
                    # For now, just try another kick.
                    # To avoid infinite loops of rejections, we can force a restart if T is low.
                    if temperature <= min_temperature:
                        break
                    else:
                        # Try again
                        continue
        
        # If we broke out of the inner loop due to time or low T, check time for next restart
        if time.time() >= deadline:
            break

    if best_c is None:
        # Fallback
        return [(i // 2) % 2 for i in range(256)]
        
    return best_c
