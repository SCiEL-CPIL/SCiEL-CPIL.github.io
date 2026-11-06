import time
import random
import math

def construct():
    start_time = time.time()
    # The task contract mentions 60s limit. The directive mentions 90s kill.
    # To be safe against the "60s" constraint in the contract text, we use 55s.
    deadline = start_time + 55.0
    
    N = 256
    
    # Precompute all 3-APs and 4-APs
    # three_ap_others[x] = list of masks of the other 2 elements in 3-APs with x
    # four_ap_others[x] = list of masks of the other 3 elements in 4-APs with x
    three_ap_others = [[] for _ in range(N + 1)]
    four_ap_others = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        max_d_3 = (N - a) // 2
        for d in range(1, max_d_3 + 1):
            b = a + d
            c = a + 2 * d
            
            # 3-AP: a, b, c
            mask_bc = (1 << b) | (1 << c)
            mask_ac = (1 << a) | (1 << c)
            mask_ab = (1 << a) | (1 << b)
            
            three_ap_others[a].append(mask_bc)
            three_ap_others[b].append(mask_ac)
            three_ap_others[c].append(mask_ab)
            
            # 4-AP: a, b, c, e
            if a + 3 * d <= N:
                e = a + 3 * d
                mask_bce = (1 << b) | (1 << c) | (1 << e)
                mask_ace = (1 << a) | (1 << c) | (1 << e)
                mask_abe = (1 << a) | (1 << b) | (1 << e)
                mask_abc = (1 << a) | (1 << b) | (1 << c)
                
                four_ap_others[a].append(mask_bce)
                four_ap_others[b].append(mask_ace)
                four_ap_others[c].append(mask_abe)
                four_ap_others[e].append(mask_abc)

    # Helper: Check if adding x is valid (no 4-AP created)
    def is_valid_add(A_mask, x):
        if A_mask & (1 << x):
            return False
        for other_mask in four_ap_others[x]:
            if (A_mask & other_mask) == other_mask:
                return False
        return True

    # Helper: Calculate gain in 3-APs if x is added
    def gain_3aps(A_mask, x):
        if A_mask & (1 << x):
            return 0
        gain = 0
        for other_mask in three_ap_others[x]:
            if (A_mask & other_mask) == other_mask:
                gain += 1
        return gain

    # Helper: Calculate loss in 3-APs if x is removed
    def loss_3aps(A_mask, x):
        if not (A_mask & (1 << x)):
            return 0
        loss = 0
        A_mask_without_x = A_mask & ~(1 << x)
        for other_mask in three_ap_others[x]:
            if (A_mask_without_x & other_mask) == other_mask:
                loss += 1
        return loss

    # Count total 3-APs in A_mask (full scan, used for init and final check)
    # We can also compute this incrementally, but for initialization it's simpler to just sum gains?
    # No, summing gains double/triple counts.
    # Efficient count: iterate all a, d.
    def count_3aps(A_mask):
        count = 0
        for a in range(1, N + 1):
            if not (A_mask & (1 << a)):
                continue
            max_d = (N - a) // 2
            for d in range(1, max_d + 1):
                b = a + d
                c = a + 2 * d
                if (A_mask & (1 << b)) and (A_mask & (1 << c)):
                    count += 1
        return count

    # Precompute degrees for initialization
    deg3 = [0] * (N + 1)
    deg4 = [0] * (N + 1)
    for x in range(1, N + 1):
        deg3[x] = len(three_ap_others[x])
        deg4[x] = len(four_ap_others[x])
        
    # Sort elements by potential: high 3-AP degree, low 4-AP degree
    # Using a ratio to balance density and constraint
    elements = list(range(1, N + 1))
    elements.sort(key=lambda x: (deg3[x] / (deg4[x] + 1), deg3[x]), reverse=True)

    def greedy_init():
        A_mask = 0
        # Use the sorted order for a strong deterministic start
        # Add slight randomness to allow different restarts to explore different basins
        # We will just use the sorted order, but maybe shuffle the last 50%?
        # For now, deterministic greedy is good for the first few, then we can vary.
        # Let's add a small probability to skip an element to create variety
        for x in elements:
            if is_valid_add(A_mask, x):
                # Always add if it increases 3-APs, or with some probability if it doesn't
                g = gain_3aps(A_mask, x)
                if g > 0 or random.random() < 0.1:
                    A_mask |= (1 << x)
        return A_mask

    best_overall_mask = 0
    best_overall_score = 0
    
    # Simulated Annealing Loop
    # We will run multiple SA instances if time permits, or one long one with restarts.
    # Given the complexity, one long SA with restarts on stagnation is robust.
    
    T = 10.0  # Initial temperature
    T_min = 0.1
    cooling = 0.998  # Slow cooling
    
    current_mask = greedy_init()
    current_score = count_3aps(current_mask)
    
    if current_score > best_overall_score:
        best_overall_score = current_score
        best_overall_mask = current_mask
        
    steps_since_improvement = 0
    max_steps_without_improvement = 5000 # Restart threshold
    
    while time.time() < deadline - 2:
        if T < T_min:
            # Restart SA with a new initial solution
            current_mask = greedy_init()
            current_score = count_3aps(current_mask)
            T = 10.0
            steps_since_improvement = 0
            if current_score > best_overall_score:
                best_overall_score = current_score
                best_overall_mask = current_mask
            continue

        # Generate a move
        move_type = random.random()
        accepted = False
        delta = 0
        new_mask = current_mask
        new_score = current_score
        
        if move_type < 0.3: # Add
            x = random.randint(1, N)
            if not (current_mask & (1 << x)):
                if is_valid_add(current_mask, x):
                    delta = gain_3aps(current_mask, x)
                    if delta > 0 or random.random() < math.exp(delta / T):
                        new_mask = current_mask | (1 << x)
                        new_score = current_score + delta
                        accepted = True
                        
        elif move_type < 0.6: # Remove
            # Pick random element in mask
            # Optimization: Precompute list of elements in mask? 
            # Doing it every step is O(N). N=256 is small.
            elems_in_mask = []
            for i in range(1, N + 1):
                if current_mask & (1 << i):
                    elems_in_mask.append(i)
                    
            if elems_in_mask:
                x = random.choice(elems_in_mask)
                delta = -loss_3aps(current_mask, x)
                # Removal is always valid
                if delta > 0 or random.random() < math.exp(delta / T):
                    new_mask = current_mask & ~(1 << x)
                    new_score = current_score + delta
                    accepted = True
                    
        else: # Swap
            # Pick random element in mask to remove, random element out to add
            elems_in_mask = []
            for i in range(1, N + 1):
                if current_mask & (1 << i):
                    elems_in_mask.append(i)
            
            if elems_in_mask:
                x_rem = random.choice(elems_in_mask)
                x_add = random.randint(1, N)
                if not (current_mask & (1 << x_add)):
                    # Check validity of swap
                    A_temp = current_mask & ~(1 << x_rem)
                    valid = True
                    for other_mask in four_ap_others[x_add]:
                        if (A_temp & other_mask) == other_mask:
                            valid = False
                            break
                            
                    if valid:
                        # Calculate net gain
                        gain = 0
                        for other_mask in three_ap_others[x_add]:
                            if (A_temp & other_mask) == other_mask:
                                gain += 1
                            
                        # Loss of removing x_rem from current_mask
                        # Note: loss_3aps calculates loss from current_mask.
                        # Since x_add is not in current_mask, and x_rem is, 
                        # and x_add != x_rem, the APs lost by removing x_rem are independent of x_add 
                        # EXCEPT if x_add was part of an AP with x_rem? 
                        # Wait. 
                        # Loss is # of APs containing x_rem that are fully in current_mask.
                        # Gain is # of APs containing x_add that are fully in A_temp.
                        # Are these disjoint?
                        # An AP containing both x_rem and x_add would be counted in both?
                        # If an AP {x_rem, x_add, z} exists:
                        # 1. It is fully in current_mask? No, x_add is not in current_mask. So it's not lost.
                        # 2. It is fully in A_temp? A_temp has x_rem and z. x_add is added. So yes, it is gained.
                        # So delta = gain - loss is correct.
                        
                        loss = loss_3aps(current_mask, x_rem)
                        delta = gain - loss
                        if delta > 0 or random.random() < math.exp(delta / T):
                            new_mask = A_temp | (1 << x_add)
                            new_score = current_score + delta
                            accepted = True

        if accepted:
            current_mask = new_mask
            current_score = new_score
            steps_since_improvement += 1
            
            if current_score > best_overall_score:
                best_overall_score = current_score
                best_overall_mask = current_mask
                steps_since_improvement = 0
            else:
                # Even if not global best, if it's a local improvement, we might reset a smaller counter?
                # For restart logic, we usually track global best.
                pass
        else:
            steps_since_improvement += 1

        # Cool down
        T *= cooling
        
        # Optional: Restart if stuck for too long
        if steps_since_improvement > max_steps_without_improvement:
            # Perturb current best or restart fresh?
            # Restarting fresh from greedy is safer to explore new basins.
            current_mask = greedy_init()
            current_score = count_3aps(current_mask)
            T = 10.0
            steps_since_improvement = 0
            if current_score > best_overall_score:
                best_overall_score = current_score
                best_overall_mask = current_mask

    # Convert best mask to sorted list
    result = []
    for i in range(1, N + 1):
        if best_overall_mask & (1 << i):
            result.append(i)
            
    return result
