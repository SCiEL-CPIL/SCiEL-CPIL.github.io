import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    
    # Precomputation
    three_aps = []
    four_aps = []
    
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
            
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    elem_to_3aps = {x: [] for x in range(1, N + 1)}
    for i, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(i)
            
    elem_to_4aps = {x: [] for x in range(1, N + 1)}
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(i)

    # Helper functions
    def count_3aps(s):
        count = 0
        for ap in three_aps:
            if ap[0] in s and ap[1] in s and ap[2] in s:
                count += 1
        return count

    def delta_add(s, x):
        """Change in 3-AP count if we add x to s."""
        delta = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if x == a:
                if b in s and c in s: delta += 1
            elif x == b:
                if a in s and c in s: delta += 1
            else: # x == c
                if a in s and b in s: delta += 1
        return delta

    def delta_remove(s, x):
        """Change in 3-AP count if we remove x from s."""
        return delta_add(s, x)

    def is_valid_add(s, x):
        """Check if adding x to s creates a 4-AP."""
        for ap_idx in elem_to_4aps[x]:
            a, b, c, d = four_aps[ap_idx]
            if x == a:
                if b in s and c in s and d in s: return False
            elif x == b:
                if a in s and c in s and d in s: return False
            elif x == c:
                if a in s and b in s and d in s: return False
            else: # x == d
                if a in s and b in s and c in s: return False
        return True

    def is_valid_swap(s, x_out, x_in):
        """Check if S - {x_out} + {x_in} is valid."""
        for ap_idx in elem_to_4aps[x_in]:
            a, b, c, d = four_aps[ap_idx]
            others = []
            if x_in == a: others = [b, c, d]
            elif x_in == b: others = [a, c, d]
            elif x_in == c: others = [a, b, d]
            else: others = [a, b, c]
            
            valid = True
            for y in others:
                if y not in s:
                    valid = False
                    break
                if y == x_out:
                    valid = False
                    break
            if valid:
                return False
        return True

    # Greedy Initialization
    s = set()
    for x in range(1, N + 1):
        if is_valid_add(s, x):
            s.add(x)
            
    best_s = set(s)
    best_score = count_3aps(best_s)
    current_score = best_score

    # Simulated Annealing
    random.seed(42)
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    
    # Multi-restart strategy
    num_restarts = 3
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        # Reset current state to best found so far, or a perturbed version
        if restart == 0:
            s = set(best_s)
            current_score = best_score
        else:
            # Perturb the best solution found so far
            s = set(best_s)
            # Randomly remove some elements and re-add to escape local optimum
            for _ in range(5):
                if s:
                    x = random.choice(list(s))
                    s.remove(x)
            # Re-greedy add
            for x in range(1, N + 1):
                if x not in s and is_valid_add(s, x):
                    s.add(x)
            current_score = count_3aps(s)
            if current_score > best_score:
                best_score = current_score
                best_s = set(s)
        
        # Reset temperature for each restart
        T = 5.0
        
        restart_deadline = time.time() + time_per_restart
        
        while time.time() < restart_deadline:
            if T < min_T:
                T = min_T
                
            move_type = random.random()
            accepted = False
            delta = 0
            
            if move_type < 0.4: # Add
                candidates = [x for x in range(1, N + 1) if x not in s]
                if candidates:
                    x = random.choice(candidates)
                    if is_valid_add(s, x):
                        delta = delta_add(s, x)
                        if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)) or (delta == 0 and random.random() < 0.5):
                            s.add(x)
                            current_score += delta
                            accepted = True
                            if current_score > best_score:
                                best_score = current_score
                                best_s = set(s)
                                
            elif move_type < 0.6: # Remove
                if s:
                    x = random.choice(list(s))
                    delta = -delta_remove(s, x) # Loss is negative
                    if random.random() < math.exp(delta / T):
                        s.remove(x)
                        current_score += delta
                        accepted = True
                        
            else: # Swap
                if s:
                    x_out = random.choice(list(s))
                    candidates = [x for x in range(1, N + 1) if x not in s]
                    if candidates:
                        x_in = random.choice(candidates)
                        if is_valid_swap(s, x_out, x_in):
                            loss = delta_remove(s, x_out)
                            gain = 0
                            for ap_idx in elem_to_3aps[x_in]:
                                a, b, c = three_aps[ap_idx]
                                others = []
                                if x_in == a: others = [b, c]
                                elif x_in == b: others = [a, c]
                                else: others = [a, b]
                                
                                if all(y in s for y in others) and all(y != x_out for y in others):
                                    gain += 1
                                    
                            delta = gain - loss
                            if delta > 0 or random.random() < math.exp(delta / T):
                                s.remove(x_out)
                                s.add(x_in)
                                current_score += delta
                                accepted = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_s = set(s)
                                    
            if not accepted:
                T *= cooling
            else:
                T *= cooling
            
    return sorted(best_s)
