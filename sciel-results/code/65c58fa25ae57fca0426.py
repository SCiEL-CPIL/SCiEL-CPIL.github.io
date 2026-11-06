import time
import random
import math

def construct():
    deadline = time.time() + 55  # Safe margin
    
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
            # x is one of a, b, c. Check if the other two are in s.
            # Since x is in the AP, we just need the other two.
            # We can't just check 'in s' for all 3 because x is not in s yet.
            # So we check the specific other two.
            if x == a:
                if b in s and c in s: delta += 1
            elif x == b:
                if a in s and c in s: delta += 1
            else: # x == c
                if a in s and b in s: delta += 1
        return delta

    def delta_remove(s, x):
        """Change in 3-AP count if we remove x from s."""
        # This is the negative of delta_add if x were not in s, 
        # BUT careful: delta_add checks if others are in s.
        # Here x IS in s. The APs lost are those where the other two are in s.
        # So the logic is identical to delta_add(s, x) but x is currently in s.
        # Wait, delta_add(s, x) assumes x NOT in s.
        # If we remove x, we lose APs where the other 2 are in s.
        # The set of other elements is the same.
        # So the count of lost APs is exactly the same calculation.
        return delta_add(s, x) 
        # Note: This works because 'in s' for the other elements is the same 
        # whether we are considering adding x to s or removing x from s (assuming x is in s for remove case, 
        # and the 'in s' check for the *other* elements is independent of x's presence for the *count* of valid APs involving x).
        # Actually, strictly speaking:
        # Add: x not in s. AP (a,b,c) containing x is formed if other 2 in s.
        # Remove: x in s. AP (a,b,c) containing x is broken if other 2 in s.
        # Yes, the logic is identical.

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
        # We need to check if x_in creates a 4-AP with (S - {x_out}).
        # A 4-AP containing x_in is invalid if the other 3 elements are in S - {x_out}.
        # This means: other 3 in S AND none of the other 3 is x_out.
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
                    # y is being removed, so it's not in the new set
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
    
    while time.time() < deadline:
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
                # We rarely want to remove unless it enables better swaps later
                # Or if T is high.
                if random.random() < math.exp(delta / T):
                    s.remove(x)
                    current_score += delta
                    accepted = True
                    # We don't update best_s here if score drops, 
                    # but we keep current_s for further exploration.
                    
        else: # Swap
            if s:
                x_out = random.choice(list(s))
                candidates = [x for x in range(1, N + 1) if x not in s]
                if candidates:
                    x_in = random.choice(candidates)
                    if is_valid_swap(s, x_out, x_in):
                        # Calculate delta
                        # Loss from x_out
                        loss = delta_remove(s, x_out)
                        # Gain from x_in (in S - {x_out})
                        # We need to calculate delta_add for x_in in (S - {x_out})
                        # delta_add checks if others are in the set.
                        # So we temporarily remove x_out? 
                        # Or we can just check manually.
                        # Let's just do it by modifying set temporarily for safety in logic, 
                        # or optimize.
                        # Optimization: 
                        # gain = 0
                        # for ap in elem_to_3aps[x_in]:
                        #   check if other 2 are in s AND other 2 != x_out
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
            T *= cooling # Cool regardless, or only on accept? Standard SA cools always.
            
    return sorted(best_s)
