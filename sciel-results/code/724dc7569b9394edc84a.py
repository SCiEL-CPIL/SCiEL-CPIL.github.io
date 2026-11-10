import random
import numpy as np

def construct():
    n = 256
    
    # Precompute all 3-APs for counting
    all_3aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 2 + 1):
            all_3aps.append((a, a + d, a + 2 * d))

    def count_3aps(S):
        count = 0
        for ap in all_3aps:
            if ap[0] in S and ap[1] in S and ap[2] in S:
                count += 1
        return count

    def is_4ap_free(S):
        # Check all 4-APs
        for a in range(1, n + 1):
            for d in range(1, (n - a) // 3 + 1):
                if a in S and (a+d) in S and (a+2*d) in S and (a+3*d) in S:
                    return False
        return True

    def check_add_4ap(S, x):
        # Check if adding x to S creates a 4-AP
        # x is 1st
        for d in range(1, (n - x) // 3 + 1):
            if (x + d) in S and (x + 2*d) in S and (x + 3*d) in S:
                return False
        # x is 2nd
        for d in range(1, min(x, (n - x) // 2) + 1):
            if (x - d) in S and (x + d) in S and (x + 2*d) in S:
                return False
        # x is 3rd
        for d in range(1, (x + 1) // 2):
            if (x - 2*d) in S and (x - d) in S and (x + d) in S:
                return False
        # x is 4th
        for d in range(1, (x + 1) // 3 + 1):
            if (x - 3*d) in S and (x - 2*d) in S and (x - d) in S:
                return False
        return True

    random.seed(42)
    
    # Initialize S
    # Try to find a random 4-AP free set
    S = set()
    for _ in range(200):
        S = set(random.sample(range(1, n + 1), 40))
        if is_4ap_free(S):
            break
            
    if not is_4ap_free(S):
        # Greedy cleanup
        changed = True
        while changed:
            changed = False
            for a in range(1, n + 1):
                for d in range(1, (n - a) // 3 + 1):
                    if a in S and (a+d) in S and (a+2*d) in S and (a+3*d) in S:
                        S.remove(random.choice([a, a+d, a+2*d, a+3*d]))
                        changed = True
                        break
                if changed:
                    break

    current_score = count_3aps(S)
    best_S = set(S)
    best_score = current_score

    # Simulated Annealing
    T = 5.0
    cooling = 0.99
    steps = 3000
    
    for step in range(steps):
        if T < 0.001:
            break
            
        move_type = random.choice([0, 1, 2])
        new_S = set(S)
        
        if move_type == 0: # Add
            candidates = [x for x in range(1, n + 1) if x not in S]
            if not candidates:
                continue
            x = random.choice(candidates)
            if not check_add_4ap(S, x):
                continue
            new_S.add(x)
            
        elif move_type == 1: # Remove
            if not S:
                continue
            x = random.choice(list(S))
            new_S.remove(x)
            # Removal is always valid
            
        else: # Swap
            if not S:
                continue
            candidates = [x for x in range(1, n + 1) if x not in S]
            if not candidates:
                continue
            x_out = random.choice(list(S))
            x_in = random.choice(candidates)
            
            # Check if x_in creates 4-AP in S - {x_out} + {x_in}
            temp_S = set(S)
            temp_S.remove(x_out)
            if not check_add_4ap(temp_S, x_in):
                continue
                
            new_S.remove(x_out)
            new_S.add(x_in)

        new_score = count_3aps(new_S)
        delta = new_score - current_score
        
        if delta >= 0 or random.random() < np.exp(delta / T):
            S = new_S
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
                
        T *= cooling

    return sorted(best_S)
