import random
import math
import time
from itertools import combinations

def construct():
    def can_add(S, x):
        n = len(S)
        if n < 2:
            return True
            
        # Check 1: x divides sum of any 3 distinct in S
        # We can use a set of sums? No, just iterate.
        # Optimization: If x is large, sums are small? No.
        for i in range(n):
            si = S[i]
            for j in range(i+1, n):
                sj = S[j]
                for k in range(j+1, n):
                    if (si + S[j] + S[k]) % x == 0:
                        return False
                        
        # Check 2: a in S divides (u + v + x) for u,v in S\{a}
        # Iterate a
        for i, a in enumerate(S):
            # We need pairs u, v from S excluding a
            # To avoid checking pairs involving a, we can just check all pairs and skip if index is i
            for j in range(n):
                if j == i: continue
                uj = S[j]
                for k in range(j+1, n):
                    if k == i: continue
                    uk = S[k]
                    if (uj + uk + x) % a == 0:
                        return False
        return True

    # Greedy initialization
    # Sort candidates? 
    # Numbers that are "hard" to place (small numbers, divisors of many things) should be placed later?
    # Or "easy" to place (large numbers) first?
    # Large numbers divide fewer sums. So greedy large-to-small is good.
    
    greedy_selected = []
    # We might want to try multiple greedy orders?
    # For now, stick to 192..1
    
    for x in range(192, 0, -1):
        if can_add(greedy_selected, x):
            greedy_selected.append(x)
            
    best = list(greedy_selected)
    best_size = len(best)
    
    # Simulated Annealing
    current = list(greedy_selected)
    current_size = len(current)
    
    T = 10.0
    cooling = 0.999
    iterations = 2000 # Reduce to ensure speed
    
    random.seed(42)
    
    start_time = time.time()
    
    for it in range(iterations):
        if time.time() - start_time > 50:
            break
            
        move_type = random.random()
        
        if move_type < 0.3:
            # Add
            candidates = [x for x in range(1, 193) if x not in current]
            if candidates:
                x = random.choice(candidates)
                if can_add(current, x):
                    new_state = current + [x]
                    if len(new_state) > current_size:
                        current = new_state
                        current_size = len(current)
                        if current_size > best_size:
                            best = list(current)
                            best_size = current_size
                    else:
                        # Same size? No, add increases size.
                        # Wait, if we add, size increases by 1.
                        # So always accept if valid?
                        # Yes, larger size is better.
                        current = new_state
                        current_size = len(current)
                        if current_size > best_size:
                            best = list(current)
                            best_size = current_size
        elif move_type < 0.6:
            # Remove
            if current:
                idx = random.randint(0, len(current) - 1)
                removed = current[idx]
                new_state = current[:idx] + current[idx+1:]
                # Accept with prob exp(-1/T)
                if random.random() < math.exp(-1 / T):
                    current = new_state
                    current_size = len(current)
        else:
            # Swap
            if current:
                idx = random.randint(0, len(current) - 1)
                removed = current[idx]
                temp = current[:idx] + current[idx+1:]
                candidates = [x for x in range(1, 193) if x not in temp]
                if candidates:
                    x = random.choice(candidates)
                    if can_add(temp, x):
                        new_state = temp + [x]
                        # Size is same
                        # Accept with prob 1 (delta 0) or some prob?
                        # Usually swap is neutral. We can just accept to explore.
                        current = new_state
                        # current_size unchanged
                        if current_size > best_size:
                            best = list(current)
                            best_size = current_size
                            
        T *= cooling
        
    return best
