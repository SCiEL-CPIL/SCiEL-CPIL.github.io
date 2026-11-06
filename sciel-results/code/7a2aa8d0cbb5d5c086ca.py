import random
import math

def construct():
    # Calculate LCM of 1 to 30
    L = 1
    for i in range(1, 31):
        L = L * i // math.gcd(L, i)
    
    weights = [L // i for i in range(1, 31)]
    N = 30
    
    best_c = [0] * N
    best_val = float('inf')
    
    # Try to find a solution with sum 1 or -1
    # Use local search with random restarts
    
    for restart in range(1000):
        # Random initialization
        c = [random.choice([-1, 0, 1]) for _ in range(N)]
        current_sum = sum(c[i] * weights[i] for i in range(N))
        
        # If we hit 0, we must move away
        if current_sum == 0:
             # Find an index to flip to non-zero
             idx = random.randint(0, N-1)
             c[idx] = 1 if c[idx] == -1 else -1
             current_sum = sum(c[i] * weights[i] for i in range(N))
        
        if abs(current_sum) < best_val:
            best_val = abs(current_sum)
            best_c = c[:]
            if best_val == 1:
                return best_c
        
        # Local search phase
        for _ in range(1000): # iterations per restart
            if abs(current_sum) == 1:
                return c
            
            # Pick a random index to modify
            i = random.randint(0, N-1)
            old_val = c[i]
            
            # Try changing c[i] to another value in {-1, 0, 1}
            possible_vals = [v for v in [-1, 0, 1] if v != old_val]
            random.shuffle(possible_vals)
            
            improved = False
            for new_val in possible_vals:
                # Calculate delta
                delta = (new_val - old_val) * weights[i]
                new_sum = current_sum + delta
                
                # Check if new sum is 0 (invalid)
                if new_sum == 0:
                    continue
                
                # Check if improvement
                if abs(new_sum) < abs(current_sum):
                    c[i] = new_val
                    current_sum = new_sum
                    improved = True
                    break
            
            if not improved:
                # Try a random move (annealing style) or break
                # To keep it simple, let's just break if no immediate improvement
                # But we might be stuck. Let's allow a small probability of random move
                if random.random() < 0.1:
                    new_val = random.choice(possible_vals)
                    delta = (new_val - old_val) * weights[i]
                    new_sum = current_sum + delta
                    if new_sum != 0:
                        c[i] = new_val
                        current_sum = new_sum
                else:
                    break

    return best_c
