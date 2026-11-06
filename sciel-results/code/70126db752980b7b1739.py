import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 30
    
    # Precompute LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
    
    # W[i] = L // (i+1)
    W = [L // (i + 1) for i in range(n)]
    
    best_abs_K = float('inf')
    best_coeffs = None
    
    # Helper to calculate K
    def calc_K(coeffs):
        return sum(c * W[i] for i, c in enumerate(coeffs))
    
    # Initialize best_coeffs with a simple non-zero sum
    # e.g., c[0]=1, others 0 -> K = W[0] = L
    # Or c[0]=1, c[1]=-1 -> K = L - L/2 = L/2
    init_coeffs = [0] * n
    init_coeffs[0] = 1
    init_coeffs[1] = -1
    best_coeffs = init_coeffs
    best_abs_K = abs(calc_K(best_coeffs))
    
    random.seed(42)
    
    # Multi-start local search
    # We will perform several restarts.
    # Each restart:
    # 1. Start from best_coeffs or random.
    # 2. Perform local search (greedy descent).
    # 3. If stuck, do a "kick" (random perturbation) and continue.
    
    max_restarts = 10000  # We will break based on time
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Decide starting point
        if restart == 0:
            current = list(best_coeffs)
        else:
            # Start from best_coeffs and perturb
            current = list(best_coeffs)
            # Perturb 2-5 random indices
            num_perturb = random.randint(2, 5)
            for _ in range(num_perturb):
                idx = random.randint(0, n - 1)
                current[idx] = random.choice([-1, 0, 1])
        
        current_K = calc_K(current)
        if current_K == 0:
            continue
            
        current_abs_K = abs(current_K)
        
        # Local Search Phase
        # We use a "steepest descent" with random tie-breaking and occasional random moves.
        # To escape local minima, we allow "sideways" moves or small uphill moves (Simulated Annealing).
        
        T = 1000.0  # Initial temperature (in terms of K difference? K is huge, so scale it)
        # K is around L ~ 10^12.
        # Let's normalize or just use raw difference.
        # Actually, let's just do greedy descent first, then random walk.
        
        improved = True
        local_iterations = 0
        max_local_iterations = 5000  # Cap per restart to prevent infinite loops
        
        while improved and local_iterations < max_local_iterations and time.time() < deadline:
            improved = False
            local_iterations += 1
            
            # Try all single moves
            # Optimization: Pre-compute potential moves?
            # Just iterate.
            
            best_move = None
            best_new_K = current_K
            best_new_abs = current_abs_K
            
            # Randomize order to avoid bias
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                old_c = current[i]
                w_i = W[i]
                
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    
                    delta = (new_c - old_c) * w_i
                    new_K = current_K + delta
                    
                    if new_K == 0:
                        continue
                    
                    new_abs = abs(new_K)
                    
                    # Acceptance criteria
                    # 1. Always accept if better.
                    # 2. Accept if equal (to explore plateau).
                    # 3. Accept if worse with prob exp(-delta_abs / T).
                    
                    if new_abs < best_new_abs:
                        best_new_abs = new_abs
                        best_new_K = new_K
                        best_move = (i, new_c)
                    elif new_abs == best_new_abs and random.random() < 0.1:
                        # Occasionally take a sideways move
                        best_move = (i, new_c)
                        best_new_K = new_K
                        best_new_abs = new_abs
                    else:
                        # Simulated Annealing: accept worse move
                        # Scale temperature
                        # Delta is in units of L.
                        # Let's use a fixed probability for small worsenings
                        # If new_abs is not much worse, accept.
                        if new_abs < best_new_abs + 100: # 100 is arbitrary small
                            if random.random() < 0.1:
                                best_move = (i, new_c)
                                best_new_K = new_K
                                best_new_abs = new_abs
            
            if best_move:
                i, new_c = best_move
                current[i] = new_c
                current_K = best_new_K
                current_abs_K = best_new_abs
                
                if current_abs_K < best_abs_K:
                    best_abs_K = current_abs_K
                    best_coeffs = list(current)
                
                improved = True
            else:
                # No improving move found.
                # Try a random kick to escape local minimum
                # Change 1-3 random coefficients
                kick_size = random.randint(1, 3)
                for _ in range(kick_size):
                    idx = random.randint(0, n - 1)
                    current[idx] = random.choice([-1, 0, 1])
                
                current_K = calc_K(current)
                if current_K == 0:
                    # Reset to best
                    current = list(best_coeffs)
                    current_K = calc_K(current)
                    current_abs_K = abs(current_K)
                    break
                current_abs_K = abs(current_K)
                
                # Allow one more round of search
                improved = True 

    return best_coeffs
