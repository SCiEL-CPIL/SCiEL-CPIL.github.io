import random
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Keep 5s buffer for safety
    
    n = 30
    recip = [1.0 / i for i in range(1, n + 1)]
    
    # Initial best: seed solution
    best_coeffs = [0] * 28 + [1, -1]
    best_val = abs(sum(c * recip[i] for i, c in enumerate(best_coeffs)))
    
    def greedy_improve(coeffs, current_sum, current_val):
        improved = True
        while improved:
            improved = False
            best_delta_val = 0
            best_move = None
            for i in range(n):
                old_c = coeffs[i]
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    delta = (new_c - old_c) * recip[i]
                    new_sum = current_sum + delta
                    new_val = abs(new_sum)
                    if new_val < 1e-12:
                        new_val = 1.0  # Penalize zero
                    if new_val < current_val - best_delta_val:
                        best_delta_val = current_val - new_val
                        best_move = (i, new_c, new_sum, new_val)
            if best_move:
                i, new_c, new_sum, new_val = best_move
                coeffs[i] = new_c
                current_sum = new_sum
                current_val = new_val
                improved = True
        return coeffs, current_sum, current_val

    def simulated_annealing(coeffs, current_sum, current_val):
        T = 0.05
        T_min = 1e-8
        cooling = 0.995
        steps = 20000
        
        for _ in range(steps):
            if time.time() > deadline:
                break
            
            i = random.randint(0, n - 1)
            old_c = coeffs[i]
            new_c = random.choice([-1, 0, 1])
            if new_c == old_c:
                continue
            
            delta = (new_c - old_c) * recip[i]
            new_sum = current_sum + delta
            new_val = abs(new_sum)
            
            if new_val < 1e-12:
                new_val = 1.0
            
            if new_val < current_val:
                coeffs[i] = new_c
                current_sum = new_sum
                current_val = new_val
            else:
                if T > T_min:
                    # Accept worse move
                    prob = math.exp((current_val - new_val) / T)
                    if random.random() < prob:
                        coeffs[i] = new_c
                        current_sum = new_sum
                        current_val = new_val
            T *= cooling
            
        return coeffs, current_sum, current_val

    # Run multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        if restart == 0:
            coeffs = best_coeffs[:]
        else:
            coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        
        current_sum = sum(c * recip[i] for i, c in enumerate(coeffs))
        current_val = abs(current_sum)
        if current_val < 1e-12:
            current_val = 1.0
        
        # Greedy first
        coeffs, current_sum, current_val = greedy_improve(coeffs, current_sum, current_val)
        
        # Then SA
        coeffs, current_sum, current_val = simulated_annealing(coeffs, current_sum, current_val)
        
        if current_val < best_val:
            best_val = current_val
            best_coeffs = coeffs[:]
            
    # Final verification with Fraction
    final_frac = sum(Fraction(c, i) for i, c in enumerate(best_coeffs, 1))
    if final_frac == 0:
        # Fallback to seed
        return [0] * 28 + [1, -1]
        
    return best_coeffs
