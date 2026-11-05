"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import random

def construct():
    # Set a seed for reproducibility
    random.seed(42)
    
    best_sum = None
    best_coeff = None
    
    # Precompute reciprocals
    recips = [Fraction(1, i + 1) for i in range(30)]

    def local_search(coeff):
        current_coeff = coeff[:]
        current_sum = Fraction(0, 1)
        for i, c in enumerate(current_coeff):
            if c != 0:
                current_sum += c * recips[i]
        
        improved = True
        # Limit iterations to prevent infinite loops in theory (though finite space guarantees termination)
        max_iters = 1000
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(30):
                old_val = current_coeff[i]
                # Try all other values in {-1, 0, 1}
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    # Calculate new sum
                    delta = (new_val - old_val) * recips[i]
                    new_sum = current_sum + delta
                    
                    if new_sum == 0:
                        continue # Must be non-zero
                    
                    # Accept if strictly better
                    if abs(new_sum) < abs(current_sum):
                        current_coeff[i] = new_val
                        current_sum = new_sum
                        improved = True
                        break # Restart inner loop to check other values for this i? No, we found the best move for this i given current state. 
                        # Actually, breaking just exits the 'new_val' loop. We continue to next 'i'.
            
        # Final check
        if current_sum == 0:
            return None, None
            
        return current_coeff, abs(current_sum)

    # Try multiple random starts
    # We can also try a few greedy starts
    starts = []
    
    # Greedy start 1: Fill from high to low
    c_greedy = [0] * 30
    s_greedy = Fraction(0, 1)
    for i in range(29, -1, -1):
        best_c = 0
        best_s = s_greedy
        for c in [-1, 0, 1]:
            s_new = s_greedy + c * recips[i]
            if abs(s_new) < abs(best_s):
                best_s = s_new
                best_c = c
        c_greedy[i] = best_c
        s_greedy = best_s
    starts.append(c_greedy)
    
    # Greedy start 2: Fill from low to high
    c_greedy2 = [0] * 30
    s_greedy2 = Fraction(0, 1)
    for i in range(30):
        best_c = 0
        best_s = s_greedy2
        for c in [-1, 0, 1]:
            s_new = s_greedy2 + c * recips[i]
            if abs(s_new) < abs(best_s):
                best_s = s_new
                best_c = c
        c_greedy2[i] = best_c
        s_greedy2 = best_s
    starts.append(c_greedy2)
    
    # Random starts
    for _ in range(500):
        coeff = [0] * 30
        # Sparse random
        num_nonzero = random.randint(1, 15)
        indices = random.sample(range(30), num_nonzero)
        for idx in indices:
            coeff[idx] = random.choice([-1, 1])
        starts.append(coeff)
        
    # Also try dense random
    for _ in range(100):
        coeff = [random.choice([-1, 0, 1]) for _ in range(30)]
        starts.append(coeff)

    for init_coeff in starts:
        res_coeff, res_sum = local_search(init_coeff)
        
        if res_coeff is not None:
            if best_sum is None or res_sum < best_sum:
                best_sum = res_sum
                best_coeff = res_coeff

    # Fallback
    if best_coeff is None:
        best_coeff = [0] * 30
        best_coeff[28] = 1
        best_coeff[29] = -1
        
    return best_coeff
# EVOLVE-BLOCK-END
