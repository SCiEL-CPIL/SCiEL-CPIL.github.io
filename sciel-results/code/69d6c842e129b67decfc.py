"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute 1/i as Fractions
    inv = [Fraction(1, i) for i in range(1, 31)]
    
    def compute_sum(coeffs):
        s = Fraction(0, 1)
        for i in range(30):
            if coeffs[i] != 0:
                s += coeffs[i] * inv[i]
        return s
    
    def abs_frac(f):
        return -f if f < 0 else f
    
    def try_local_search(start_coeffs, max_steps=500):
        best_coeffs = start_coeffs[:]
        best_sum = compute_sum(best_coeffs)
        best_abs = abs_frac(best_sum)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            improved = False
            # Try flipping each coefficient
            for i in range(30):
                old_val = best_coeffs[i]
                # Try all three values
                for new_val in (-1, 0, 1):
                    if new_val == old_val:
                        continue
                    best_coeffs[i] = new_val
                    new_sum = compute_sum(best_coeffs)
                    new_abs = abs_frac(new_sum)
                    if new_abs < best_abs:
                        best_abs = new_abs
                        best_sum = new_sum
                        improved = True
                    else:
                        best_coeffs[i] = old_val
                # Restore original
                best_coeffs[i] = old_val
            
            if not improved:
                break
            # If we found a better one, we need to re-set it properly
            # Actually the above loop has a bug - let me fix the approach
        
        return best_coeffs, best_sum, best_abs
    
    def local_search_v2(start_coeffs, max_steps=200):
        """Local search that properly tracks the best state."""
        current = start_coeffs[:]
        current_sum = compute_sum(current)
        current_abs = abs_frac(current_sum)
        best = current[:]
        best_sum = current_sum
        best_abs = current_abs
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            improved = False
            for i in range(30):
                old_val = current[i]
                for new_val in (-1, 0, 1):
                    if new_val == old_val:
                        continue
                    current[i] = new_val
                    new_sum = compute_sum(current)
                    new_abs = abs_frac(new_sum)
                    if new_abs < current_abs:
                        current_abs = new_abs
                        current_sum = new_sum
                        if new_abs < best_abs:
                            best_abs = new_abs
                            best_sum = new_sum
                            best = current[:]
                        improved = True
                    else:
                        current[i] = old_val
                current[i] = old_val
            
            if not improved:
                break
        
        return best, best_sum, best_abs
    
    # Start with seed solution
    best_overall = [0] * 28 + [1, -1]
    best_overall_sum = compute_sum(best_overall)
    best_overall_abs = abs_frac(best_overall_sum)
    
    # Try local search from seed
    c, s, a = local_search_v2(best_overall[:])
    if a < best_overall_abs:
        best_overall = c
        best_overall_sum = s
        best_overall_abs = a
    
    # Multi-start: try various random and structured starts
    random.seed(42)
    
    # Generate diverse starting points
    starts = []
    
    # All zeros except one pair
    for i in range(29):
        c = [0]*30
        c[i] = 1
        c[i+1] = -1
        starts.append(c)
    
    # Random starts with few nonzeros
    for _ in range(200):
        c = [0]*30
        n_nonzero = random.randint(2, 8)
        idx = random.sample(range(30), n_nonzero)
        for i in idx:
            c[i] = random.choice([-1, 1])
        starts.append(c)
    
    # Random starts with more nonzeros
    for _ in range(100):
        c = [0]*30
        for i in range(30):
            r = random.random()
            if r < 0.3:
                c[i] = random.choice([-1, 1])
        if sum(1 for x in c if x != 0) < 2:
            continue
        starts.append(c)
    
    # Structured: alternating patterns
    for period in [2, 3, 4, 5, 6]:
        for offset in range(period):
            c = [0]*30
            for i in range(30):
                if (i - offset) % period == 0:
                    c[i] = 1
                elif (i - offset) % period == 1:
                    c[i] = -1
            starts.append(c)
    
    # Process each start
    for idx, start in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Quick check: if sum is zero, skip
        s = compute_sum(start)
        if s == 0:
            continue
        a = abs_frac(s)
        
        # If already better, update
        if a < best_overall_abs:
            best_overall = start[:]
            best_overall_sum = s
            best_overall_abs = a
        
        # Run local search
        c, s, a = local_search_v2(start[:])
        if a < best_overall_abs:
            best_overall = c
            best_overall_sum = s
            best_overall_abs = a
    
    # Final check
    if best_overall_sum == 0:
        # Should not happen, but fallback
        return [0]*28 + [1, -1]
    
    return best_overall
# EVOLVE-BLOCK-END
