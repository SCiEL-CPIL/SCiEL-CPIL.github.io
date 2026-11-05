"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin before 60s hard limit
    
    def compute_sum(coeffs):
        s = Fraction(0, 1)
        for i, c in enumerate(coeffs, 1):
            if c != 0:
                s += Fraction(c, i)
        return s
    
    def objective(coeffs):
        s = compute_sum(coeffs)
        if s == 0:
            return None  # Invalid
        return abs(s)
    
    # Best solution found
    best_coeffs = None
    best_val = None
    
    def update_best(coeffs):
        nonlocal best_coeffs, best_val
        val = objective(coeffs)
        if val is not None:
            if best_val is None or val < best_val:
                best_val = val
                best_coeffs = list(coeffs)
    
    # Local search: try flipping each coefficient
    def local_search(coeffs, max_iter=200):
        nonlocal best_coeffs, best_val
        current = list(coeffs)
        current_val = objective(current)
        if current_val is not None:
            update_best(current)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            for i in range(30):
                if time.time() > deadline:
                    break
                old = current[i]
                # Try all three values: -1, 0, 1
                for new_val in [-1, 0, 1]:
                    if new_val == old:
                        continue
                    current[i] = new_val
                    new_obj = objective(current)
                    if new_obj is not None:
                        if current_val is None or new_obj < current_val:
                            current_val = new_obj
                            improved = True
                            update_best(current)
                            break
                    current[i] = old
                if improved:
                    break
            if not improved:
                break
        return current
    
    # Multiple restarts with different initial configurations
    random.seed(42)
    
    # Initial: try the seed configuration first
    seed_coeffs = [0]*28 + [1, -1]
    local_search(seed_coeffs, max_iter=100)
    
    # Try various random initializations
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization: mix of -1, 0, 1
        coeffs = []
        for i in range(30):
            if random.random() < 0.3:
                coeffs.append(0)
            else:
                coeffs.append(random.choice([-1, 1]))
        
        local_search(coeffs, max_iter=50)
    
    # More targeted restarts: start from near-cancellation
    # Try to balance positive and negative sums
    for r in range(100):
        if time.time() > deadline:
            break
        
        # Randomly assign signs to all 30 terms, then try to zero out some
        coeffs = [random.choice([-1, 1]) for _ in range(30)]
        local_search(coeffs, max_iter=50)
    
    # Final targeted search: try to find very small values
    # Use a greedy approach: build up terms to minimize partial sum
    for r in range(50):
        if time.time() > deadline:
            break
        
        # Start with all 1s, then flip to minimize
        coeffs = [1] * 30
        local_search(coeffs, max_iter=30)
    
    if best_coeffs is not None:
        return best_coeffs
    else:
        # Fallback: return seed
        return [0]*28 + [1, -1]
# EVOLVE-BLOCK-END
