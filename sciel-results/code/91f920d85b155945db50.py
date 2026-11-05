"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
from math import log, exp

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s hard limit
    
    N = 30
    # Precompute harmonic denominators
    inv = [Fraction(1, i) for i in range(1, N + 1)]
    
    def compute_sum(coeffs):
        s = Fraction(0, 1)
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * inv[i]
        return s
    
    def objective(coeffs):
        s = compute_sum(coeffs)
        if s == 0:
            return None  # Invalid
        return abs(s)
    
    # Best solution found
    best_coeffs = None
    best_val = None
    best_sum = None
    
    def update_best(coeffs):
        nonlocal best_coeffs, best_val, best_sum
        s = compute_sum(coeffs)
        if s == 0:
            return
        val = abs(s)
        if best_val is None or val < best_val:
            best_val = val
            best_coeffs = list(coeffs)
            best_sum = s
    
    # Local search: try flipping each coefficient to minimize
    def local_search(coeffs, max_iter=100):
        nonlocal best_coeffs, best_val, best_sum
        current = list(coeffs)
        current_sum = compute_sum(current)
        current_val = abs(current_sum) if current_sum != 0 else None
        if current_val is not None:
            update_best(current)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Try flipping each position
            for i in range(N):
                if time.time() > deadline:
                    break
                old = current[i]
                # Try the other two values
                candidates = [v for v in [-1, 0, 1] if v != old]
                best_new = None
                best_new_val = current_val
                for new_val in candidates:
                    current[i] = new_val
                    new_sum = current_sum + (new_val - old) * inv[i]
                    new_val = abs(new_sum) if new_sum != 0 else None
                    if new_val is not None:
                        if best_new_val is None or new_val < best_new_val:
                            best_new_val = new_val
                            best_new = new_val
                            best_new_sum = new_sum
                    current[i] = old
                
                if best_new is not None:
                    current[i] = best_new
                    current_sum = best_new_sum
                    current_val = best_new_val
                    improved = True
                    update_best(current)
            
            if not improved:
                break
        return current
    
    # Simulated annealing with smooth surrogate
    def simulated_annealing(initial_coeffs, iterations=2000):
        nonlocal best_coeffs, best_val, best_sum
        current = list(initial_coeffs)
        current_sum = compute_sum(current)
        current_val = abs(current_sum) if current_sum != 0 else float('inf')
        if abs(current_sum) > 0:
            update_best(current)
        
        T = 0.1
        cooling = 0.999
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            
            # Random move: change one or two positions
            if random.random() < 0.7:
                i = random.randint(0, N - 1)
                old = current[i]
                new_val = random.choice([v for v in [-1, 0, 1] if v != old])
                delta = (new_val - old) * inv[i]
                new_sum = current_sum + delta
                new_val = abs(new_sum) if new_sum != 0 else float('inf')
                
                # Acceptance criterion
                if new_val < current_val:
                    accept = True
                elif new_val == float('inf'):
                    accept = False
                else:
                    delta_obj = new_val - current_val
                    accept = random.random() < exp(-delta_obj / T)
                
                if accept:
                    current[i] = new_val
                    current_sum = new_sum
                    current_val = new_val
                    if new_val != float('inf'):
                        update_best(current)
            else:
                # Swap two positions
                i, j = random.sample(range(N), 2)
                old_i, old_j = current[i], current[j]
                new_i = random.choice([v for v in [-1, 0, 1] if v != old_i])
                new_j = random.choice([v for v in [-1, 0, 1] if v != old_j])
                delta = (new_i - old_i) * inv[i] + (new_j - old_j) * inv[j]
                new_sum = current_sum + delta
                new_val = abs(new_sum) if new_sum != 0 else float('inf')
                
                if new_val < current_val:
                    accept = True
                elif new_val == float('inf'):
                    accept = False
                else:
                    delta_obj = new_val - current_val
                    accept = random.random() < exp(-delta_obj / T)
                
                if accept:
                    current[i] = new_i
                    current[j] = new_j
                    current_sum = new_sum
                    current_val = new_val
                    if new_val != float('inf'):
                        update_best(current)
            
            T *= cooling
        
        return current
    
    random.seed(12345)
    
    # Phase 1: Local search from multiple random starts
    for r in range(50):
        if time.time() > deadline:
            break
        coeffs = [random.choice([-1, 0, 1]) for _ in range(N)]
        local_search(coeffs, max_iter=50)
    
    # Phase 2: Simulated annealing from best found and random starts
    if best_coeffs is not None:
        simulated_annealing(best_coeffs, iterations=3000)
    
    for r in range(20):
        if time.time() > deadline:
            break
        coeffs = [random.choice([-1, 0, 1]) for _ in range(N)]
        simulated_annealing(coeffs, iterations=2000)
    
    # Phase 3: More local search polish
    if best_coeffs is not None:
        local_search(best_coeffs, max_iter=200)
    
    # Phase 4: Targeted search for small sums
    # Try to find configurations with near-zero sum
    for r in range(30):
        if time.time() > deadline:
            break
        # Start with balanced signs
        coeffs = [1 if i % 2 == 0 else -1 for i in range(N)]
        # Randomly zero out some
        for i in range(N):
            if random.random() < 0.2:
                coeffs[i] = 0
        local_search(coeffs, max_iter=100)
    
    if best_coeffs is not None:
        return best_coeffs
    else:
        # Fallback
        return [0]*28 + [1, -1]
# EVOLVE-BLOCK-END
