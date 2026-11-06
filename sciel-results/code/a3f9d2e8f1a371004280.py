import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    time_limit = 58.0  # Leave 2s safety margin
    
    # 1. Calculate LCM of 1 to 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    D = reduce(lcm, range(1, 31))
    
    # Weights for each coefficient
    # w[i] corresponds to the term c_{i+1} / (i+1)
    # So c_1/1 uses w[0] = D/1
    # c_30/30 uses w[29] = D/30
    weights = [D // i for i in range(1, 31)]
    
    def evaluate(coeffs):
        """Returns the numerator N for the given coefficients."""
        n = 0
        for i in range(30):
            if coeffs[i] != 0:
                n += coeffs[i] * weights[i]
        return n

    def local_search(coeffs):
        """
        Perform local search to minimize |N|.
        Returns the improved coeffs.
        """
        current_n = evaluate(coeffs)
        if current_n == 0:
            return coeffs # Should not happen if we avoid 0, but safe
        
        improved = True
        while improved:
            improved = False
            # Shuffle indices to avoid bias in order
            indices = list(range(30))
            random.shuffle(indices)
            
            for i in indices:
                old_val = coeffs[i]
                current_abs = abs(current_n)
                
                # Try other values
                for new_val in (-1, 0, 1):
                    if new_val == old_val:
                        continue
                    
                    # Calculate new N
                    # N_new = N_old + (new_val - old_val) * w_i
                    delta = (new_val - old_val) * weights[i]
                    new_n = current_n + delta
                    
                    # Constraint: sum must be nonzero
                    if new_n == 0:
                        continue
                    
                    new_abs = abs(new_n)
                    
                    if new_abs < current_abs:
                        # Accept move
                        coeffs[i] = new_val
                        current_n = new_n
                        current_abs = new_abs
                        improved = True
                        break # Restart the loop with new current_n
                
                # Optional: Early exit if we hit a very small value?
                # But we want to keep going to minimize further.
                
        return coeffs

    best_coeffs = None
    best_abs_n = float('inf')
    
    # Strategy: Multiple random restarts
    # Each restart: random init -> local search
    # We can also try "greedy" init: start with all 0s and add terms that reduce the sum?
    # Or start with random and polish.
    
    # Let's try a mix.
    # First, a few structured starts.
    # Start 1: All 1s
    # Start 2: Alternating 1, -1
    # Start 3: Random
    
    initial_strategies = [
        [1] * 30,
        [1 if i % 2 == 0 else -1 for i in range(30)],
        [random.choice([-1, 0, 1]) for _ in range(30)]
    ]
    
    # Ensure initial state is not zero sum (though local search will handle it, 
    # it's better to start non-zero to avoid trivial 0 moves if we get stuck)
    # Actually, if start is 0, local search won't move because any move increases |N|.
    # So we must ensure initial N != 0.
    
    for strat in initial_strategies:
        if time.time() - start_time > time_limit:
            break
            
        # Check if initial sum is 0
        init_n = evaluate(strat)
        if init_n == 0:
            # Perturb slightly
            # Flip first non-zero? Or just set c_1 = 1?
            # If all 0, set c_1 = 1.
            # If sum 0 with non-zeros, flip one.
            if all(c == 0 for c in strat):
                strat[0] = 1
            else:
                # Find a non-zero and flip it
                for i in range(30):
                    if strat[i] != 0:
                        strat[i] = -strat[i]
                        break
            init_n = evaluate(strat)
            
        # Local search
        result = local_search(strat)
        res_n = evaluate(result)
        
        if res_n != 0 and abs(res_n) < best_abs_n:
            best_abs_n = abs(res_n)
            best_coeffs = result.copy()
            
        # If we found a very small value, we might spend more time polishing
        # but for now, just continue with random restarts.
        
    # Continue with random restarts until time is up
    while time.time() - start_time < time_limit:
        # Random init
        init = [random.choice([-1, 0, 1]) for _ in range(30)]
        init_n = evaluate(init)
        
        # If 0, perturb
        if init_n == 0:
            # Try to find a move that makes it non-zero
            # Just flip a random non-zero, or set one to 1 if all 0
            if all(c == 0 for c in init):
                init[0] = 1
            else:
                idx = random.randint(0, 29)
                if init[idx] == 0:
                    init[idx] = 1
                else:
                    init[idx] = -init[idx]
            init_n = evaluate(init)
            
        result = local_search(init)
        res_n = evaluate(result)
        
        if res_n != 0 and abs(res_n) < best_abs_n:
            best_abs_n = abs(res_n)
            best_coeffs = result.copy()
            
            # If we get extremely small, we can try to intensify search around this solution
            # But for simplicity, we rely on the loop.
            # We can add a check: if best_abs_n < 100, run a more intensive search?
            # The local search above is already quite thorough (full pass until convergence).
            
    # If we didn't find anything (should not happen), fallback to seed
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
        
    return best_coeffs

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    D = 2329089562800
    n = 0
    for i, c in enumerate(res):
        if c != 0:
            n += c * (D // (i+1))
    print(f"Time: {t1-t0:.2f}s")
    print(f"Result: {res}")
    print(f"Numerator: {n}, Abs: {abs(n)}")
    print(f"Value: {abs(n)/D}")
    print(f"Score: {D/abs(n)}")
