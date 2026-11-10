"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
from math import gcd
import sys

# EVOLVE-BLOCK-START

def _abs_frac(f):
    """Return absolute value of a Fraction."""
    if f < 0:
        return -f
    return f

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin within 90s wall-clock
    
    n = 30
    # Precompute LCM of 1..30
    lcm = 1
    for i in range(1, n + 1):
        lcm = lcm * i // gcd(lcm, i)
    
    # Precompute denominators divided by lcm for fast integer arithmetic
    # coeff[i-1]/i -> coeff[i-1] * (lcm // i) / lcm
    # So sum = (sum coeff[i-1] * (lcm // i)) / lcm
    # We want to minimize |sum| subject to sum != 0
    
    # The numerator is an integer: S = sum_{i=1}^{30} c_i * (lcm // i) where c_i in {-1,0,1}
    # We want to minimize |S| / lcm, i.e., minimize |S|
    
    # lcm of 1..30 is quite large. Let's compute it.
    # 30! is huge but lcm(1..30) is manageable.
    # lcm(1..30) = 2^4 * 3^3 * 5^2 * 7 * 11 * 13 * 17 * 19 * 23 * 29
    # = 16 * 27 * 25 * 7 * 11 * 13 * 17 * 19 * 23 * 29
    
    # Let's use the approach: find coefficients such that the integer sum S is as small as possible but nonzero.
    
    # Strategy: This is essentially finding a small nonzero value in the set of all possible sums.
    # The number of possible sums is 3^30 which is ~2e14, too many to enumerate.
    
    # Alternative: Use meet-in-the-middle or a local search / simulated annealing approach.
    
    # Let's try a randomized local search approach:
    # 1. Start with random coefficients
    # 2. Try flipping individual coefficients or pairs to reduce |S|
    # 3. Keep the best found
    
    # First, let's compute the weights
    weights = [lcm // i for i in range(1, n + 1)]
    
    # The sum S = sum c_i * w_i where w_i = lcm // i
    # We want to minimize |S| subject to S != 0
    
    # Since all c_i are in {-1, 0, 1}, S is an integer.
    # The minimum possible |S| is 1 (if achievable).
    
    # Let's check if S=1 is achievable. This is a subset sum variant.
    # Actually, it's a signed subset sum: can we pick signs and zeros to get sum = 1?
    
    # Given the large weights, let's use a different approach.
    # The largest weight is lcm // 1 = lcm itself.
    # weights[0] = lcm (for i=1)
    # weights[1] = lcm // 2
    # etc.
    
    # Let me think about this differently. The problem is to find c_i in {-1,0,1} 
    # minimizing |sum c_i * w_i| where w_i = lcm//i.
    
    # Since w_1 = lcm is much larger than the sum of all other weights combined?
    # sum_{i=2}^{30} lcm//i = lcm * sum_{i=2}^{30} 1/i
    # sum_{i=2}^{30} 1/i ≈ H_30 - 1 ≈ 3.99 - 1 = 2.99
    # So sum_{i=2}^{30} w_i ≈ 2.99 * lcm
    # And w_1 = lcm
    # So w_1 < sum of the rest, so we can't just set c_1 to cancel the rest.
    
    # Actually wait, let me reconsider. The weights decrease: w_1 = lcm, w_2 = lcm/2, etc.
    # The ratio w_1 / w_2 = 2, so w_1 > w_2.
    # w_2 / w_3 = 3/2 = 1.5, so w_2 > w_3.
    # In general w_i > w_{i+1} for all i.
    
    # This looks like a balanced ternary / signed digit representation problem.
    # If the weights were powers of 2, we could represent any integer.
    # Here the weights are lcm/i, which don't have a simple multiplicative relationship.
    
    # Let me try a practical approach: use a greedy or local search method.
    
    # Approach: 
    # 1. Use a randomized approach with local optimization
    # 2. Also try some structured approaches
    
    best_abs = None
    best_coeffs = None
    
    def compute_sum(coeffs):
        s = 0
        for i in range(n):
            if coeffs[i] != 0:
                s += coeffs[i] * weights[i]
        return s
    
    # Try all-zero except one element
    # That gives |w_i| / lcm = 1/i, minimum is 1/30
    # So best so far: set c_30 = 1, rest 0, value = 1/30
    
    # Can we do better? We need |S| < 30 (since 1/30 = 30/lcm, so we need |S| < 30)
    # Actually |S|/lcm < 1/30 means |S| < lcm/30 = w_30
    # w_30 = lcm // 30
    
    # Let's try to find S with |S| < w_30 = lcm//30
    
    # Use simulated annealing / local search
    random.seed(42)
    
    def local_search(init_coeffs, max_iter=50000):
        coeffs = list(init_coeffs)
        current_sum = compute_sum(coeffs)
        current_abs = abs(current_sum)
        
        if current_abs == 0:
            # Need to make it nonzero
            # Flip a random coefficient
            idx = random.randint(0, n-1)
            coeffs[idx] = 1
            current_sum = compute_sum(coeffs)
            current_abs = abs(current_sum)
        
        best_local_abs = current_abs
        best_local_coeffs = list(coeffs)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random position to modify
            idx = random.randint(0, n - 1)
            old_val = coeffs[idx]
            
            # Try changing to the other values
            new_vals = [v for v in (-1, 0, 1) if v != old_val]
            random.shuffle(new_vals)
            
            improved = False
            for new_val in new_vals:
                delta = (new_val - old_val) * weights[idx]
                new_sum = current_sum + delta
                new_abs = abs(new_sum)
                
                if new_abs > 0 and new_abs < best_local_abs:
                    coeffs[idx] = new_val
                    current_sum = new_sum
                    current_abs = new_abs
                    best_local_abs = new_abs
                    best_local_coeffs = list(coeffs)
                    improved = True
                    break
            
            if not improved:
                # Try a random perturbation (accept worse with some probability)
                if random.random() < 0.1:
                    idx2 = random.randint(0, n - 1)
                    old_val2 = coeffs[idx2]
                    new_val2 = random.choice([v for v in (-1, 0, 1) if v != old_val2])
                    delta2 = (new_val2 - old_val2) * weights[idx2]
                    new_sum2 = current_sum + delta2
                    new_abs2 = abs(new_sum2)
                    
                    if new_abs2 > 0:
                        # Accept if not too much worse
                        if new_abs2 < current_abs * 2 or random.random() < 0.05:
                            coeffs[idx2] = new_val2
                            current_sum = new_sum2
                            current_abs = new_abs2
        
        return best_local_abs, best_local_coeffs
    
    # Start with various initial configurations
    initial_configs = []
    
    # Config 1: Only last element
    c = [0] * n
    c[n-1] = 1
    initial_configs.append(c)
    
    # Config 2: Random sparse
    for _ in range(20):
        c = [0] * n
        num_nonzero = random.randint(1, 10)
        positions = random.sample(range(n), num_nonzero)
        for p in positions:
            c[p] = random.choice([-1, 1])
        initial_configs.append(c)
    
    # Config 3: Pairs that might cancel
    for i in range(min(5, n - 1)):
        c = [0] * n
        c[i] = 1
        c[i+1] = -1
        initial_configs.append(c)
    
    # Run local search from each config
    for init_c in initial_configs:
        if time.time() > deadline:
            break
        
        local_abs, local_coeffs = local_search(init_c, max_iter=20000)
        
        if best_abs is None or local_abs < best_abs:
            best_abs = local_abs
            best_coeffs = local_coeffs
    
    # If we found a good solution, return it
    if best_coeffs is not None and best_abs > 0:
        # Verify
        final_sum = compute_sum(best_coeffs)
        if final_sum == 0:
            # Fallback
            return [0] * (n-1) + [1]
        return best_coeffs
    
    # Fallback: 1/30
    return [0] * (n-1) + [1]

# EVOLVE-BLOCK-END
