import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    best_sq = 999
    best_coeffs = None

    def _time_check():
        return time.time() > deadline

    # Fast evaluation using sparse representation
    def _count_sq_nonzero_from_pairs(pairs):
        """pairs: list of (exponent, coeff) where coeff != 0"""
        n = len(pairs)
        # Use a dict for the square
        sq = {}
        for i in range(n):
            ei, ci = pairs[i]
            for j in range(i, n):
                ej, cj = pairs[j]
                deg = ei + ej
                val = ci * cj
                if i != j:
                    val *= 2
                sq[deg] = sq.get(deg, 0) + val
        return sum(1 for v in sq.values() if v != 0)

    def _make_coeff_list(exps, coefs):
        max_exp = max(exps)
        cl = [0] * (max_exp + 1)
        for e, c in zip(exps, coefs):
            cl[e] = c
        return cl

    # Generate candidate exponent sets
    # Focus on structures that maximize collisions in the square
    # 1. Arithmetic Progressions (APs)
    # 2. Sums of two APs
    # 3. Random sets
    
    exp_sets = []
    
    # APs with small step d
    for d in range(1, 8):
        max_start = 41 - 11 * d
        for a in range(0, max_start + 1):
            exps = tuple(a + i * d for i in range(12))
            exp_sets.append(exps)
    
    # Sums of two APs: {a+id} U {b+jd} where i=0..k-1, j=0..m-1, k+m=12
    # This creates more collisions if the two sequences interleave or overlap in sums
    for d in range(1, 6):
        for k in range(6, 12):
            m = 12 - k
            max_a = 41 - (k-1)*d
            max_b = 41 - (m-1)*d
            for a in range(0, max_a + 1):
                if _time_check():
                    break
                for b in range(0, max_b + 1):
                    exps1 = [a + i*d for i in range(k)]
                    exps2 = [b + j*d for j in range(m)]
                    exps = tuple(sorted(exps1 + exps2))
                    if len(set(exps)) == 12: # Ensure distinct exponents
                        exp_sets.append(exps)

    # Remove duplicates
    exp_sets = list(set(exp_sets))
    
    # Add some random sets
    random.seed(123)
    for _ in range(200):
        exps = tuple(sorted(random.sample(range(0, 41), 12)))
        exp_sets.append(exps)
    
    exp_sets = list(set(exp_sets))

    # For each exponent set, optimize coefficients
    # We use a local search approach on the coefficients
    # Coefficients are integers in [-64, 64], but small values are more likely to cause cancellation
    
    # Pre-define a set of "interesting" coefficient values
    # Small integers are key for cancellation
    coeff_values = list(range(-10, 11))
    
    for exps in exp_sets:
        if _time_check():
            break
            
        max_exp = max(exps)
        if max_exp + 1 > 41:
            continue
            
        # Initialize with alternating signs to encourage cancellation
        current_coefs = [1 if i % 2 == 0 else -1 for i in range(12)]
        current_pairs = [(e, c) for e, c in zip(exps, current_coefs) if c != 0]
        current_sq = _count_sq_nonzero_from_pairs(current_pairs)
        
        if current_sq < best_sq:
            best_sq = current_sq
            best_coeffs = _make_coeff_list(exps, current_coefs)
            
        # Local search: Hill climbing with random restarts
        T = 1.0
        no_improve = 0
        max_iter = 5000
        
        for _ in range(max_iter):
            if _time_check():
                break
                
            # Pick random index to mutate
            idx = random.randint(0, 11)
            old_val = current_coefs[idx]
            
            # Generate new value
            # Bias towards small integers
            new_val = random.choice(coeff_values)
            if new_val == old_val:
                continue
                
            current_coefs[idx] = new_val
            
            # Evaluate
            # Rebuild pairs only if coefficient became 0 or was 0
            # But for simplicity and speed, just rebuild all pairs
            # Optimization: only update the specific term in pairs
            # However, _count_sq_nonzero_from_pairs is O(144) which is very fast
            current_pairs = [(e, c) for e, c in zip(exps, current_coefs) if c != 0]
            new_sq = _count_sq_nonzero_from_pairs(current_pairs)
            
            delta = new_sq - current_sq
            
            if delta < 0:
                # Accept
                current_sq = new_sq
                no_improve = 0
                if new_sq < best_sq:
                    best_sq = new_sq
                    best_coeffs = _make_coeff_list(exps, current_coefs)
            else:
                # Reject or accept with probability
                if random.random() < math.exp(-delta / T):
                    current_sq = new_sq
                    no_improve += 1
                else:
                    # Revert
                    current_coefs[idx] = old_val
                    no_improve += 1
                    
            if no_improve > 100:
                # Restart with new random coefs or lower T
                T *= 0.9
                if T < 0.1:
                    break
                # Random restart
                current_coefs = [random.choice([-1, 1, 2, -2]) for _ in range(12)]
                current_pairs = [(e, c) for e, c in zip(exps, current_coefs) if c != 0]
                current_sq = _count_sq_nonzero_from_pairs(current_pairs)
                no_improve = 0
                if current_sq < best_sq:
                    best_sq = current_sq
                    best_coeffs = _make_coeff_list(exps, current_coefs)

    if best_coeffs is not None:
        return best_coeffs
    
    # Fallback
    return [1] * 12

