"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_range(n):
    return reduce(_lcm, range(1, n + 1), 1)

def _evaluate(coeffs, weights):
    """Compute the weighted sum of coefficients."""
    return sum(c * w for c, w in zip(coeffs, weights))

def _local_search(coeffs, weights, max_iters=5000, time_limit=None, start_time=None):
    """Perform local search to minimize |sum| while keeping it nonzero."""
    n = len(coeffs)
    current = list(coeffs)
    current_N = _evaluate(current, weights)
    
    if current_N == 0:
        # Find a valid starting point
        for i in range(n):
            current[i] = 1
            new_N = _evaluate(current, weights)
            if new_N != 0:
                current_N = new_N
                break
            current[i] = 0
        if current_N == 0:
            return None, None
    
    best_abs = abs(current_N)
    best_coeffs = list(current)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        improved = False
        # Try all single-coordinate changes
        for i in range(n):
            old_val = current[i]
            for new_val in (-1, 0, 1):
                if new_val == old_val:
                    continue
                current[i] = new_val
                new_N = current_N + (new_val - old_val) * weights[i]
                current_N = new_N
                
                if new_N == 0:
                    current[i] = old_val
                    current_N -= (new_val - old_val) * weights[i]
                    continue
                
                if abs(new_N) < best_abs:
                    best_abs = abs(new_N)
                    best_coeffs = list(current)
                    improved = True
                else:
                    current[i] = old_val
                    current_N = _evaluate(current, weights)
            
            if improved:
                break
        
        if not improved:
            break
    
    if best_abs < abs(_evaluate(best_coeffs, weights)) or _evaluate(best_coeffs, weights) == 0:
        # Verify
        final_N = _evaluate(best_coeffs, weights)
        if final_N == 0:
            return None, None
        return best_coeffs, final_N
    return best_coeffs, best_abs

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 30
    L = _lcm_range(n)
    weights = [L // (i + 1) for i in range(n)]
    
    # Trivial fallback: 1/29 - 1/30
    best_coeffs = [0] * 28 + [1, -1]
    best_N = _evaluate(best_coeffs, weights)
    best_abs_N = abs(best_N)
    
    # Phase 1: Try structured initial solutions
    # Pattern 1: alternating signs
    for pattern in range(10):
        if time.time() > deadline - 30:
            break
        coeffs = []
        for i in range(n):
            if i % 2 == 0:
                coeffs.append(1)
            else:
                coeffs.append(-1)
        # Add some randomness
        for _ in range(pattern * 3):
            idx = random.randint(0, n - 1)
            coeffs[idx] = random.choice([-1, 0, 1])
        
        N = _evaluate(coeffs, weights)
        if N != 0:
            result_coeffs, result_N = _local_search(coeffs, weights, max_iters=2000, 
                                                      time_limit=deadline - 10, 
                                                      start_time=start_time)
            if result_coeffs is not None and abs(result_N) < best_abs_N:
                best_coeffs = result_coeffs
                best_N = result_N
                best_abs_N = abs(result_N)
    
    # Phase 2: Multiple random restarts with local search
    random.seed(42)
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Generate random initial solution
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        
        N = _evaluate(coeffs, weights)
        if N == 0:
            # Perturb to get nonzero
            idx = random.randint(0, n - 1)
            coeffs[idx] = 1 if coeffs[idx] != 1 else -1
            N = _evaluate(coeffs, weights)
            if N == 0:
                continue
        
        # Local search
        result_coeffs, result_N = _local_search(coeffs, weights, max_iters=3000,
                                                  time_limit=deadline - 5,
                                                  start_time=start_time)
        
        if result_coeffs is not None and abs(result_N) < best_abs_N:
            best_coeffs = result_coeffs
            best_N = result_N
            best_abs_N = abs(result_N)
            
            # If we got very close to optimal, do more refinement
            if best_abs_N <= L // 1000:
                # Try more aggressive local search
                result_coeffs2, result_N2 = _local_search(best_coeffs, weights, 
                                                            max_iters=10000,
                                                            time_limit=deadline - 3,
                                                            start_time=start_time)
                if result_coeffs2 is not None and abs(result_N2) < best_abs_N:
                    best_coeffs = result_coeffs2
                    best_N = result_N2
                    best_abs_N = abs(result_N2)
    
    # Phase 3: Final refinement with 2-coordinate moves
    if time.time() < deadline - 3:
        current = list(best_coeffs)
        current_N = _evaluate(current, weights)
        current_abs = abs(current_N)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(n):
                if time.time() > deadline - 2:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 2:
                        break
                    old_i, old_j = current[i], current[j]
                    for new_i in (-1, 0, 1):
                        if new_i == old_i:
                            continue
                        for new_j in (-1, 0, 1):
                            if new_j == old_j:
                                continue
                            current[i] = new_i
                            current[j] = new_j
                            new_N = current_N + (new_i - old_i) * weights[i] + (new_j - old_j) * weights[j]
                            
                            if new_N == 0:
                                current[i] = old_i
                                current[j] = old_j
                                continue
                            
                            if abs(new_N) < current_abs:
                                current_abs = abs(new_N)
                                current_N = new_N
                                improved = True
                            else:
                                current[i] = old_i
                                current[j] = old_j
                                current_N -= (new_j - old_j) * weights[j]  # Restore
                                current_N -= (new_i - old_i) * weights[i]  # Restore
            
            if not improved:
                break
        
        if current_abs < best_abs_N:
            best_coeffs = current
            best_N = current_N
            best_abs_N = current_abs
    
    # Final verification
    final_N = _evaluate(best_coeffs, weights)
    if final_N == 0:
        # Fallback
        best_coeffs = [0] * 28 + [1, -1]
    
    return best_coeffs

# EVOLVE-BLOCK-END
