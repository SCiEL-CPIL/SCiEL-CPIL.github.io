import math
import time
import sys
import random
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np
import scipy.sparse as sp

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s limit
    
    N = 256
    
    # Always include 1
    # Solve for 2..N
    candidates = list(range(2, N + 1))
    n = len(candidates)
    
    # 1. Try MILP for Exact Solution
    try:
        # Build conflict constraints
        # x_i + x_j <= 1 for gcd(i, j) > 1
        row_idx = []
        col_idx = []
        data = []
        
        constraint_id = 0
        # Optimization: Only check pairs where gcd > 1
        # Precompute prime factors for faster gcd checks
        def get_prime_factors(num):
            factors = set()
            d = 2
            while d * d <= num:
                while num % d == 0:
                    factors.add(d)
                    num //= d
                d += 1
            if num > 1:
                factors.add(num)
            return factors
        
        # Precompute prime factors for all candidates
        prime_factors = {}
        for i, ci in enumerate(candidates):
            prime_factors[i] = get_prime_factors(ci)
        
        for i in range(n):
            ci = candidates[i]
            pi = prime_factors[i]
            for j in range(i + 1, n):
                cj = candidates[j]
                pj = prime_factors[j]
                if pi & pj:  # If they share any prime factor, gcd > 1
                    row_idx.append(constraint_id)
                    col_idx.append(i)
                    data.append(1)
                    
                    row_idx.append(constraint_id)
                    col_idx.append(j)
                    data.append(1)
                    
                    constraint_id += 1
                    
                    # Time check during construction (should be negligible)
                    if time.time() > deadline - 5.0:
                        raise TimeoutError("Construction taking too long")
        
        if constraint_id == 0:
            # No conflicts, take all
            return [1] + candidates
        
        A_ub = sp.csr_matrix((data, (row_idx, col_idx)), shape=(constraint_id, n))
        
        # Objective: Maximize sum(i * x_i) -> Minimize -sum(i * x_i)
        c = np.array([-float(ci) for ci in candidates])
        
        # Constraints: A_ub @ x <= 1
        lincons = LinearConstraint(A_ub, lb=0, ub=1)
        
        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: x must be integer
        integrality = np.ones(n)
        
        # Solve with time limit
        remaining_time = deadline - time.time() - 2.0
        if remaining_time < 1.0:
            return _fallback_construct(N, max(0.5, deadline - time.time()))
            
        res = milp(c=c, constraints=lincons, integrality=integrality, bounds=bounds,
                   options={'time_limit': remaining_time})
        
        if time.time() > deadline:
            return _fallback_construct(N, max(0.5, deadline - time.time()))
            
        if not res.success or res.x is None:
            # If solver fails, fallback
            return _fallback_construct(N, max(0.5, deadline - time.time()))
            
        x = res.x
        # Round to nearest integer to handle floating point noise
        x = np.round(x).astype(int)
        
        # Construct result
        result = [1]  # Always include 1
        
        for i in range(n):
            if x[i] == 1:
                result.append(candidates[i])
        
        # Sanity check: Verify pairwise coprimality
        # If invalid, fallback
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if math.gcd(result[i], result[j]) != 1:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            return _fallback_construct(N, max(0.5, deadline - time.time()))
                    
        # Ensure types are Python int
        result = [int(r) for r in result]
        return result
        
    except Exception as e:
        # In case of any error, fallback
        return _fallback_construct(N, deadline - time.time())

def _fallback_construct(N, time_budget):
    """
    Fallback heuristic if ILP fails or times out.
    Uses Greedy + Local Search with random restarts.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    if deadline < 1.0:
        # Very little time left, return a simple valid set
        return [1] + [p for p in range(2, N+1) if is_prime(p)]
        
    # Precompute primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = False
    primes = [p for p in range(2, N + 1) if is_prime[p]]
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
        
    def is_coprime_set(nums):
        # nums is a set
        if len(nums) < 2:
            return True
        lst = list(nums)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    best_result = [1]
    best_sum = 1
    
    # Multi-start Greedy with more seeds
    max_seeds = 200
    for seed in range(max_seeds):
        if time.time() > deadline - 1.0:
            break
            
        random.seed(seed)
        
        # Strategy 1: Greedy by descending value
        # Strategy 2: Randomized Greedy
        # Strategy 3: Greedy by ascending value (to fill with small numbers first)
        
        if seed % 3 == 0:
            # Deterministic Greedy descending
            selected = {1}
            for n in range(N, 1, -1):
                if all(gcd(n, s) == 1 for s in selected):
                    selected.add(n)
        elif seed % 3 == 1:
            # Randomized Greedy
            selected = {1}
            candidates = list(range(N, 1, -1))
            random.shuffle(candidates)
            for n in candidates:
                if all(gcd(n, s) == 1 for s in selected):
                    selected.add(n)
        else:
            # Greedy ascending - try to maximize count first
            selected = {1}
            for n in range(2, N + 1):
                if all(gcd(n, s) == 1 for s in selected):
                    selected.add(n)
        
        # Local Search
        current_sum = sum(selected)
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            # Try to add a number
            for n in range(1, N + 1):
                if n not in selected:
                    if all(gcd(n, s) == 1 for s in selected):
                        selected.add(n)
                        current_sum += n
                        improved = True
                        break # Restart inner loop with new state
            
            if not improved:
                # Try to swap - optimized version
                selected_list = list(selected)
                for n_out in selected_list:
                    if n_out == 1:
                        continue # Don't remove 1
                    # Check if we can find a better n_in
                    # n_in must be coprime to all s in selected where s != n_out
                    rest = selected - {n_out}
                    for n_in in range(N, n_out, -1): # Only try larger numbers
                        if n_in in selected:
                            continue
                        if all(gcd(n_in, s) == 1 for s in rest):
                            # Valid swap
                            selected.remove(n_out)
                            selected.add(n_in)
                            current_sum += n_in - n_out
                            improved = True
                            break
                    if improved:
                        break
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = list(selected)
            
    # Final validation
    if not is_coprime_set(set(best_result)):
        # Emergency fallback: Just primes
        return [1] + primes
        
    return [int(x) for x in best_result]

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

if __name__ == "__main__":
    res = construct()
    print(f"Sum: {sum(res)}, Count: {len(res)}")
