import math
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 80.0  # Safety margin before 90s kill
    N = 256

    # 1. Preprocessing
    # Sieve to find primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Precompute distinct prime factors for each number 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        x = i
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                prime_factors[i].add(p)
                while x % p == 0:
                    x //= p
        if x > 1:
            prime_factors[i].add(x)
    prime_factors[1] = set()
    
    # Precompute conflict: for each number, which other numbers conflict
    # Two numbers conflict if they share a prime factor
    # We'll use prime_factors for this
    
    def is_valid_solution(sol_list):
        """Check types, range, distinctness, and coprimality"""
        if not isinstance(sol_list, (list, tuple)):
            return False
        n = len(sol_list)
        if n == 0:
            return False
        seen = set()
        for x in sol_list:
            if not isinstance(x, int) or isinstance(x, bool):
                return False
            if x < 1 or x > N:
                return False
            if x in seen:
                return False
            seen.add(x)
        # Check pairwise coprimality
        lst = list(sol_list)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    # 2. Attempt Exact Solution via MILP
    best_result = None
    milp_optimal = False
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Variables x_n for n in 2..N (we'll add 1 separately)
        indices = list(range(2, N + 1))
        num_vars = len(indices)
        num_primes = len(primes)
        
        # A matrix: num_primes x num_vars
        # A[p_idx, v_idx] = 1 if primes[p_idx] divides indices[v_idx]
        A = np.zeros((num_primes, num_vars), dtype=np.int8)
        for idx, p in enumerate(primes):
            for v_idx, n in enumerate(indices):
                if n % p == 0:
                    A[idx, v_idx] = 1
        
        # Objective: Maximize sum(n * x_n) -> Minimize sum(-n * x_n)
        c = np.zeros(num_vars, dtype=np.float64)
        for v_idx, n in enumerate(indices):
            c[v_idx] = -float(n)
            
        # Constraints: For each prime, sum of x_n for multiples <= 1
        cons = LinearConstraint(A, lb=0, ub=1)
        
        # Bounds: 0 <= x_n <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: all x_n are integers
        integrality = np.ones(num_vars, dtype=np.int8)
        
        # Time limit: Use most of the budget, leave 10s for local search and validation
        time_left = max(1.0, deadline - time.time() - 10.0)
        
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds, 
                   options={'time_limit': time_left})
        
        if res.x is not None:
            # res.x corresponds to indices 2..N
            sol_set = {1}  # Always include 1
            for v_idx, val in enumerate(res.x):
                if val > 0.5:
                    sol_set.add(indices[v_idx])
            
            # Validate
            if is_valid_solution(list(sol_set)):
                best_result = list(sol_set)
                if res.status == 0:  # Optimal
                    milp_optimal = True
                    
    except Exception as e:
        # If MILP fails, we will rely on heuristics
        pass

    # 3. Local Search / Polish
    if not milp_optimal:
        # Initialize best_set
        if best_result is not None:
            best_set = set(best_result)
        else:
            # Greedy initialization: sort by value descending, add if no conflict
            best_set = {1}
            candidates = sorted(range(N, 1, -1))
            used_primes = set()
            for n in candidates:
                if not (prime_factors[n] & used_primes):
                    best_set.add(n)
                    used_primes |= prime_factors[n]
        
        best_sum = sum(best_set)
        used_primes = set()
        for s in best_set:
            used_primes |= prime_factors[s]
        
        # Precompute for faster lookups
        # For each number, store its prime factors as a frozenset for hashing
        pf_frozen = [frozenset(prime_factors[i]) for i in range(N + 1)]
        
        unchanged = 0
        max_unchanged = 20
        while unchanged < max_unchanged and time.time() < deadline - 5.0:
            improved = False
            
            # Move type 1: Add a single number (if possible)
            for n in range(N, 1, -1):
                if n not in best_set and len(pf_frozen[n] & used_primes) == 0:
                    best_set.add(n)
                    used_primes |= pf_frozen[n]
                    best_sum += n
                    improved = True
                    break
            
            # Move type 2: 1-for-1 swap
            if not improved:
                best_gain = 0
                best_move = None
                for n_out in list(best_set):
                    if n_out == 1:
                        continue
                    freed = pf_frozen[n_out]
                    remaining = used_primes - freed
                    for n_in in range(N, 1, -1):
                        if n_in not in best_set and len(pf_frozen[n_in] & remaining) == 0:
                            gain = n_in - n_out
                            if gain > best_gain:
                                best_gain = gain
                                best_move = (n_out, n_in)
                            if n_in <= n_out:
                                break  # Can't do better with smaller n_in
                if best_move is not None:
                    n_out, n_in = best_move
                    best_set.remove(n_out)
                    best_set.add(n_in)
                    used_primes = (used_primes - pf_frozen[n_out]) | pf_frozen[n_in]
                    best_sum += best_gain
                    improved = True
            
            # Move type 3: 2-for-1 swap (remove two, add one)
            if not improved:
                best_gain = 0
                best_move = None
                lst = [x for x in best_set if x != 1]
                for i in range(len(lst)):
                    for j in range(i + 1, len(lst)):
                        a, b = lst[i], lst[j]
                        freed = pf_frozen[a] | pf_frozen[b]
                        remaining = used_primes - freed
                        for n_in in range(N, 1, -1):
                            if n_in not in best_set and len(pf_frozen[n_in] & remaining) == 0:
                                gain = n_in - a - b
                                if gain > best_gain:
                                    best_gain = gain
                                    best_move = (a, b, n_in)
                                if n_in <= a + b:
                                    break
                if best_move is not None:
                    a, b, n_in = best_move
                    best_set.remove(a)
                    best_set.remove(b)
                    best_set.add(n_in)
                    used_primes = (used_primes - pf_frozen[a] - pf_frozen[b]) | pf_frozen[n_in]
                    best_sum += best_gain
                    improved = True
            
            # Move type 4: 1-for-2 swap (remove one, add two)
            if not improved:
                best_gain = 0
                best_move = None
                for n_out in list(best_set):
                    if n_out == 1:
                        continue
                    freed = pf_frozen[n_out]
                    remaining = used_primes - freed
                    # Find pair (n_in1, n_in2) that are both disjoint from remaining and each other
                    for n_in1 in range(N, 1, -1):
                        if n_in1 not in best_set and len(pf_frozen[n_in1] & remaining) == 0:
                            blocked_for_in2 = remaining | pf_frozen[n_in1]
                            for n_in2 in range(N, 1, -1):
                                if n_in2 != n_in1 and n_in2 not in best_set and len(pf_frozen[n_in2] & blocked_for_in2) == 0:
                                    gain = n_in1 + n_in2 - n_out
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_move = (n_out, n_in1, n_in2)
                                    if n_in1 + n_in2 <= n_out:
                                        break
                            if n_in1 * 2 <= n_out:
                                break
                if best_move is not None:
                    n_out, n_in1, n_in2 = best_move
                    best_set.remove(n_out)
                    best_set.add(n_in1)
                    best_set.add(n_in2)
                    used_primes = (used_primes - pf_frozen[n_out]) | pf_frozen[n_in1] | pf_frozen[n_in2]
                    best_sum += best_gain
                    improved = True
            
            if not improved:
                unchanged += 1
                
        if best_result is None:
            best_result = list(best_set)
        else:
            if best_sum > sum(best_result):
                best_result = list(best_set)

    # Final: ensure 1 is included
    if best_result is None:
        best_result = [1]
    else:
        if 1 not in best_result:
            best_result.append(1)
    
    # Final Validation
    if not is_valid_solution(best_result):
        # Fallback: greedy
        best_result = [1]
        used_primes = set()
        for n in range(N, 1, -1):
            if not (prime_factors[n] & used_primes):
                best_result.append(n)
                used_primes |= prime_factors[n]
        if not is_valid_solution(best_result):
            best_result = [1]

    # Convert to list of Python ints
    return [int(x) for x in best_result]
