import time
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    MAX_N = 256
    
    # 1. Find all primes up to MAX_N
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(MAX_N)
    
    # Precompute prime factors for each number 2..256
    prime_factors = {}
    for n in range(2, MAX_N + 1):
        pf = set()
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                pf.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            pf.add(temp)
        prime_factors[n] = pf
    
    # 2. ILP approach
    ns = list(range(2, MAX_N + 1))
    num_vars = len(ns)
    
    c = [-n for n in ns]
    
    A_rows = []
    for p in primes:
        row = [0] * num_vars
        for i, n in enumerate(ns):
            if n % p == 0:
                row[i] = 1
        A_rows.append(row)
    
    integrality = [1] * num_vars
    lb = [0] * num_vars
    ub = [1] * num_vars
    
    selected = []
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        constraint = LinearConstraint(A_rows, lb=[0]*len(primes), ub=[1]*len(primes))
        bounds = Bounds(lb, ub)
        
        res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            selected = [ns[i] for i in range(num_vars) if res.x[i] > 0.5]
        else:
            raise Exception("ILP failed")
            
    except Exception:
        # Fallback: Greedy - pick largest numbers that are pairwise coprime
        selected = []
        for n in range(MAX_N, 1, -1):
            ok = True
            for s in selected:
                if gcd(n, s) != 1:
                    ok = False
                    break
            if ok:
                selected.append(n)
    
    # Always include 1 (coprime to everything)
    if 1 not in selected:
        selected.append(1)
    
    selected_set = set(selected)
    
    # Local search: try improvements
    # Pattern 1: Remove one, add one (swap)
    # Pattern 2: Remove one, add two
    # Pattern 3: Remove two, add three
    # Pattern 4: Add one if possible (shouldn't happen if ILP is optimal, but check)
    
    def is_valid_set(s):
        s = list(s)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if gcd(s[i], s[j]) != 1:
                    return False
        return True
    
    def get_candidates_excluding(temp_selected, excluded):
        """Get numbers not in temp_selected and not in excluded, coprime to all in temp_selected"""
        cands = []
        for n in range(2, MAX_N + 1):
            if n in temp_selected or n in excluded:
                continue
            ok = True
            for s in temp_selected:
                if gcd(n, s) != 1:
                    ok = False
                    break
            if ok:
                cands.append(n)
        return cands
    
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        
        # Pattern 1: Swap - remove one, add one
        for i in range(len(selected)):
            if time.time() >= deadline - 3.0:
                break
            removed = selected[i]
            temp_selected = set(selected) - {removed}
            cands = get_candidates_excluding(temp_selected, set())
            best_add = None
            best_val = removed
            for c in cands:
                if c > best_val:
                    best_val = c
                    best_add = c
            if best_add is not None:
                new_selected = list(temp_selected) + [best_add]
                if is_valid_set(new_selected):
                    selected = new_selected
                    selected_set = set(selected)
                    improved = True
                    break
        
        if improved:
            continue
        
        # Pattern 2: Remove one, add two
        for i in range(len(selected)):
            if time.time() >= deadline - 3.0:
                break
            removed = selected[i]
            temp_selected = set(selected) - {removed}
            cands = get_candidates_excluding(temp_selected, set())
            
            best_pair = None
            best_sum = removed
            for a_idx in range(len(cands)):
                a = cands[a_idx]
                for b_idx in range(a_idx + 1, len(cands)):
                    b = cands[b_idx]
                    if gcd(a, b) == 1:
                        s = a + b
                        if s > best_sum:
                            best_sum = s
                            best_pair = (a, b)
            
            if best_pair is not None:
                new_selected = list(temp_selected) + list(best_pair)
                if is_valid_set(new_selected):
                    selected = new_selected
                    selected_set = set(selected)
                    improved = True
                    break
    
    # Ensure we return a list of Python ints
    result = [int(x) for x in selected]
    return result
