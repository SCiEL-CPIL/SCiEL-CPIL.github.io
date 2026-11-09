import time
import random
from math import isqrt

_N = 256

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True

def _can_add(S_set, S_list, x):
    """
    Check if x can be added to S_list without violating constraints.
    Constraint: For every distinct a, b, c in S, a does not divide b*c.
    When adding x, we must check:
    1. x does not divide a*b for any distinct a, b in S_list.
    2. For any a, b in S_list (distinct), a does not divide x*b.
    3. For any a, b in S_list (distinct), b does not divide x*a.
    
    Note: Since a, b, x are distinct, the pairs are (x, a, b), (a, x, b), (a, b, x).
    - a | x*b
    - b | x*a
    - x | a*b
    """
    n = len(S_list)
    if n < 2:
        return True
        
    # Check x | a*b
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            if (a * b) % x == 0:
                return False
                
    # Check a | x*b and b | x*a
    # a | x*b  <=> a/gcd(a,x) | b * (x/gcd(a,x)) ? No, just check (x*b) % a == 0
    for i in range(n):
        a = S_list[i]
        # a | x*b for any b != a in S_list
        # b | x*a for any b != a in S_list
        for j in range(n):
            if i == j:
                continue
            b = S_list[j]
            if (x * b) % a == 0:
                return False
            if (x * a) % b == 0:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Generate candidates
    # Prioritize primes and prime powers
    candidates = list(range(2, _N + 1))
    
    # Sort candidates to maximize chances of adding
    # Primes are "safe" to start with
    def _prime_power_score(n):
        if _is_prime(n):
            return (0, n)
        # Check if prime power
        d = 2
        is_pp = False
        while d * d <= n:
            if n % d == 0:
                temp = n
                while temp % d == 0:
                    temp //= d
                if temp == 1:
                    is_pp = True
                    break
            d += 1
        if is_pp:
            return (1, n)
        return (2, n)
        
    candidates.sort(key=_prime_power_score)
    
    S = []
    S_set = set()
    
    # Greedy add
    for x in candidates:
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if _can_add(S_set, S, x):
            S.append(x)
            S_set.add(x)
            
    # Local search: try to swap elements to increase size or allow more additions
    # Since we want to maximize size, we primarily try to add.
    # If we can't add, we might swap a "blocking" element for a "less blocking" one.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_size = len(S)
        
        # Try to add any remaining candidate
        for x in range(2, _N + 1):
            if x in S_set:
                continue
            if _can_add(S_set, S, x):
                S.append(x)
                S_set.add(x)
                improved = True
                break
        
        if improved:
            continue
            
        # If no simple add, try 1-for-1 swap
        # Remove a, try to add b, then see if we can add more
        for i, a in enumerate(list(S)):
            if time.time() > deadline:
                break
            
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            
            # Try to add any b not in S_temp
            best_b = None
            for b in range(2, _N + 1):
                if b in S_temp_set:
                    continue
                if _can_add(S_temp_set, S_temp, b):
                    # Check if this allows further additions
                    S_after = S_temp + [b]
                    S_after_set = S_temp_set | {b}
                    
                    # Try to add more
                    added = 0
                    for c in range(2, _N + 1):
                        if c in S_after_set:
                            continue
                        if _can_add(S_after_set, S_after, c):
                            S_after.append(c)
                            S_after_set.add(c)
                            added += 1
                            
                    if len(S_after) > current_size:
                        S = S_after
                        S_set = S_after_set
                        improved = True
                        break
            if improved:
                break
                
    return [int(x) for x in S]
