import time
import random
import sys
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Identify Primes
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    # Large Primes: p > 128
    # Small/Medium Primes: p <= 128
    large_primes = [p for p in primes if p > 128]
    small_primes = [p for p in primes if p <= 128]
    
    # Initialize with Large Primes
    S = set(large_primes)
    
    # Function to check if x can be added to S
    # S contains LP and some composites.
    # We assume S is valid.
    # Check x against S.
    def can_add(x, S):
        S_list = list(S)
        n = len(S_list)
        
        # Case 1: x is divisor a.
        # x | b*c for distinct b,c in S.
        # If b,c are LP: b*c > 256 (since min LP is 131, 131*137 > 256).
        # So x cannot divide b*c if b,c are LP.
        # If one is LP, one is Composite c:
        #   b=LP, c=Comp. b*c is large.
        #   x | LP * c. Since x <= 256.
        #   If x is composite, x might divide LP*c?
        #   LP is prime. x | LP*c implies x | c (since x not divisible by LP, as LP > 128 and x < 256, x cannot be multiple of LP unless x=LP which is prime).
        #   So x | c.
        #   So if x divides any composite c in S, and there is an LP in S, we have violation?
        #   Wait, distinctness.
        #   Triple (x, LP, c).
        #   x | LP * c.
        #   If x | c, then x | LP * c. Violation.
        #   So: If x divides any composite c in S, we CANNOT add x (because we have LP in S).
        
        # If both b,c are Composites:
        #   x | c1 * c2.
        
        # Case 2: x is multiplicand.
        # a | x * c.
        # If a is LP:
        #   LP | x * c.
        #   LP is prime. LP | x or LP | c.
        #   LP > 128. x < 256. x cannot be multiple of LP (unless x=LP, prime).
        #   c < 256. c cannot be multiple of LP.
        #   So LP ∤ x*c. Safe.
        # If a is Composite c1:
        #   c1 | x * c2.
        
        # So, simplified checks for adding x to S (which has LPs):
        # 1. x must not divide any composite c in S. (Because LP exists).
        # 2. For any composite c1, c2 in S (distinct), x must not divide c1*c2? 
        #    Wait, if x | c1*c2, violation.
        # 3. For any composite c1 in S, c1 must not divide x*c2 for any c2 in S (c2 != c1).
        
        # Let's just do the full check, it's fast enough for ~100 elements.
        
        for i in range(n):
            b = S_list[i]
            for j in range(i+1, n):
                c = S_list[j]
                # Check x | b*c
                if (b * c) % x == 0:
                    return False
                # Check b | x*c
                if (x * c) % b == 0:
                    return False
                # Check c | x*b
                if (x * b) % c == 0:
                    return False
        return True

    # Add Composites
    # Sort composites? Maybe by value? Or random?
    # Let's try adding all composites in random order, multiple restarts.
    
    best_S = set(S)
    best_size = len(best_S)
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 5: break
        
        current_S = set(S)
        
        # Shuffle composites
        c_list = composites[:]
        random.shuffle(c_list)
        
        for x in c_list:
            if time.time() > deadline - 5: break
            if can_add(x, current_S):
                current_S.add(x)
        
        # Try to add Small Primes
        sp_list = small_primes[:]
        random.shuffle(sp_list)
        for p in sp_list:
            if time.time() > deadline - 5: break
            # Check if p can be added
            # p | b*c?
            # p is prime.
            # If p divides any composite c in current_S, then p | c * LP (if LP in S).
            # Also p | c1 * c2?
            # Simplest: use can_add
            if can_add(p, current_S):
                current_S.add(p)
                
        if len(current_S) > best_size:
            best_size = len(current_S)
            best_S = set(current_S)
            
    # Final local search?
    # Maybe try to swap?
    # Given time, return best_S.
    
    # Ensure types
    return list(best_S)
