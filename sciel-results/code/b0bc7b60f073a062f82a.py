"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def construct():
    """
    Find 1..256 distinct denominators in [1,10000000] summing reciprocally to 5/31,
    minimizing the largest denominator.
    
    Strategy: Exhaustive search over 2-term solutions using the number-theoretic
    characterization (5a-31)(5b-31) = 31^2, then 3-term solutions via divisor
    enumeration of the remaining fraction's numerator.
    """
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    target = Fraction(5, 31)
    best_max = float('inf')
    best_solution = None
    
    # Phase 1: Exhaustive 2-term search
    # 1/a + 1/b = 5/31 => 31(a+b) = 5ab => (5a-31)(5b-31) = 961
    # 961 = 31^2, so divisors d of 961: 1, 31, 961
    # For each divisor d: 5a - 31 = d, 5b - 31 = 961/d
    # => a = (d + 31) / 5, b = (961/d + 31) / 5
    # Need d+31 ≡ 0 (mod 5) and 961/d + 31 ≡ 0 (mod 5)
    
    divisors_961 = [1, 31, 961]
    for d in divisors_961:
        if time.time() > deadline:
            break
        if (d + 31) % 5 != 0:
            continue
        e = 961 // d
        if (e + 31) % 5 != 0:
            continue
        a = (d + 31) // 5
        b = (e + 31) // 5
        if a < 1 or b < 1:
            continue
        if a == b:
            continue
        if a > 10000000 or b > 10000000:
            continue
        # Verify
        if Fraction(1, a) + Fraction(1, b) == target:
            current_max = max(a, b)
            if current_max < best_max:
                best_max = current_max
                best_solution = sorted([a, b])
    
    # Phase 2: Exhaustive 3-term search using divisor enumeration
    # 1/a + 1/b + 1/c = 5/31
    # For fixed a, b: 1/c = 5/31 - 1/a - 1/b
    # => c = 1 / (5/31 - 1/a - 1/b)
    # For this to be an integer, (5/31 - 1/a - 1/b) must be the reciprocal of an integer
    
    # Better approach: 1/b + 1/c = 5/31 - 1/a = (5a - 31) / (31a)
    # Let r = (5a - 31) / (31a) in lowest terms
    # Then 1/b + 1/c = r => (b+c)/(bc) = r => bc*r = b + c
    # => (rb - 1)(rc - 1) = 1
    # => rb - 1 divides 1, so rb - 1 ∈ {1, -1}
    # => b = 2/r or b = 0 (invalid)
    # Wait, let me redo this.
    # (rb - 1)(rc - 1) = 1
    # Let u = rb - 1, v = rc - 1, uv = 1
    # u, v are integers (since r is rational and b, c are integers)
    # Actually, u = rb - 1 = (5a-31)b/(31a) - 1 = ((5a-31)b - 31a) / (31a)
    # For u to be an integer, 31a must divide ((5a-31)b - 31a)
    # This is getting complex. Let me use a different approach.
    
    # For fixed a, we need 1/b + 1/c = (5a-31)/(31a)
    # Let p = 5a - 31, q = 31a
    # Reduce: g = gcd(p, q), p' = p/g, q' = q/g
    # Then 1/b + 1/c = p'/q'
    # => q'(b+c) = p'bc
    # => (p'b - q')(p'c - q') = q'^2
    # So p'b - q' must be a divisor of q'^2
    
    for a in range(7, 20):  # a must be at least 7 since 1/6 > 5/31
        if time.time() > deadline:
            break
        
        p = 5 * a - 31
        if p <= 0:
            continue
        
        q = 31 * a
        g = gcd(p, q)
        p_prime = p // g
        q_prime = q // g
        
        # Find all divisors of q_prime^2
        q2 = q_prime * q_prime
        
        # Generate divisors of q2
        divisors = set()
        for i in range(1, int(q2**0.5) + 1):
            if q2 % i == 0:
                divisors.add(i)
                divisors.add(q2 // i)
        
        for d in divisors:
            if time.time() > deadline:
                break
            
            # p_prime * b - q_prime = d
            # => b = (d + q_prime) / p_prime
            if (d + q_prime) % p_prime != 0:
                continue
            b = (d + q_prime) // p_prime
            
            if b <= a or b > 10000000:
                continue
            
            # p_prime * c - q_prime = q2 // d
            e = q2 // d
            if (e + q_prime) % p_prime != 0:
                continue
            c = (e + q_prime) // p_prime
            
            if c <= b or c > 10000000:
                continue
            
            # Verify
            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                current_max = max(a, b, c)
                if current_max < best_max:
                    best_max = current_max
                    best_solution = sorted([a, b, c])
    
    # Phase 3: Try 4-term solutions if needed
    # For fixed a, b: 1/c + 1/d = 5/31 - 1/a - 1/b
    # Similar divisor approach
    
    if best_max > 15:  # If 2 or 3 term didn't find something small
        for a in range(7, 12):
            if time.time() > deadline:
                break
            for b in range(a + 1, 15):
                if time.time() > deadline:
                    break
                
                rem = target - Fraction(1, a) - Fraction(1, b)
                if rem <= 0:
                    continue
                
                # 1/c + 1/d = rem = p/q in lowest terms
                p = rem.numerator
                q = rem.denominator
                g = gcd(p, q)
                p_prime = p // g
                q_prime = q // g
                
                q2 = q_prime * q_prime
                
                # Find divisors of q2
                divisors = set()
                for i in range(1, int(q2**0.5) + 1):
                    if q2 % i == 0:
                        divisors.add(i)
                        divisors.add(q2 // i)
                
                for d in divisors:
                    if time.time() > deadline:
                        break
                    
                    if (d + q_prime) % p_prime != 0:
                        continue
                    c = (d + q_prime) // p_prime
                    
                    if c <= b or c > 10000000:
                        continue
                    
                    e = q2 // d
                    if (e + q_prime) % p_prime != 0:
                        continue
                    d_val = (e + q_prime) // p_prime
                    
                    if d_val <= c or d_val > 10000000:
                        continue
                    
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d_val) == target:
                        current_max = max(a, b, c, d_val)
                        if current_max < best_max:
                            best_max = current_max
                            best_solution = sorted([a, b, c, d_val])
    
    if best_solution is not None:
        return best_solution
    else:
        return [8, 31, 248]

# EVOLVE-BLOCK-END
