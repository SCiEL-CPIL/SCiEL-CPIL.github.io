"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill
    
    target = Fraction(5, 31)
    
    # Known baseline solution
    best_solution = [8, 28, 1736]
    best_max_den = 1736
    
    def find_2term_solutions(rem, min_d, max_d):
        """
        Find all pairs (b, c) with min_d <= b < c <= max_d such that 1/b + 1/c = rem.
        Returns a list of tuples (b, c).
        """
        if rem <= 0:
            return []
        
        p = rem.numerator
        q = rem.denominator
        
        # Equation: 1/b + 1/c = p/q => q(b+c) = pbc => pbc - qb - qc = 0
        # (pb - q)(pc - q) = q^2
        # Let X = pb - q, Y = pc - q. Then XY = q^2.
        # Since b < c, we have pb < pc => X < Y.
        # Also X*Y = q^2 => X < q (since if X >= q, then Y <= q, so X >= Y, contradiction).
        # So we iterate X from 1 to q-1.
        
        q2 = q * q
        solutions = []
        
        # Optimization: If q is very large, this loop might be slow.
        # But for our bounds, q is manageable.
        # We can also break if X >= q, but range handles that.
        
        for X in range(1, q):
            if q2 % X != 0:
                continue
            Y = q2 // X
            # Check if Y > X (ensures b < c)
            if Y <= X:
                continue
            
            # b = (X + q) / p
            # c = (Y + q) / p
            # Must be integers
            
            num_b = X + q
            num_c = Y + q
            
            if num_b % p != 0 or num_c % p != 0:
                continue
                
            b = num_b // p
            c = num_c // p
            
            # Check constraints
            if b < min_d or c > max_d or c <= b:
                continue
                
            # Ensure distinctness and positivity (implied by min_d >= 1)
            if b >= 1 and c >= 1:
                solutions.append((b, c))
                
        return solutions

    def check_3term(K):
        """
        Check if there is a 3-term solution with max denom <= K.
        Returns the solution list or None.
        """
        # 1/a + 1/b + 1/c = 5/31
        # a < b < c
        # 3/a > 5/31 => a < 93/5 = 18.6 => a <= 18
        # 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7
        
        max_a = min(18, K)
        for a in range(7, max_a + 1):
            rem = target - Fraction(1, a)
            if rem <= 0:
                continue
            
            sols = find_2term_solutions(rem, a + 1, K)
            for (b, c) in sols:
                return [a, b, c]
                
        return None

    def check_4term(K):
        """
        Check if there is a 4-term solution with max denom <= K.
        Returns the solution list or None.
        """
        # 1/a + 1/b + 1/c + 1/d = 5/31
        # a < b < c < d
        # 4/a > 5/31 => a < 124/5 = 24.8 => a <= 24
        # a >= 7
        
        max_a = min(24, K)
        for a in range(7, max_a + 1):
            rem_a = target - Fraction(1, a)
            if rem_a <= 0:
                continue
            
            # 3/b > rem_a => b < 3/rem_a
            b_max = min(int(3 / rem_a), K)
            # Also b > a
            for b in range(a + 1, b_max + 1):
                rem_b = rem_a - Fraction(1, b)
                if rem_b <= 0:
                    break # Since b is increasing, rem_b will decrease further? 
                         # No, 1/b decreases, so rem_b increases? 
                         # rem_b = rem_a - 1/b. As b increases, 1/b decreases, so rem_b increases.
                         # Wait. We need rem_b > 0.
                         # If rem_b <= 0, then 1/b >= rem_a.
                         # For larger b, 1/b is smaller, so rem_b becomes larger (less negative or positive).
                         # So we should NOT break. We should continue.
                         # However, if rem_b <= 0, it's not a valid remainder for positive denominators.
                         # But for larger b, it might become positive.
                         # Example: rem_a = 0.1. b=10 => 1/b=0.1 => rem_b=0.
                         # b=11 => 1/b=0.09 => rem_b=0.01.
                         # So we continue.
                        
                if rem_b <= 0:
                    continue
                    
                sols = find_2term_solutions(rem_b, b + 1, K)
                for (c, d) in sols:
                    return [a, b, c, d]
                    
        return None

    def check_5term(K):
        """
        Check if there is a 5-term solution with max denom <= K.
        Returns the solution list or None.
        """
        # 5/a > 5/31 => a < 31 => a <= 30
        # a >= 7
        
        max_a = min(30, K)
        for a in range(7, max_a + 1):
            if time.time() - start_time > time_limit:
                return None
            rem_a = target - Fraction(1, a)
            if rem_a <= 0:
                continue
            
            b_max = min(int(4 / rem_a), K)
            for b in range(a + 1, b_max + 1):
                if time.time() - start_time > time_limit:
                    return None
                rem_b = rem_a - Fraction(1, b)
                if rem_b <= 0:
                    continue
                
                c_max = min(int(3 / rem_b), K)
                for c in range(b + 1, c_max + 1):
                    if time.time() - start_time > time_limit:
                        return None
                    rem_c = rem_b - Fraction(1, c)
                    if rem_c <= 0:
                        continue
                    
                    sols = find_2term_solutions(rem_c, c + 1, K)
                    for (d, e) in sols:
                        return [a, b, c, d, e]
                        
        return None

    # Binary search or linear search for minimal K
    # We know K=1736 works.
    # Let's iterate downwards from 1736 to 1.
    # Given the tight bounds on 'a', the checks are fast.
    
    best_solution_found = None
    
    for K in range(1736, 0, -1):
        if time.time() - start_time > time_limit:
            break
            
        sol = check_3term(K)
        if sol is not None:
            best_solution_found = sol
            break
            
        sol = check_4term(K)
        if sol is not None:
            best_solution_found = sol
            break
            
        sol = check_5term(K)
        if sol is not None:
            best_solution_found = sol
            break

    if best_solution_found is not None:
        return best_solution_found
    else:
        return best_solution
# EVOLVE-BLOCK-END
