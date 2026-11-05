from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    target = Fraction(5, 31)
    best_max = float('inf')
    best_sol = None
    
    # Known best: [8, 31, 248] with max 248
    # Check if any solution has max < 248
    
    # 3-term search
    # 1/a + 1/b + 1/c = 5/31
    # (5a-31)(5b-31)(5c-31) ... no, pairwise.
    # For fixed a, 1/b + 1/c = R_a.
    # (b-c) ... (N_b b - D_b)(N_c c - D_c) = D_b^2.
    
    for a in range(7, 100):
        if time.time() > deadline: break
        R_a = target - Fraction(1, a)
        if R_a <= 0: continue
        
        # R_a = n/d
        n, d = R_a.numerator, R_a.denominator
        # 1/b + 1/c = n/d => d(b+c) = nbc => nbc - db - dc = 0
        # (nb - d)(nc - d) = d^2
        # b > a
        
        b_min = int(d / n) + 1
        if b_min < a + 1: b_min = a + 1
        b_max = int(2 * d / n)
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline: break
            # Check if (nb - d) is a divisor of d^2
            X = n * b - d
            if X <= 0: continue
            if (d * d) % X != 0: continue
            
            Y = (d * d) // X
            # nc - d = Y => c = (Y + d) / n
            if (Y + d) % n != 0: continue
            c = (Y + d) // n
            
            if c > b and c < best_max:
                best_max = c
                best_sol = [a, b, c]
                
    # 4-term search
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # a < b < c < d < best_max (initially 248)
    
    for a in range(7, 248):
        if time.time() > deadline: break
        R_a = target - Fraction(1, a)
        if R_a <= 0: continue
        
        b_min = int(1 / R_a) + 1
        if b_min < a + 1: b_min = a + 1
        b_max = int(3 / R_a)
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline: break
            R_ab = R_a - Fraction(1, b)
            if R_ab <= 0: continue
            
            c_min = int(1 / R_ab) + 1
            if c_min < b + 1: c_min = b + 1
            c_max = int(2 / R_ab)
            
            for c in range(c_min, c_max + 1):
                if time.time() > deadline: break
                R_abc = R_ab - Fraction(1, c)
                if R_abc <= 0: continue
                
                d = 1 / R_abc
                if d.denominator == 1:
                    d_int = d.numerator
                    if d_int > c and d_int < best_max:
                        best_max = d_int
                        best_sol = [a, b, c, d_int]
                        
    if best_sol is None:
        return [8, 31, 248]
        
    return best_sol
