"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    target = Fraction(5, 31)
    
    best_max = 10**7 + 1
    best_result = None
    
    # d1 must satisfy 1/d1 < 5/31 => d1 >= 7
    for d1 in range(7, 200):
        if time.time() > deadline:
            break
        
        rem1 = target - Fraction(1, d1)
        if rem1 <= 0:
            continue
        
        # Try d2 > d1
        p1, q1 = rem1.numerator, rem1.denominator
        d2_min = max(d1 + 1, (q1 + p1 - 1) // p1)
        
        for d2 in range(d2_min, d2_min + 200):  # limited range
            if time.time() > deadline:
                break
            
            rem2 = rem1 - Fraction(1, d2)
            if rem2 < 0:
                break
            
            if rem2 == 0:
                vals = [d1, d2]
                mx = max(vals)
                if mx < best_max:
                    best_max = mx
                    best_result = vals
                continue
            
            # Try d3 > d2
            p2, q2 = rem2.numerator, rem2.denominator
            d3_min = max(d2 + 1, (q2 + p2 - 1) // p2)
            
            for d3 in range(d3_min, d3_min + 200):
                if time.time() > deadline:
                    break
                
                rem3 = rem2 - Fraction(1, d3)
                if rem3 < 0:
                    break
                
                if rem3 == 0:
                    vals = [d1, d2, d3]
                    mx = max(vals)
                    if mx < best_max:
                        best_max = mx
                        best_result = vals
                    continue
                
                # Try d4 > d3
                p3, q3 = rem3.numerator, rem3.denominator
                d4_min = max(d3 + 1, (q3 + p3 - 1) // p3)
                
                # For 4th term, just try a few
                for d4 in range(d4_min, d4_min + 100):
                    if time.time() > deadline:
                        break
                    
                    rem4 = rem3 - Fraction(1, d4)
                    if rem4 < 0:
                        break
                    
                    if rem4 == 0:
                        vals = [d1, d2, d3, d4]
                        mx = max(vals)
                        if mx < best_max:
                            best_max = mx
                            best_result = vals
                        continue
                    
                    # Try d5
                    p4, q4 = rem4.numerator, rem4.denominator
                    d5_min = max(d4 + 1, (q4 + p4 - 1) // p4)
                    
                    for d5 in range(d5_min, d5_min + 50):
                        if time.time() > deadline:
                            break
                        
                        rem5 = rem4 - Fraction(1, d5)
                        if rem5 < 0:
                            break
                        
                        if rem5 == 0:
                            vals = [d1, d2, d3, d4, d5]
                            mx = max(vals)
                            if mx < best_max:
                                best_max = mx
                                best_result = vals
                        elif rem5 > 0:
                            # Try one more level
                            p5, q5 = rem5.numerator, rem5.denominator
                            d6_min = max(d5 + 1, (q5 + p5 - 1) // p5)
                            d6_max = min(d6_min + 20, best_max - 1)
                            
                            for d6 in range(d6_min, d6_max + 1):
                                if time.time() > deadline:
                                    break
                                
                                rem6 = rem5 - Fraction(1, d6)
                                if rem6 < 0:
                                    break
                                
                                if rem6 == 0:
                                    vals = [d1, d2, d3, d4, d5, d6]
                                    mx = max(vals)
                                    if mx < best_max:
                                        best_max = mx
                                        best_result = vals
    
    if best_result is None:
        # Fallback: greedy
        r = target - Fraction(1, 8)
        values = [8]
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
    
    return best_result

# EVOLVE-BLOCK-END
