"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 50
    
    def verify(denoms):
        if not denoms or len(denoms) > 256:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        s = Fraction(0)
        for d in denoms:
            if d < 5 or d > 10000000:
                return False
            s += Fraction(1, d)
        return s == Fraction(1)
    
    # Known 12-term solution
    fallback12 = [5, 6, 7, 10, 14, 21, 30, 42, 70, 105, 140, 210]
    
    def search_k(k, max_denom=150):
        best = None
        found = [False]
        
        def rec(denoms, curr, min_d, rem):
            if found[0]:
                return
            if time.time() > deadline:
                return
            if rem == 0:
                if curr == Fraction(1):
                    best = denoms[:]
                    found[0] = True
                return
            if curr > Fraction(1):
                return
            
            max_add = Fraction(0)
            for i in range(rem):
                max_add += Fraction(1, min_d + i)
            if curr + max_add < Fraction(1):
                return
            
            diff = Fraction(1) - curr
            if diff <= 0:
                return
            
            d_lo = max(min_d, int(1/float(diff)) + 1)
            if d_lo < min_d:
                d_lo = min_d
            
            d_hi = int(rem / float(diff))
            if d_hi < d_lo:
                return
            d_hi = min(d_hi, max_denom)
            
            for d in range(d_lo, d_hi + 1):
                ns = curr + Fraction(1, d)
                if ns > Fraction(1):
                    break
                denoms.append(d)
                rec(denoms, ns, d + 1, rem - 1)
                denoms.pop()
                if found[0]:
                    return
        
        rec([], Fraction(0), 5, k)
        return best
    
    for k in range(8, 13):
        if time.time() > deadline - 3:
            break
        result = search_k(k)
        if result and verify(result):
            return result
    
    if verify(fallback12):
        return fallback12
    
    raise RuntimeError("no valid representation found")
# EVOLVE-BLOCK-END
