"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def _evaluate(coeffs):
    """Compute the signed harmonic sum as (num, den) in reduced form."""
    num = 0
    den = 1
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        d = i + 1
        g = gcd(den, d)
        den = den // g * d
        num = num * (d // g) + c * (den // d)
    g = gcd(abs(num), den)
    if g:
        num //= g
        den //= g
    return num, den

def _abs_val(num, den):
    return abs(num) / den

def construct():
    deadline = time.time() + 80
    n = 30
    
    # Default: 1/29 - 1/30 = 1/870
    best_val = float('inf')
    best_coeffs = [0] * 28 + [1, -1]
    best_num, best_den = 1, 870
    
    def update(coeffs):
        nonlocal best_val, best_coeffs, best_num, best_den
        num, den = _evaluate(coeffs)
        if num == 0:
            return
        val = abs(num) / den
        if val < best_val:
            best_val = val
            best_coeffs = list(coeffs)
            best_num = num
            best_den = den
    
    # Phase 1: Systematic enumeration of small subsets (pairs, triples, small sets)
    # Try all pairs
    for i in range(n):
        for j in range(i+1, n):
            coeffs = [0] * n
            coeffs[i] = 1
            coeffs[j] = -1
            update(coeffs)
            if time.time() > deadline:
                return best_coeffs
    
    # Try all triples
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                coeffs = [0] * n
                coeffs[i] = 1
                coeffs[j] = 1
                coeffs[k] = -1
                update(coeffs)
                coeffs = [0] * n
                coeffs[i] = 1
                coeffs[j] = -1
                coeffs[k] = -1
                update(coeffs)
                if time.time() > deadline:
                    return best_coeffs
    
    # Try all 4-element subsets with various sign patterns
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    coeffs = [0] * n
                    coeffs[i] = 1
                    coeffs[j] = 1
                    coeffs[k] = 1
                    coeffs[l] = -1
                    update(coeffs)
                    coeffs = [0] * n
                    coeffs[i] = 1
                    coeffs[j] = 1
                    coeffs[k] = -1
                    coeffs[l] = -1
                    update(coeffs)
                    coeffs = [0] * n
                    coeffs[i] = 1
                    coeffs[j] = -1
                    coeffs[k] = 1
                    coeffs[l] = -1
                    update(coeffs)
                    if time.time() > deadline:
                        return best_coeffs
    
    # Phase 2: Random local search
    random.seed(42)
    
    # Start from best found so far
    current = list(best_coeffs)
    current_num, current_den = _evaluate(current)
    current_val = _abs_val(current_num, current_den)
    
    # Also try starting from random configurations
    for _ in range(20000):
        if time.time() > deadline:
            break
        
        # Random restart occasionally
        if random.random() < 0.1:
            current = [random.choice([-1, 0, 1]) for _ in range(n)]
        else:
            # Local move: flip one coefficient
            idx = random.randint(0, n-1)
            current = list(best_coeffs)
            if random.random() < 0.5:
                current[idx] = -current[idx]
            else:
                current[idx] = random.choice([-1, 0, 1])
        
        num, den = _evaluate(current)
        if num == 0:
            continue
        val = abs(num) / den
        if val < best_val:
            best_val = val
            best_coeffs = list(current)
            best_num = num
            best_den = den
            current = list(best_coeffs)
    
    # Phase 3: More systematic search with larger subsets
    # Try 5-element subsets
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    for m in range(l+1, n):
                        coeffs = [0] * n
                        coeffs[i] = 1
                        coeffs[j] = 1
                        coeffs[k] = 1
                        coeffs[l] = 1
                        coeffs[m] = -1
                        update(coeffs)
                        coeffs = [0] * n
                        coeffs[i] = 1
                        coeffs[j] = 1
                        coeffs[k] = 1
                        coeffs[l] = -1
                        coeffs[m] = -1
                        update(coeffs)
                        coeffs = [0] * n
                        coeffs[i] = 1
                        coeffs[j] = 1
                        coeffs[k] = -1
                        coeffs[l] = 1
                        coeffs[m] = -1
                        update(coeffs)
                        if time.time() > deadline:
                            return best_coeffs
    
    # Phase 4: Final random search with more iterations
    for _ in range(50000):
        if time.time() > deadline:
            break
        
        if random.random() < 0.2:
            current = [random.choice([-1, 0, 1]) for _ in range(n)]
        else:
            current = list(best_coeffs)
            # Try multiple flips
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, n-1)
                current[idx] = random.choice([-1, 0, 1])
        
        num, den = _evaluate(current)
        if num == 0:
            continue
        val = abs(num) / den
        if val < best_val:
            best_val = val
            best_coeffs = list(current)
            best_num = num
            best_den = den
    
    return best_coeffs
# EVOLVE-BLOCK-END
