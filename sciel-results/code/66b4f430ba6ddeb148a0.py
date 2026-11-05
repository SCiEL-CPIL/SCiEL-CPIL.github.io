"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    target = Fraction(5, 31)
    
    # Greedy fallback solution
    def greedy_solution(target):
        denominators = []
        r = target
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            denominators.append(d)
            r -= Fraction(1, d)
        return denominators
    
    best_solution = greedy_solution(target)
    best_max = max(best_solution)
    
    # Core mathematical approach:
    # For 1/x + 1/y = p/q (in lowest terms), we have:
    # (px - q)(py - q) = q^2
    # So px - q must be a divisor of q^2, and py - q = q^2 / (px - q)
    # Both must be positive, and x, y must be integers > some minimum
    
    def find_pairs_for_remainder(r, min_val, max_val):
        """Find all pairs (x, y) with min_val < x < y <= max_val such that 1/x + 1/y = r"""
        pairs = []
        if r <= 0:
            return pairs
        
        p, q = r.numerator, r.denominator
        q_sq = q * q
        
        # Iterate over divisors of q^2
        # For efficiency, only check divisors in a useful range
        # px - q = d, so d | q^2, and x = (d + q) / p must be integer
        # y = (q^2/d + q) / p must be integer
        
        # Generate divisors of q^2 efficiently
        if q_sq <= 10**8:
            # Find all divisors of q_sq
            divisors = set()
            i = 1
            while i * i <= q_sq:
                if q_sq % i == 0:
                    divisors.add(i)
                    divisors.add(q_sq // i)
                i += 1
            
            for d in sorted(divisors):
                # x = (d + q) / p
                if (d + q) % p != 0:
                    continue
                x = (d + q) // p
                if x <= min_val or x > max_val:
                    continue
                
                # y = (q_sq // d + q) / p
                d2 = q_sq // d
                if (d2 + q) % p != 0:
                    continue
                y = (d2 + q) // p
                if y <= x or y > max_val:
                    continue
                
                # Verify
                if Fraction(1, x) + Fraction(1, y) != r:
                    continue
                
                pairs.append((x, y))
        else:
            # For large q, use a different approach: iterate x and compute y
            # 1/x + 1/y = p/q => y = x*q / (p*x - q)
            # p*x - q must divide x*q and be positive
            min_x = max(min_val + 1, q // p + 1)
            max_x = min(max_val, 2 * q // p)
            
            for x in range(min_x, max_x + 1):
                denom = p * x - q
                if denom <= 0:
                    continue
                if (x * q) % denom != 0:
                    continue
                y = (x * q) // denom
                if y <= x or y > max_val:
                    continue
                if Fraction(1, x) + Fraction(1, y) != r:
                    continue
                pairs.append((x, y))
        
        return pairs
    
    # Meet-in-the-middle for 3-term solutions
    # Precompute all pairs (b, c) with 2 <= b < c <= B, storing 1/b + 1/c
    # Then for each a, check if target - 1/a exists in the precomputed sums
    def search_3_terms_mitm(B):
        nonlocal best_solution, best_max
        
        # Precompute all pair sums
        pair_sums = {}  # Fraction -> list of (b, c)
        
        for b in range(2, B + 1):
            if time.time() > deadline:
                return
            for c in range(b + 1, B + 1):
                s = Fraction(1, b) + Fraction(1, c)
                if s not in pair_sums:
                    pair_sums[s] = []
                pair_sums[s].append((b, c))
        
        # For each a, check if target - 1/a is in pair_sums
        for a in range(2, B + 1):
            if time.time() > deadline:
                return
            
            r = target - Fraction(1, a)
            if r <= 0:
                continue
            
            if r in pair_sums:
                for b, c in pair_sums[r]:
                    if b > a and c > b and c <= B:
                        candidate = [a, b, c]
                        if c < best_max:
                            best_max = c
                            best_solution = candidate
    
    # Systematic search for 3-term solutions using meet-in-the-middle
    # Try increasing bounds
    for B in [50, 100, 150, 200, 300, 500]:
        if time.time() > deadline:
            break
        search_3_terms_mitm(B)
        if best_max <= B:
            break
    
    # Also try 4-term solutions with smaller bounds
    def search_4_terms(max_denom):
        nonlocal best_solution, best_max
        
        for a in range(2, min(max_denom, 30) + 1):
            if time.time() > deadline:
                break
            
            r_after_a = target - Fraction(1, a)
            if r_after_a <= 0:
                continue
            
            for b in range(a + 1, min(max_denom, 50) + 1):
                if time.time() > deadline:
                    break
                
                r = r_after_a - Fraction(1, b)
                if r <= 0:
                    continue
                
                # Find pairs (c, d) with b < c < d <= max_denom
                pairs = find_pairs_for_remainder(r, b, max_denom)
                for c, d in pairs:
                    if c > b and d > c and d <= max_denom:
                        candidate = [a, b, c, d]
                        if d < best_max:
                            best_max = d
                            best_solution = candidate
    
    # Try 4-term solutions if 3-term didn't find something very good
    if best_max > 100:
        for bound in [50, 80, 100]:
            if time.time() > deadline:
                break
            search_4_terms(bound)
    
    # Final verification
    if best_solution is not None:
        s = sum(Fraction(1, d) for d in best_solution)
        if s != target:
            # Fallback to greedy
            best_solution = greedy_solution(target)
            best_max = max(best_solution)
        
        # Ensure constraints
        if len(best_solution) > 256 or len(set(best_solution)) != len(best_solution) or any(d < 1 or d > 10000000 for d in best_solution):
            best_solution = greedy_solution(target)
            best_max = max(best_solution)
    
    return best_solution
# EVOLVE-BLOCK-END
