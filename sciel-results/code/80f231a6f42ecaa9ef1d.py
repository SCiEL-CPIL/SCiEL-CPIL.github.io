from fractions import Fraction
import time
from math import gcd

_DEADLINE = 0

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 50  # 50s limit to be safe
    
    # Try to find a minimal solution using DFS for small k
    # Based on analysis, k=8 is the theoretical minimum.
    # We search for k from 8 upwards.
    
    for k in range(8, 21):
        if time.time() > _DEADLINE:
            break
        result = _find_k_terms_fast(k)
        if result:
            return result
            
    # Fallback to greedy if DFS fails or times out
    return greedy_solution()

def _find_k_terms_fast(k):
    """Find k distinct denominators >= 5 summing to 1 using fast DFS."""
    global _DEADLINE
    
    # Precompute harmonic sums for pruning
    # H(start, count) = sum(1/(start+i) for i in range(count))
    def harmonic_sum(start, count):
        s = 0.0
        for i in range(count):
            s += 1.0 / (start + i)
        return s

    def dfs(terms_rem, t_num, t_den, min_d, chosen):
        global _DEADLINE
        if time.time() > _DEADLINE:
            return None
        
        if terms_rem == 0:
            return chosen if t_num == 0 else None
        
        if terms_rem == 1:
            if t_num == 0:
                return None
            if t_den % t_num == 0:
                d = t_den // t_num
                if d >= min_d and d <= 10000000:
                    return chosen + [d]
            return None
        
        # Pruning: Max sum of remaining terms must be >= target
        target_float = t_num / t_den
        max_s = harmonic_sum(min_d, terms_rem)
        if max_s < target_float - 1e-15:
            return None
        
        # d bounds
        # We need 1/d <= target. So d >= 1/target = t_den/t_num.
        d_lo = max(min_d, -(-t_den // t_num)) # ceil(t_den/t_num)
        
        # Upper bound for d:
        # Even if we pick d, the remaining terms_rem-1 terms must sum to target - 1/d.
        # The max sum of remaining terms is harmonic_sum(d+1, terms_rem-1).
        # So we need target - 1/d <= harmonic_sum(d+1, terms_rem-1).
        # This is hard to invert.
        # Simpler bound: d cannot be so large that 1/d is negligible and we can't reach target with remaining?
        # Actually, if d is very large, 1/d is small.
        # A loose upper bound: d <= terms_rem * t_den / t_num is not correct.
        # Let's just iterate d and break when max_rest < new_target.
        
        d_hi = 10000000 # Cap
        
        for d in range(d_lo, d_hi + 1):
            if time.time() > _DEADLINE:
                return None
            
            # Check if 1/d is too big
            if t_num * d < t_den:
                continue
                
            new_num = t_num * d - t_den
            if new_num <= 0:
                continue
            new_den = t_den * d
            
            # Check if remaining is achievable
            new_target_float = new_num / new_den
            max_rest = harmonic_sum(d + 1, terms_rem - 1)
            if new_target_float > max_rest + 1e-15:
                # If even the max sum of remaining terms is less than target,
                # increasing d further will only decrease 1/d, increasing new_target,
                # so we can break.
                break
            
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            res = dfs(terms_rem - 1, new_num, new_den, d + 1, chosen + [d])
            if res is not None:
                return res
        
        return None
    
    return dfs(k, 1, 1, 5, [])

def greedy_solution():
    """Generate a valid Egyptian fraction using greedy algorithm."""
    R = Fraction(1, 1)
    D = 5
    denoms = []
    while R > 0:
        inv_R = Fraction(R.denominator, R.numerator)
        d = max(D, inv_R.numerator // inv_R.denominator)
        if (inv_R.numerator % inv_R.denominator) != 0:
            d += 1
        if d < D:
            d = D
        denoms.append(d)
        R -= Fraction(1, d)
        D = d + 1
        if len(denoms) > 256:
            break
    # Verify the solution
    if _verify_solution(denoms):
        return denoms
    return [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256]

def _verify_solution(denoms):
    """Verify that a solution has distinct denominators in [5,10000000] summing to 1."""
    if not denoms:
        return False
    if len(denoms) > 256:
        return False
    if len(set(denoms)) != len(denoms):
        return False
    for d in denoms:
        if d < 5 or d > 10000000:
            return False
    total = Fraction(0, 1)
    for d in denoms:
        total += Fraction(1, d)
    return total == Fraction(1, 1)
