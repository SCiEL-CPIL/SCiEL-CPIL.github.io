"""Bootstrap portfolio: MODEL-AND-OPTIMIZE for Egyptian fraction of 5/31."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd, isqrt
import sys

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 55.0  # Safety margin for 60s limit

def _try_solution(denoms):
    """
    Validate a list of denominators and return the largest one if valid, else None.
    """
    if not denoms:
        return None
    if len(denoms) > MAX_TERMS:
        return None
    if len(set(denoms)) != len(denoms):
        return None
    if any(d < 1 or d > MAX_DENOM for d in denoms):
        return None
    
    total = sum(Fraction(1, d) for d in denoms)
    if total != TARGET:
        return None
    
    return max(denoms)

def _two_term_solutions(p, q, min_d, max_d, used):
    """
    Find all pairs (b, c) with b < c, min_d <= b < c <= max_d,
    b,c not in used, such that 1/b + 1/c = p/q.
    Uses (pb-q)(pc-q) = q^2.
    Returns list of (b, c) pairs.
    """
    if p <= 0 or q <= 0:
        return []
    
    g = gcd(p, q)
    p //= g
    q //= g
    
    q2 = q * q
    solutions = []
    
    # Enumerate divisors of q2
    i = 1
    while i * i <= q2:
        if q2 % i == 0:
            j = q2 // i
            # Check pair (i, j)
            # pb - q = i => b = (i + q) / p
            # pc - q = j => c = (j + q) / p
            if (i + q) % p == 0 and (j + q) % p == 0:
                b = (i + q) // p
                c = (j + q) // p
                if b < c and min_d <= b and c <= max_d:
                    if b not in used and c not in used:
                        solutions.append((b, c))
            # Check pair (j, i) - same as above since b < c required
            # But also check if i and j give valid b,c with b<c
            # Already handled above
        i += 1
    
    return solutions

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Known solution: [8, 31, 248] -> max 248
    known_sol = [8, 31, 248]
    best_max = _try_solution(known_sol)
    best_list = known_sol
    
    if best_max is None:
        return known_sol
    
    # Strategy: Search for solutions with max < best_max
    # Use algebraic 2-term completion for efficiency
    
    def _search_with_algebraic_completion(max_limit):
        """
        Search for a valid Egyptian fraction of 5/31 with all denominators <= max_limit.
        Returns the solution list or None.
        """
        if time.time() > deadline:
            return None
        
        # Try 2-term: 1/a + 1/b = 5/31
        # (5a-31)(5b-31) = 31^2 = 961
        # 961 = 31^2. Divisors: 1, 31, 961
        # 5a-31=1 => a=32/5 not int
        # 5a-31=31 => a=62/5 not int  
        # 5a-31=961 => a=992/5 not int
        # No 2-term solution.
        
        # Try 3-term: 1/a + 1/b + 1/c = 5/31 with a < b < c <= max_limit
        # For each a, solve 1/b + 1/c = 5/31 - 1/a = (5a-31)/(31a)
        for a in range(7, max_limit):
            if time.time() > deadline:
                return None
            if a >= best_max:
                break
            rem_num = 5 * a - 31
            rem_den = 31 * a
            if rem_num <= 0:
                continue
            
            # Find 1/b + 1/c = rem_num/rem_den with a < b < c <= max_limit
            sols = _two_term_solutions(rem_num, rem_den, a + 1, max_limit, {a})
            for (b, c) in sols:
                candidate = [a, b, c]
                cm = _try_solution(candidate)
                if cm is not None and cm < best_max:
                    return candidate
        
        # Try 4-term: 1/a + 1/b + 1/c + 1/d = 5/31
        # For each a, b, solve 1/c + 1/d = 5/31 - 1/a - 1/b
        for a in range(7, max_limit):
            if time.time() > deadline:
                return None
            if a >= best_max:
                break
            rem1_num = 5 * a - 31
            rem1_den = 31 * a
            if rem1_num <= 0:
                continue
            
            for b in range(a + 1, max_limit):
                if time.time() > deadline:
                    return None
                if b >= best_max:
                    break
                # 1/c + 1/d = rem1 - 1/b = (rem1_num * b - rem1_den) / (rem1_den * b)
                rem2_num = rem1_num * b - rem1_den
                rem2_den = rem1_den * b
                if rem2_num <= 0:
                    continue
                
                sols = _two_term_solutions(rem2_num, rem2_den, b + 1, max_limit, {a, b})
                for (c, d) in sols:
                    candidate = [a, b, c, d]
                    cm = _try_solution(candidate)
                    if cm is not None and cm < best_max:
                        return candidate
        
        return None
    
    # Binary search on max_limit to find the minimum
    # We know 248 works, so search [10, 247]
    lo, hi = 10, best_max - 1
    found_limit = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        sol = _search_with_algebraic_completion(mid)
        if sol is not None:
            cm = _try_solution(sol)
            if cm is not None:
                best_max = cm
                best_list = sol
                found_limit = mid
                hi = mid - 1
            else:
                hi = mid - 1
        else:
            lo = mid + 1
    
    # Also try direct search for even better solutions if time permits
    if time.time() < deadline - 10:
        # Try to find solutions with max < best_max using more systematic approach
        for a in range(7, best_max):
            if time.time() > deadline:
                break
            rem_num = 5 * a - 31
            rem_den = 31 * a
            if rem_num <= 0:
                continue
            
            # For 3-term completion
            sols = _two_term_solutions(rem_num, rem_den, a + 1, best_max - 1, {a})
            for (b, c) in sols:
                candidate = [a, b, c]
                cm = _try_solution(candidate)
                if cm is not None and cm < best_max:
                    best_max = cm
                    best_list = candidate
                    break
            
            # For 4-term completion  
            if time.time() > deadline:
                break
            for b in range(a + 1, best_max):
                if time.time() > deadline:
                    break
                rem2_num = rem_num * b - rem_den
                rem2_den = rem_den * b
                if rem2_num <= 0:
                    continue
                sols2 = _two_term_solutions(rem2_num, rem2_den, b + 1, best_max - 1, {a, b})
                for (c, d) in sols2:
                    candidate = [a, b, c, d]
                    cm = _try_solution(candidate)
                    if cm is not None and cm < best_max:
                        best_max = cm
                        best_list = candidate
                        break
                if time.time() > deadline:
                    break
    
    return best_list

# EVOLVE-BLOCK-END
