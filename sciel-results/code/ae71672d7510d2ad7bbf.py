"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from math import isqrt

def _is_smooth(n, primes):
    """Check if n is composed only of primes in the set primes."""
    if n <= 1:
        return True
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            while n % p == 0:
                n //= p
    return n == 1

def _find_smooth_set(n_needed, limit, primes):
    """
    Find n_needed distinct integers x_i in [1, limit] such that 
    all pairwise sums x_i + x_j are primes-smooth.
    Uses backtracking.
    """
    if limit < n_needed:
        return None
    
    candidates = list(range(1, limit + 1))
    solution = []
    
    def backtrack(start_idx, current_list):
        if len(current_list) == n_needed:
            return True
        for i in range(start_idx, len(candidates)):
            candidate = candidates[i]
            valid = True
            for existing in current_list:
                s = candidate + existing
                if not _is_smooth(s, primes):
                    valid = False
                    break
            if valid:
                current_list.append(candidate)
                if backtrack(i + 1, current_list):
                    return True
                current_list.pop()
        return False

    if backtrack(0, solution):
        return solution
    return None

def construct():
    # Strategy:
    # 1. Iterate over candidate prime sets S in order of increasing size.
    # 2. For each S, let M = product(S).
    # 3. Search for 12 distinct integers x_i <= 10000/M such that x_i + x_j is S-smooth.
    # 4. If found, return [M * x_i].
    # 5. If not found with the "multiple of M" constraint, fallback to the original 
    #    "a_i is S-smooth" search (which is less restrictive on a_i but more on sums).
    #    Actually, the "multiple of M" search is a subset of the "a_i is S-smooth" search?
    #    If a_i is a multiple of M, and M is product of S, a_i is S-smooth? No.
    #    a_i = M * x_i. If x_i is not S-smooth, a_i is not S-smooth.
    #    But the sum a_i+a_j = M(x_i+x_j).
    #    If x_i+x_j is S-smooth, then a_i+a_j is S-smooth.
    #    This allows a_i to have prime factors outside S (from x_i), as long as the SUMS are smooth.
    #    This is a MORE powerful search space than just picking S-smooth a_i.
    
    candidate_prime_sets = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
    ]
    
    for primes in candidate_prime_sets:
        M = 1
        for p in primes:
            M *= p
        
        max_x = 10000 // M
        if max_x < 12:
            continue
            
        x_solution = _find_smooth_set(12, max_x, primes)
        if x_solution:
            return [M * x for x in x_solution]
            
    # Fallback to original logic if the scaled search fails
    # This handles cases where a_i are not multiples of M but sums are still smooth
    # (e.g. sums not divisible by M)
    
    candidate_prime_sets_orig = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
    ]
    
    for primes in candidate_prime_sets_orig:
        # Generate all S-smooth numbers up to 10000
        smooth_numbers = set()
        def gen_smooth(p_idx, current):
            if p_idx == len(primes):
                if 1 <= current <= 10000:
                    smooth_numbers.add(current)
                return
            p = primes[p_idx]
            val = 1
            while current * val <= 10000:
                gen_smooth(p_idx + 1, current * val)
                val *= p
        
        gen_smooth(0, 1)
        smooth_list = sorted(smooth_numbers)
        
        solution = []
        
        def backtrack(start_idx, current_list):
            if len(current_list) == 12:
                return True
            for i in range(start_idx, len(smooth_list)):
                candidate = smooth_list[i]
                valid = True
                for existing in current_list:
                    s = candidate + existing
                    if not _is_smooth(s, primes):
                        valid = False
                        break
                if valid:
                    current_list.append(candidate)
                    if backtrack(i + 1, current_list):
                        return True
                    current_list.pop()
            return False
        
        if backtrack(0, solution):
            return solution

    return list(range(1, 13))
# EVOLVE-BLOCK-END
