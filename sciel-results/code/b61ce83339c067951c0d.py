import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    max_val = 1000000
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, divisors)  # L = 30030
    n = len(divisors)
    
    # For each divisor, precompute the sorted list of multiples in [0, L)
    # This helps with quick lookup
    div_mults = []
    for d in divisors:
        muls = list(range(d, L, d))
        div_mults.append(muls)
    
    # Search for minimal W
    # Theoretical min W is n-1 = 9
    best_solution = None
    best_W = None
    
    for W in range(9, 500):  # Upper bound on W
        if time.time() > deadline:
            break
        
        # For this W, find all M in [0, L) where every divisor has at least one multiple in [M, M+W]
        # Key optimization: only check M values that are "aligned" - i.e., M mod d_i is small enough
        # that a multiple of d_i falls in [M, M+W]
        
        # For divisor d, M is valid if there exists k such that k*d in [M, M+W]
        # This means M <= k*d <= M+W, so k*d - W <= M <= k*d
        # So M must be in the union over k of [k*d - W, k*d]
        
        # Instead of checking all M, let's enumerate candidate M values
        # A candidate M is one where for at least one divisor, M is close to a multiple boundary
        # More precisely, M must satisfy: for all i, exists k_i: M <= k_i * d_i <= M + W
        
        # Efficient approach: iterate M from 0 to L-1, but use early rejection
        # For each M, check if all divisors have a multiple in [M, M+W]
        # This is O(n) per M, so O(n*L) per W = 10 * 30030 = 300300 checks per W
        # For W up to 500, that's 150M checks... too slow in Python
        
        # Better: for each W, collect the set of valid M values by intersecting constraints
        # For divisor d_i, valid M values are those where [M, M+W] contains a multiple of d_i
        # i.e., M in [k*d_i - W, k*d_i] for some k, intersected with [0, L)
        
        # Precompute for each divisor the bitmask of valid M positions
        # But L=30030, so a bitset of 30030 bits is fine
        
        # For each divisor, compute which M in [0, L) have a multiple in [M, M+W]
        valid_M_mask = None  # We'll use a boolean array
        
        # For speed, use a bytearray of length L, initialized to 1, then mark invalid
        # Actually, let's use a different approach: for each divisor, generate the valid M ranges
        
        # For divisor d, the valid M values are: for each multiple m of d in [0, L+W),
        # M can be in [max(0, m-W), min(L-1, m)]
        # Union these intervals for all multiples of d
        
        # Let's just iterate M and check, but with early exit
        # For each M, for each divisor, check if ceil(M/d)*d <= M+W
        
        found = False
        for M in range(L):
            if time.time() > deadline:
                if best_solution:
                    found = True  # break out
                    break
                else:
                    # No solution yet, need to continue or use fallback
                    break
            
            # Check if all divisors have a multiple in [M, M+W]
            candidates = []
            feasible = True
            for i, d in enumerate(divisors):
                # Find multiples of d in [M, M+W]
                k_min = (M + d - 1) // d  # ceil(M/d)
                k_max = (M + W) // d       # floor((M+W)/d)
                
                if k_min > k_max:
                    feasible = False
                    break
                
                cands = []
                for k in range(k_min, k_max + 1):
                    x = k * d
                    if x >= 1 and x <= max_val:
                        cands.append(x)
                
                if not cands:
                    feasible = False
                    break
                candidates.append(cands)
            
            if not feasible:
                continue
            
            # Try to find distinct representatives via backtracking
            # Sort by number of candidates (most constrained first)
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            # If any has 0 candidates, skip (already checked)
            # Quick check: if total candidates < n, impossible
            total_cands = sum(len(candidates[i]) for i in range(n))
            if total_cands < n:
                continue
            
            # Backtracking with MRV (Minimum Remaining Values)
            assignment = [None] * n
            used = set()
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = order[idx]
                for x in candidates[i]:
                    if x not in used:
                        used.add(x)
                        assignment[i] = x
                        if backtrack(idx + 1):
                            return True
                        used.remove(x)
                        assignment[i] = None
                return False
            
            if backtrack(0):
                # Found a solution with width W
                result = [assignment[i] for i in range(n)]
                # Verify
                if len(set(result)) == n and all(result[i] % divisors[i] == 0 for i in range(n)):
                    return result
                found = True
                break
        
        if found:
            break
    
    # If we exhausted W or time, return a valid solution
    # Use a simple valid solution: x_i = (i+1) * L
    # All distinct, all <= 10*L = 300300 <= 1000000, all divisible by d_i since L is multiple of d_i
    fallback = [(i + 1) * L for i in range(n)]
    return fallback
