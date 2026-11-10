import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s limit (or 60s if that's the real limit)
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # Calculate LCM of divisors to determine periodicity
    # LCM(6, 10, 14, 15, 21, 22, 26, 33, 35, 39)
    # Primes involved: 2, 3, 5, 7, 11, 13
    # LCM = 2 * 3 * 5 * 7 * 11 * 13 = 30030
    lcm_val = 30030

    # Precompute candidate lists for each divisor within one period [1, lcm_val]
    # candidates[i] will be a list of (value, index_in_window) but we just need values
    # Actually, for a sliding window [S, S+W], the candidates for divisor d are
    # multiples of d in that range.
    # Since the pattern repeats every lcm_val, we can work with modular arithmetic
    # or just precompute the sorted list of multiples for each divisor in [1, lcm_val].
    
    multiples = []
    for d in divisors:
        ms = []
        # Multiples of d in [1, lcm_val]
        for k in range(1, lcm_val // d + 1):
            ms.append(k * d)
        multiples.append(ms)
        
    # For a window [S, S+W], the candidates for divisor i are:
    # m in multiples[i] such that S <= m <= S+W
    # OR S <= m + lcm_val <= S+W (if S is near the end of the period)
    # OR S - lcm_val <= m <= S+W - lcm_val (if S is near the start)
    # Actually, since we restrict S to [1, lcm_val], the window [S, S+W] covers
    # at most two periods' worth of "wraps".
    # A value v is in [S, S+W] if and only if:
    # (v % lcm_val) is in the "unwrapped" interval [S, S+W] modulo lcm_val.
    # Let's just generate candidates on the fly using the precomputed list,
    # checking both m and m +/- lcm_val.

    def get_candidates(S, W, i):
        """Get distinct multiples of divisors[i] in [S, S+W]"""
        d = divisors[i]
        ms = multiples[i]
        cands = set()
        
        # The window [S, S+W] in the integer line.
        # We look for m in ms such that m, m+lcm, m-lcm fall in [S, S+W].
        # Since S in [1, lcm] and W is small (< lcm), the window intersects
        # at most 2 consecutive periods.
        
        # Find approximate index in ms
        # ms is sorted.
        # We can just iterate if W is small? No, ms can be large (lcm/2 ~ 15000).
        # But W is small. So the number of candidates is W/d + 1.
        # We can compute the first multiple >= S and iterate.
        
        # First multiple >= S
        # k = ceil(S/d)
        k_start = (S + d - 1) // d
        first_val = k_start * d
        
        val = first_val
        end_val = S + W
        while val <= end_val:
            cands.add(val)
            val += d
            
        return list(cands)

    def check_assignment(S, W):
        """
        Check if there is a valid distinct assignment for window [S, S+W].
        Returns the assignment list or None.
        """
        candidates = []
        for i in range(n):
            c = get_candidates(S, W, i)
            if not c:
                return None
            candidates.append(c)
        
        # Backtracking with MRV heuristic
        # Sort indices by length of candidates
        indices = list(range(n))
        indices.sort(key=lambda i: len(candidates[i]))
        
        assigned = [0] * n
        used = set()
        
        def backtrack(idx):
            if idx == len(indices):
                return True
            
            i = indices[idx]
            for val in candidates[i]:
                if val not in used:
                    assigned[i] = val
                    used.add(val)
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
                    assigned[i] = 0
            return False
        
        if backtrack(0):
            return assigned
        return None

    best_solution = None
    best_W = float('inf')
    
    # Search for minimal W
    # Lower bound: n-1 = 9 (since we need 10 distinct integers)
    # Upper bound: The trivial solution (divisors themselves) has span 
    # max(divisors) - min(divisors) = 39 - 6 = 33.
    # So we search W from 9 to 32.
    
    # To speed up, we can check if a window of size W is possible by counting
    # but the distinctness constraint makes it a matching problem.
    
    # Optimization:
    # If W is very small, the number of candidates per divisor is small.
    # For W < 33, the backtracking is very fast.
    # The bottleneck is iterating S from 1 to 30030 for each W.
    # 30030 * 24 (W values) * (cost of check)
    # Cost of check:
    #   get_candidates: O(W/d) which is small.
    #   backtrack: O(10!) worst case, but pruned heavily.
    #   
    # Let's try to optimize the S loop.
    # Instead of checking every S, can we skip?
    # Not easily, because the alignment of multiples matters.
    
    # However, we can precompute the "mask" of which divisors have candidates
    # for each S?
    # For a fixed W, as S increases by 1, the window shifts.
    # The set of candidates changes.
    
    # Let's just run the loop. 30030 * 24 = 720k iterations.
    # If each iteration takes ~10us, that's 7s.
    # If it takes ~100us, that's 72s (too slow).
    # We need to make check_assignment very fast or skip S values.
    
    # Idea:
    # A necessary condition for a solution is that for every divisor i,
    # there is at least one multiple in [S, S+W].
    # This is equivalent to: the gap between consecutive multiples of d_i
    # (which is d_i) must be <= W? No, that's for coverage of the line.
    # For a specific window [S, S+W] to contain a multiple of d,
    # S mod d must be such that the distance to the next multiple is <= W.
    # Let r = S % d.
    # The next multiple is at S + (d - r) % d.
    # So we need (d - S % d) % d <= W.
    # This gives a constraint on S mod d.
    # We can precompute for each d, the set of residues r in [0, d-1]
    # such that a window starting at S with S% d = r contains a multiple.
    # Actually, simpler:
    # For a fixed d and W, the "bad" S values are those where the gap
    # to the next multiple is > W.
    # Gap = (d - S % d) % d.
    # If Gap > W, then no multiple in [S, S+W].
    # So S is valid for d if (d - S % d) % d <= W.
    
    # We can combine these constraints for all d using CRT or just check.
    # But we also need the DISTINCT assignment.
    
    # Let's try to optimize the inner loop by checking the necessary condition first.
    
    # Precompute valid residues for each divisor for a given W?
    # No, W changes in the outer loop.
    
    # Let's just write the optimized loop and see.
    # We can also break early in the S loop if time is running out.
    
    for W in range(9, 33):
        if time.time() > deadline:
            break
            
        # For this W, check all S in [1, lcm_val]
        # Optimization: 
        # If W is small, most S will fail the "has candidate" check.
        # We can precompute a boolean array for each S if it has candidates for ALL divisors?
        # Let's try a faster necessary check.
        
        # For each divisor d, the condition is (d - S % d) % d <= W.
        # Let's precompute the "validity" of S for each d?
        # No, just check inline.
        
        for S in range(1, lcm_val + 1):
            if time.time() > deadline:
                break
                
            # Quick necessary check
            valid_nec = True
            for i in range(n):
                d = divisors[i]
                gap = (d - S % d) % d
                if gap > W:
                    valid_nec = False
                    break
            
            if not valid_nec:
                continue
                
            # If necessary condition passes, try to find assignment
            sol = check_assignment(S, W)
            if sol is not None:
                # Validate
                if len(set(sol)) == n and all(1 <= x <= max_val and x % divisors[i] == 0 for i, x in enumerate(sol)):
                    best_solution = sol
                    best_W = W
                    break # Found minimal W for this S, but we want global min W.
                         # Since we iterate W from low to high, the first found is minimal W.
                         # We can return immediately.
                    return [int(x) for x in sol]
    
    if best_solution is not None:
        return [int(x) for x in best_solution]
        
    # Fallback if no solution found in range (should not happen)
    # Return the divisors themselves
    return divisors[:]
