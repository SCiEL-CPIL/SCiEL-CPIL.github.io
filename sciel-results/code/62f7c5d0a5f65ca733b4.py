"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from bisect import bisect_left

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    # Compute LCM of all divisors for periodicity
    L = 1
    for d in divisors:
        L = lcm(L, d)
    
    # For each divisor, compute the set of residues modulo L that are valid
    # A residue r is valid for divisor d if r % d == 0 and r != 0 (since values must be positive)
    # Actually, value v is valid if v % d == 0 and v >= 1
    # In terms of residue mod L: v ≡ r (mod L), v = k*L + r
    # If r == 0, then v = k*L, which is divisible by d (since d|L). Smallest positive is L.
    # If r > 0, then v = k*L + r, divisible by d iff r % d == 0.
    # So valid residues for divisor d are: {r in [0, L-1] : r % d == 0}
    # But we need v >= 1, so if r == 0, the smallest is L (k=1). If r > 0, smallest is r (k=0).
    
    # Precompute valid residues for each divisor
    valid_residues = []
    for d in divisors:
        res = []
        for r in range(0, L, d):
            res.append(r)
        valid_residues.append(res)
    
    # We want to find n distinct values v_0, ..., v_{n-1} such that:
    # v_i % divisors[i] == 0, all v_i distinct, minimize max(v_i) - min(v_i)
    # 
    # Key insight: The pattern of valid residues repeats every L.
    # If we fix a window [A, A+W], we can check if there exist distinct valid values in this window.
    # 
    # More efficiently: Since the problem is periodic with period L, we can think of it as:
    # Choose residues r_0, ..., r_{n-1} (one from valid_residues[i] for each i) and offsets k_0, ..., k_{n-1}
    # such that v_i = k_i * L + r_i are all distinct and minimize max(v_i) - min(v_i).
    # 
    # But this is complex. Let's use a different approach:
    # Binary search on W (the span). For a given W, check if there exist valid distinct values in some window of length W.
    # 
    # To check feasibility for a given W:
    # We can slide a window of length W and check if we can assign distinct valid values to each divisor.
    # 
    # Even better: Since L is 30030, and we only need 10 values, we can do the following:
    # For each possible starting residue s (0 to L-1), consider the window [s, s+W] (mod L, but actually linear).
    # Wait, the window is linear, not circular. So let's think differently.
    #
    # Let's use backtracking with the periodic structure:
    # We'll try to find a solution by considering the minimum value.
    # Let min_val = m. Then all values are in [m, m+W].
    # m must be a multiple of some divisor d_j.
    # For each candidate m (multiple of some d_j), we try to assign values to all divisors.
    #
    # To make this efficient, we can use the following:
    # For a given window [A, A+W], and for each divisor d_i, the valid values in this window are:
    # {v : A <= v <= A+W, v % d_i == 0}
    # We need to pick one from each set such that all picks are distinct.
    #
    # This is a bipartite matching problem, but since n=10 is small, we can use backtracking.
    #
    # Strategy:
    # 1. Find a good initial solution (e.g., the divisors themselves or a greedy one).
    # 2. Binary search on W using a feasibility check.
    # 3. The feasibility check uses backtracking over the divisors, trying to assign values in the window.
    
    # First, get a quick initial solution using greedy
    def quick_greedy():
        # Try to find a solution by picking the smallest valid value for each divisor
        # in a smart order
        result = [0] * n
        used = set()
        # Sort by divisor size (smaller divisors have more constraints in a small window)
        order = sorted(range(n), key=lambda i: divisors[i])
        for i in order:
            d = divisors[i]
            # Find smallest multiple of d not in used
            v = d
            while v <= max_val:
                if v not in used:
                    result[i] = v
                    used.add(v)
                    break
                v += d
        return result, max(result) - min(result)
    
    best_result, best_span = quick_greedy()
    
    # Feasibility check: can we find distinct valid values in window [A, A+W]?
    def feasible(A, W):
        # For each divisor, get the list of valid values in [A, A+W]
        candidates = []
        for d in divisors:
            vals = []
            # First multiple of d >= A
            start = ((A + d - 1) // d) * d
            if start < A:
                start += d
            for v in range(start, A + W + 1, d):
                if v <= max_val:
                    vals.append(v)
            candidates.append(vals)
        
        # Backtracking to assign distinct values
        used = set()
        def backtrack(idx):
            if idx == n:
                return True
            for v in candidates[idx]:
                if v not in used:
                    used.add(v)
                    if backtrack(idx + 1):
                        return True
                    used.remove(v)
            return False
        
        # Try in order of fewest candidates first (MRV heuristic)
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        # Reorder candidates
        ordered_candidates = [candidates[i] for i in order]
        
        used2 = set()
        def backtrack2(idx):
            if idx == n:
                return True
            for v in ordered_candidates[idx]:
                if v not in used2:
                    used2.add(v)
                    if backtrack2(idx + 1):
                        return True
                    used2.remove(v)
            return False
        
        return backtrack2(0)
    
    # Binary search on W
    # Lower bound: 1 (since values must be distinct, min span is at least 1)
    # Upper bound: best_span
    lo = 1
    hi = best_span
    
    deadline = time.time() + 75
    
    # We'll do a linear search from lo upward, since the span might be small
    # But binary search is more efficient if the span is large
    # Let's try binary search first
    
    while lo < hi and time.time() < deadline:
        mid = (lo + hi) // 2
        # Check if there exists a window of length mid that works
        # We need to check various starting positions A
        # A should be such that the window [A, A+mid] contains valid values for all divisors
        # Let's try A = 1, 2, ..., up to some limit
        # But this is expensive. Instead, let's use the periodicity.
        # 
        # Actually, for a given W, the window start A doesn't need to be arbitrary.
        # The minimum value in the solution will be some multiple of some divisor.
        # So let's iterate over possible minimum values.
        
        found = False
        # Try starting from 1, and also try values that are multiples of the divisors
        # To be efficient, we'll try a limited number of starting points
        # The optimal window's minimum will be a multiple of at least one divisor
        # Let's collect all multiples of all divisors up to some limit
        
        # For efficiency, let's just try A = 1 to A = min(max_val - W, 1000)
        # And also try A that are multiples of the divisors
        tested_A = set()
        
        # First, try small A values
        for A in range(1, min(max_val - mid + 1, 200) + 1):
            if A in tested_A:
                continue
            tested_A.add(A)
            if feasible(A, mid):
                found = True
                break
        
        # If not found, try A values that are multiples of the divisors
        if not found:
            for d in divisors:
                m = d
                while m <= min(max_val - mid, 5000):
                    if m in tested_A:
                        m += d
                        continue
                    tested_A.add(m)
                    if feasible(m, mid):
                        found = True
                        break
                    m += d
                if found:
                    break
        
        if found:
            hi = mid
        else:
            lo = mid + 1
    
    # Now lo is the minimum span. Let's find the actual solution with span = lo
    final_span = lo
    
    # Find the solution
    solution_found = False
    # Try various starting positions
    for A in range(1, min(max_val - final_span + 1, 5000) + 1):
        if time.time() > deadline:
            break
        if feasible(A, final_span):
            # Now find the actual assignment
            candidates = []
            for d in divisors:
                vals = []
                start = ((A + d - 1) // d) * d
                if start < A:
                    start += d
                for v in range(start, A + final_span + 1, d):
                    if v <= max_val:
                        vals.append(v)
                candidates.append(vals)
            
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            ordered_candidates = [candidates[i] for i in order]
            
            used = set()
            assignment = [0] * n
            def find_assignment(idx):
                if idx == n:
                    return True
                for v in ordered_candidates[idx]:
                    if v not in used:
                        used.add(v)
                        assignment[order[idx]] = v
                        if find_assignment(idx + 1):
                            return True
                        used.remove(v)
                return False
            
            if find_assignment(0):
                best_result = assignment[:]
                best_span = final_span
                solution_found = True
                break
    
    # If we didn't find a solution with the binary search result, fall back to the greedy solution
    if not solution_found:
        # Use the greedy solution
        pass
    
    # Local search to potentially improve
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                d = divisors[i]
                cur = best_result[i]
                for offset in range(-2, 3):
                    if offset == 0:
                        continue
                    new_val = cur + offset * d
                    if new_val < 1 or new_val > max_val:
                        continue
                    distinct = True
                    for j in range(n):
                        if j != i and best_result[j] == new_val:
                            distinct = False
                            break
                    if not distinct:
                        continue
                    trial = best_result[:]
                    trial[i] = new_val
                    trial_span = max(trial) - min(trial)
                    if trial_span < best_span:
                        best_span = trial_span
                        best_result = trial
                        improved = True
                        break
    
    # Ensure return type is list of Python ints
    result = [int(x) for x in best_result]
    return result

# EVOLVE-BLOCK-END
