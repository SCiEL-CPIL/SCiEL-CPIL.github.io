"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    divisors = P['divisors']
    n = len(divisors)
    
    # Compute LCM of all divisors to find the period of the pattern
    # The pattern of which multiples are available repeats every LCM steps
    lcm_all = 1
    for d in divisors:
        lcm_all = _lcm(lcm_all, d)
        if lcm_all > 10**7:
            break
    
    # Strategy: Sliding window over sorted multiples
    # For each divisor i, the valid values are: k * divisors[i] for k >= 1, k * divisors[i] <= 1000000
    # We need to pick one value v_i from each divisor's set such that all v_i are distinct
    # and max(v_i) - min(v_i) is minimized.
    
    # Since the divisors are fixed and relatively small, the optimal solution will likely
    # have all values within a small range. The LCM of all divisors is large, but the
    # "density" of multiples is high enough that we can find a tight cluster.
    
    # Approach:
    # 1. Generate all possible multiples for each divisor.
    # 2. Sort all these multiples into a single list, tagged with which divisor they belong to.
    # 3. Use a sliding window to find the smallest interval [L, R] that contains at least one
    #    multiple from each divisor.
    # 4. For that interval, pick one representative from each divisor within [L, R].
    #    To ensure distinctness, if two divisors pick the same value, we need to adjust.
    #    However, since the window is small and there are many multiples, collisions are rare
    #    or can be resolved by picking the next available multiple.
    
    # Let's generate the multiples.
    # For divisor d, multiples are d, 2d, 3d, ..., floor(1000000/d)*d
    all_multiples = []
    for i, d in enumerate(divisors):
        max_k = 1000000 // d
        for k in range(1, max_k + 1):
            val = k * d
            all_multiples.append((val, i))
    
    # Sort by value
    all_multiples.sort(key=lambda x: x[0])
    
    # Sliding window to find minimal span containing all 10 divisors
    best_span = float('inf')
    best_left_idx = 0
    best_right_idx = 0
    
    count = [0] * n
    have_all = 0
    left = 0
    
    for right in range(len(all_multiples)):
        val_r, div_r = all_multiples[right]
        count[div_r] += 1
        if count[div_r] == 1:
            have_all += 1
        
        while have_all == n:
            val_l, div_l = all_multiples[left]
            span = val_r - val_l
            if span < best_span:
                best_span = span
                best_left_idx = left
                best_right_idx = right
            
            count[div_l] -= 1
            if count[div_l] == 0:
                have_all -= 1
            left += 1
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
    
    # Now we have the best window [all_multiples[best_left_idx][0], all_multiples[best_right_idx][0]]
    # We need to pick one value for each divisor within this window.
    # Let L = all_multiples[best_left_idx][0]
    # Let R = all_multiples[best_right_idx][0]
    
    L_val = all_multiples[best_left_idx][0]
    R_val = all_multiples[best_right_idx][0]
    
    # For each divisor, find the smallest multiple >= L_val and <= R_val.
    # If there's a collision (two divisors pick the same value), we need to resolve it.
    # Since the window is small, we can try to adjust.
    
    representatives = [0] * n
    used_values = set()
    
    # First pass: pick the smallest valid multiple for each divisor in [L_val, R_val]
    for i, d in enumerate(divisors):
        # Find smallest k such that k*d >= L_val and k*d <= R_val
        k_min = (L_val + d - 1) // d
        val = k_min * d
        if val > R_val:
            # This shouldn't happen if the window was valid, but just in case
            # We need to expand the window or pick a different one
            # For now, mark as invalid
            val = -1
        representatives[i] = val
    
    # Check for collisions and resolve
    # If val_i == val_j for i != j, we need to change one of them to the next multiple.
    # We'll do this greedily: sort divisors by the number of available options in the window,
    # or just iterate and fix collisions.
    
    # A simpler robust approach:
    # Since the window is minimal, the values are very close.
    # Let's collect all candidate values in [L_val, R_val] for each divisor.
    
    candidates = [[] for _ in range(n)]
    for i, d in enumerate(divisors):
        k_min = (L_val + d - 1) // d
        k_max = R_val // d
        for k in range(k_min, k_max + 1):
            candidates[i].append(k * d)
    
    # Now we need to pick one from each list such that all are distinct.
    # This is a bipartite matching problem, but n=10 is small.
    # We can use backtracking or greedy with swapping.
    
    # Greedy: Sort divisors by length of candidate list (ascending).
    # Assign the smallest available value.
    
    order = sorted(range(n), key=lambda i: len(candidates[i]))
    
    assignment = {}
    used = set()
    
    # Backtracking search for distinct assignment
    def backtrack(idx):
        if idx == len(order):
            return True
        i = order[idx]
        for val in candidates[i]:
            if val not in used:
                assignment[i] = val
                used.add(val)
                if backtrack(idx + 1):
                    return True
                used.remove(val)
                del assignment[i]
        return False
    
    if backtrack(0):
        result = [assignment[i] for i in range(n)]
        # Ensure types are Python int
        result = [int(v) for v in result]
        return result
    else:
        # Fallback: just return the divisors themselves (valid but not optimal)
        return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
