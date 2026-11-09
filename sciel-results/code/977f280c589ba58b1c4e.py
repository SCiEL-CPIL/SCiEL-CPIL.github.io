import time
import random
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    # Precompute LCM of all divisors for periodicity
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)

    # Initial solution: use the divisors themselves
    naive = divisors[:]
    best_val = max(naive) - min(naive)
    best_result = naive[:]

    def get_candidates(L, W, d):
        """Get all multiples of d in [L, L+W] (inclusive)."""
        if L < 1 or L + W > MAX_V:
            return []
        k_start = (L + d - 1) // d
        k_end = (L + W) // d
        if k_start > k_end:
            return []
        # Ensure within MAX_V
        max_k = MAX_V // d
        if k_end > max_k:
            k_end = max_k
        if k_start > k_end:
            return []
        return list(range(k_start * d, (k_end + 1) * d, d))

    def backtrack(assigned, used, order, options, idx, deadline):
        """Backtracking with MRV already applied via order."""
        if time.time() > deadline:
            return False
        if idx == len(order):
            return True
        d = order[idx]
        for val in options[idx]:
            if val not in used:
                assigned[d] = val
                used.add(val)
                if backtrack(assigned, used, order, options, idx + 1, deadline):
                    return True
                del assigned[d]
                used.remove(val)
        return False

    # Strategy: Try increasing W starting from theoretical minimum.
    # The minimum possible W is 9 (10 distinct integers need span >= 9).
    # Use LCM periodicity: for a given W, we only need to check L in [0, LCM-1]
    # because the pattern of multiples repeats every LCM.
    
    # For each W, we check L values. Since the pattern repeats with period LCM,
    # we can limit L to [1, LCM] (or a subset if that's too large).
    # LCM = 30030, which is manageable.
    
    # We'll try W from 9 upward.
    # For each W, we scan L from 1 to min(LCM, MAX_V - W).
    # If a solution is found for a given W, we can stop (since we're increasing W).
    
    # Optimization: For a given W and L, quickly check if each divisor has
    # at least one multiple in [L, L+W]. If not, skip.
    
    # Further optimization: Use bitsets for the constraint propagation.
    # For a window of width W, positions are 0..W (relative to L).
    # For each divisor i, compute which positions are valid.
    # Then use matching/backtracking on the positions.
    
    # Let's implement a faster check using bitsets.
    
    def check_and_solve(W, L, deadline):
        """Try to find a valid assignment in window [L, L+W]."""
        if L < 1 or L + W > MAX_V:
            return None
        
        # Compute candidate bitsets for each divisor
        # Position j (0-indexed) corresponds to value L + j
        # A position j is valid for divisor d iff (L + j) % d == 0
        
        candidate_bits = []
        for d in divisors:
            bits = 0
            # Find first j such that (L + j) % d == 0
            # L + j ≡ 0 (mod d) => j ≡ -L (mod d) => j ≡ (d - L % d) % d (mod d)
            first_j = (d - (L % d)) % d
            if first_j > W:
                return None  # No multiple in window
            for j in range(first_j, W + 1, d):
                bits |= (1 << j)
            candidate_bits.append(bits)
        
        # Check if each divisor has at least one candidate
        for bits in candidate_bits:
            if bits == 0:
                return None
        
        # Now we need to assign distinct positions to each divisor.
        # This is a matching problem. We'll use backtracking with MRV.
        
        # Order divisors by number of available positions (MRV)
        order = sorted(range(n), key=lambda i: bin(candidate_bits[i]).count('1'))
        ordered_bits = [candidate_bits[i] for i in order]
        
        assigned_pos = {}  # divisor index -> position
        used_positions = 0  # bitset of used positions
        
        def bt(idx, deadline):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            i = order[idx]
            avail = candidate_bits[i] & ~used_positions
            
            # Try each available position
            # For efficiency, iterate through set bits
            bits = avail
            while bits:
                # Get lowest set bit
                lb = bits & (-bits)
                j = lb.bit_length() - 1
                bits ^= lb
                
                # Assign position j to divisor i
                assigned_pos[i] = j
                new_used = used_positions | (1 << j)
                
                if bt(idx + 1, deadline):
                    return True
                
                # Undo
                del assigned_pos[i]
            
            return False
        
        if bt(0, deadline):
            # Build result
            result = [0] * n
            for i in range(n):
                result[i] = L + assigned_pos[i]
            return result
        
        return None
    
    # Search for the smallest W
    # Start from W=9 (theoretical minimum for 10 distinct integers)
    # We need to check L values. Due to periodicity, L mod LCM determines the pattern.
    # But L must be >= 1. So we can check L in [1, min(LCM, MAX_V - W)].
    
    # However, checking all L in [1, LCM] for each W might be slow if W is small
    # and most L values are infeasible. Let's add a quick feasibility check.
    
    # Quick feasibility: For each divisor d, the window [L, L+W] must contain
    # at least one multiple of d. This is equivalent to W >= d - 1 or the window
    # straddling a multiple. Actually, the condition is that there exists j in [0, W]
    # such that (L+j) % d == 0. This is always true if W >= d-1. If W < d-1,
    # it's true only for specific L values.
    
    # Let's just run the check_and_solve function for each L.
    
    # To speed up, we can skip L values where the first feasibility check fails.
    # But check_and_solve already does this quickly.
    
    # Let's try W from 9 upward.
    # For each W, scan L from 1 to min(LCM, MAX_V - W).
    # If LCM is too large, we can sample or use a smarter search.
    # LCM = 30030, which is fine.
    
    found = False
    for W in range(9, 100):  # Cap at 100 for safety
        if time.time() > deadline:
            break
        
        max_L = min(LCM, MAX_V - W)
        if max_L < 1:
            continue
        
        # Scan L values
        # For efficiency, we can step through L values.
        # Since the pattern is periodic with LCM, we only need to check one period.
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                break
            
            result = check_and_solve(W, L, deadline)
            if result is not None:
                val = max(result) - min(result)
                if val < best_val:
                    best_val = val
                    best_result = result
                found = True
                break
        
        if found:
            break
    
    # If no solution found with the periodicity-based search, try a broader search
    if not found:
        # Fallback: random search
        for _ in range(1000):
            if time.time() > deadline:
                break
            W = random.randint(9, 50)
            L = random.randint(1, MAX_V - W)
            result = check_and_solve(W, L, deadline)
            if result is not None:
                val = max(result) - min(result)
                if val < best_val:
                    best_val = val
                    best_result = result
    
    # Ensure types are correct
    best_result = [int(x) for x in best_result]
    return best_result
