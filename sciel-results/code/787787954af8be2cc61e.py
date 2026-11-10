"""Program for small nonzero signed harmonic subsums."""
import time
from fractions import Fraction
from math import gcd
from bisect import bisect_left, bisect_right

# EVOLVE-BLOCK-START
def construct():
    """Return exactly 30 integer coefficients from {-1,0,1}.
    Minimize |sum c[i-1]/i| subject to nonzero sum.
    Uses meet-in-the-middle with integer arithmetic via common denominator.
    """
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit

    n = 30

    # Compute L = lcm(1, 2, ..., 30)
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)

    # weights[i] = L // (i+1) for coefficient index i (0-based), so term is c_i * L/(i+1)
    weights = [L // (i + 1) for i in range(n)]

    # Split into two halves: first 15 and last 15
    half = n // 2  # 15
    left_weights = weights[:half]
    right_weights = weights[half:]

    # Enumerate all 3^15 sums for the left half
    # Use iterative generation to avoid recursion overhead
    left_sums = [0]
    for w in left_weights:
        new_sums = []
        for s in left_sums:
            new_sums.append(s)       # coefficient 0
            new_sums.append(s + w)   # coefficient +1
            new_sums.append(s - w)   # coefficient -1
        left_sums = new_sums

    # Sort left_sums and also track which coefficients produced each sum
    # To save memory, we'll just sort the sums and do binary search
    left_sums_sorted = sorted(left_sums)

    # Now enumerate right half and for each right sum, find the left sum
    # that makes total closest to 0 but not 0
    best_int_sum = None  # absolute integer sum value (we want min nonzero)
    best_left_sum = None
    best_right_sum = None

    # Enumerate right sums
    right_sums = [0]
    for w in right_weights:
        new_sums = []
        for s in right_sums:
            new_sums.append(s)
            new_sums.append(s + w)
            new_sums.append(s - w)
        right_sums = new_sums

    # For each right sum, binary search in left_sums_sorted for -right_sum
    # to find closest left sum making total nonzero and minimal
    for rs in right_sums:
        if time.time() > deadline:
            break
        target = -rs
        # Find insertion point
        pos = bisect_left(left_sums_sorted, target)
        # Check candidates around pos
        for p in (pos - 1, pos, pos + 1):
            if 0 <= p < len(left_sums_sorted):
                ls = left_sums_sorted[p]
                total = ls + rs
                if total != 0:
                    abs_total = abs(total)
                    if best_int_sum is None or abs_total < best_int_sum:
                        best_int_sum = abs_total
                        best_left_sum = ls
                        best_right_sum = rs

    # Reconstruct coefficients from best_left_sum and best_right_sum
    # We need to find which combination of coefficients produces best_left_sum
    # and which produces best_right_sum.
    # This is a subset-sum reconstruction problem.

    def find_coeffs(weights, target):
        """Find coefficients in {-1,0,1} such that sum c_i * w_i = target."""
        # Use DFS with memoization or backtracking
        # For 15 variables, 3^15 = 14M is too large for naive backtracking
        # Use meet-in-the-middle within this group too? No, just backtracking with pruning.
        # Actually, let's use a different approach: enumerate all and find the one.
        # Since we already enumerated all sums, we can store the path.
        # Let me re-enumerate with path tracking.
        pass

    # Alternative: re-enumerate with path tracking
    def enumerate_with_path(weights, target):
        """Enumerate all coefficient assignments and return the one matching target."""
        results = []
        current = [0] * len(weights)
        
        def dfs(idx, current_sum):
            if idx == len(weights):
                if current_sum == target:
                    results.append(current[:])
                return
            # Try 0
            current[idx] = 0
            dfs(idx + 1, current_sum)
            # Try +1
            current[idx] = 1
            dfs(idx + 1, current_sum + weights[idx])
            # Try -1
            current[idx] = -1
            dfs(idx + 1, current_sum - weights[idx])
        
        dfs(0, 0)
        return results[0] if results else None

    # This DFS might be slow. Let's use a smarter approach:
    # Store paths during enumeration.
    
    # Actually, let's just do a targeted search. Since we found the optimal integer sum,
    # we can reconstruct by trying all possibilities for smaller subsets.
    
    # Simpler: just do the full enumeration with path storage for the halves.
    # 3^15 = 14M paths is too much to store. Let's use a different strategy.
    
    # For reconstruction, use the fact that we can verify: given target, find coeffs.
    # Use greedy or backtracking on 15 variables.
    
    def reconstruct(weights, target):
        """Backtracking to find coefficients in {-1,0,1} summing to target."""
        coeffs = [0] * len(weights)
        
        def bt(idx, cur_sum):
            if idx == len(weights):
                return cur_sum == target
            w = weights[idx]
            for val in (0, 1, -1):
                coeffs[idx] = val
                if bt(idx + 1, cur_sum + val * w):
                    return True
            coeffs[idx] = 0
            return False
        
        if bt(0, 0):
            return coeffs
        return None

    left_coeffs = reconstruct(left_weights, best_left_sum)
    right_coeffs = reconstruct(right_weights, best_right_sum)

    if left_coeffs is None or right_coeffs is None:
        # Fallback to seed
        return [0] * 28 + [1, -1]

    result = left_coeffs + right_coeffs

    # Verify
    total_frac = sum(Fraction(c, i + 1) for i, c in enumerate(result) if c != 0)
    if total_frac == 0:
        # Shouldn't happen, but fallback
        return [0] * 28 + [1, -1]

    return result

# EVOLVE-BLOCK-END
