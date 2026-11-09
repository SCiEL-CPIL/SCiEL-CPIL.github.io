"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 30
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # Precompute scaled values: v[i] = L // (i+1) for coefficient of 1/(i+1)
    v = [L // (i + 1) for i in range(n)]
    
    # Split indices 0..29 into two halves of 15
    half1 = list(range(0, 15))
    half2 = list(range(15, 30))
    
    def gen_sums_with_coeffs(indices):
        """
        Generate all (sum, coeffs) for given indices.
        Returns list of (sum, tuple_of_coeffs).
        """
        results = [(0, ())]
        for idx in indices:
            val = v[idx]
            new_results = []
            for s, c in results:
                new_results.append((s - val, c + (-1,)))
                new_results.append((s, c + (0,)))
                new_results.append((s + val, c + (1,)))
            results = new_results
        return results

    def gen_sums_only(indices):
        """
        Generate all sums for given indices.
        Returns list of sums.
        """
        sums = [0]
        for idx in indices:
            val = v[idx]
            # Use list comprehension for speed
            sums = [s - val for s in sums] + sums + [s + val for s in sums]
        return sums

    # Generate sums for half 1
    sums1 = gen_sums_only(half1)
    
    if time.time() > deadline - 20:
        return [0] * 28 + [1, -1]
        
    sums1.sort()
    
    # Generate sums for half 2
    sums2 = gen_sums_only(half2)
    
    if time.time() > deadline - 10:
        return [0] * 28 + [1, -1]
        
    # Find best pair (s1, s2) such that |s1 + s2| is minimized and non-zero
    best_found_val = float('inf')
    best_s1 = None
    best_s2 = None
    
    # To speed up, we can also sort sums2 and use two-pointer, but binary search per element is fine
    # Since 3^15 ~ 14M, iterating over all s2 and doing binary search in s1 is O(3^15 * log(3^15))
    
    for s2 in sums2:
        if time.time() > deadline - 5:
            break
            
        target = -s2
        pos = bisect.bisect_left(sums1, target)
        
        candidates = []
        if pos < len(sums1):
            candidates.append(sums1[pos])
        if pos > 0:
            candidates.append(sums1[pos-1])
            
        for s1 in candidates:
            total = s1 + s2
            if total == 0:
                continue
            abs_total = abs(total)
            if abs_total < best_found_val:
                best_found_val = abs_total
                best_s1 = s1
                best_s2 = s2
                
                if best_found_val == 1:
                    break
        if best_found_val == 1:
            break
            
    if best_s1 is None:
        # Fallback: just return a simple nonzero sum
        return [0] * 29 + [1]

    # Now reconstruct coefficients for best_s1 using half1 indices
    # Use MITM on the 15 indices: split into 7 and 8
    def find_coeffs_for_target(target, indices):
        """
        Finds coefficients for the given indices that sum to target.
        Uses MITM on the indices themselves.
        """
        mid = len(indices) // 2
        left_idx = indices[:mid]
        right_idx = indices[mid:]
        
        left_sums = gen_sums_with_coeffs(left_idx)
        right_sums = gen_sums_with_coeffs(right_idx)
        
        # Build dict from right: sum -> coeffs (keep first occurrence)
        right_dict = {}
        for s, c in right_sums:
            if s not in right_dict:
                right_dict[s] = c
                
        for s_l, c_l in left_sums:
            needed = target - s_l
            if needed in right_dict:
                c_r = right_dict[needed]
                return c_l + c_r
                
        return None

    coeffs1 = find_coeffs_for_target(best_s1, half1)
    if coeffs1 is None:
        return [0] * 29 + [1]
        
    coeffs2 = find_coeffs_for_target(best_s2, half2)
    if coeffs2 is None:
        return [0] * 29 + [1]
        
    final_coeffs = list(coeffs1) + list(coeffs2)
    
    # Verify the sum is correct and nonzero
    total = sum(final_coeffs[i] * v[i] for i in range(n))
    if total == 0:
        # Should not happen, but fix it
        if final_coeffs[-1] == 0:
            final_coeffs[-1] = 1
        else:
            final_coeffs[-1] = -final_coeffs[-1]
        total = sum(final_coeffs[i] * v[i] for i in range(n))
    
    # Local search polish: try to improve further
    if time.time() < deadline - 5:
        current_sum = total
        current_abs = abs(current_sum)
        
        # Try single coefficient changes
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                old_c = final_coeffs[i]
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    delta = (new_c - old_c) * v[i]
                    new_sum = current_sum + delta
                    if new_sum != 0:
                        new_abs = abs(new_sum)
                        if new_abs < current_abs:
                            final_coeffs[i] = new_c
                            current_sum = new_sum
                            current_abs = new_abs
                            improved = True
                        else:
                            final_coeffs[i] = old_c
                    else:
                        final_coeffs[i] = old_c
        
        # Try double coefficient changes if time permits
        if time.time() < deadline - 3:
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                for i in range(n):
                    if time.time() > deadline - 3:
                        break
                    old_i = final_coeffs[i]
                    for j in range(i + 1, n):
                        if time.time() > deadline - 3:
                            break
                        old_j = final_coeffs[j]
                        for new_i in [-1, 0, 1]:
                            if new_i == old_i:
                                continue
                            for new_j in [-1, 0, 1]:
                                if new_j == old_j:
                                    continue
                                delta = (new_i - old_i) * v[i] + (new_j - old_j) * v[j]
                                new_sum = current_sum + delta
                                if new_sum != 0:
                                    new_abs = abs(new_sum)
                                    if new_abs < current_abs:
                                        final_coeffs[i] = new_i
                                        final_coeffs[j] = new_j
                                        current_sum = new_sum
                                        current_abs = new_abs
                                        improved = True
                                        break
                                    else:
                                        final_coeffs[i] = old_i
                                        final_coeffs[j] = old_j
                                else:
                                    final_coeffs[i] = old_i
                                    final_coeffs[j] = old_j
                        if improved:
                            break
                    if improved:
                        break

    # Final verification
    total = sum(final_coeffs[i] * v[i] for i in range(n))
    if total == 0:
        if final_coeffs[-1] == 0:
            final_coeffs[-1] = 1
        else:
            final_coeffs[-1] = -final_coeffs[-1]
            
    return final_coeffs

# EVOLVE-BLOCK-END
