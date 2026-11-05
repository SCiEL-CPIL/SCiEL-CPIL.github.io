import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    n = 30

    # 1. Compute L = lcm(1, ..., 30)
    def lcm(a, b):
        return a * b // gcd(a, b)

    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)

    # 2. Compute weights w_i = L // i for i = 1..n
    w = [0] * (n + 1)
    for i in range(1, n + 1):
        w[i] = L // i

    # 3. Meet-in-the-middle: split indices 1..30 into two halves
    # Left: indices 1..15, Right: indices 16..30
    mid = n // 2  # 15

    # Generate all possible signed sums for left half
    # Each index can be -1, 0, or +1
    left_indices = list(range(1, mid + 1))
    right_indices = list(range(mid + 1, n + 1))

    def generate_sums(indices):
        """Generate all possible (sum, coeffs_dict) pairs for given indices."""
        results = {}
        num_indices = len(indices)
        # 3^num_indices possibilities
        for mask in range(3 ** num_indices):
            s = 0
            coeffs = {}
            temp = mask
            for j, idx in enumerate(indices):
                digit = temp % 3
                temp //= 3
                if digit == 0:
                    val = -1
                elif digit == 1:
                    val = 0
                else:
                    val = 1
                coeffs[idx] = val
                s += val * w[idx]
            # Keep track of the minimum absolute value for each sum
            if s not in results or abs(s) < abs(results[s][0]):
                # Store (abs_sum, coeffs) - but we want to keep all coeffs for each sum
                # Actually, we just need to know if a sum is achievable and with what coeffs
                results[s] = coeffs
        return results

    # This is too slow for 3^15 = 14M. Let's use a smarter approach.
    # Actually 3^15 = 14348907, which might be borderline. Let's try a different approach.
    # Use dynamic programming or a more efficient enumeration.

    # Better approach: enumerate all 3^15 left sums, store in a dict, then for each right sum,
    # look up the best left sum that makes total close to 0.

    # But 3^15 * 3^15 = 3^30 is way too big. We need meet-in-the-middle properly.

    # Meet-in-the-middle:
    # Generate all left sums (3^15 ≈ 14M) - this is feasible in ~1-2 min in Python? Maybe not.
    # Let's use a more efficient method.

    # Alternative: Since we want min |S| where S = sum_left + sum_right,
    # we can generate all left sums, sort them, then for each right sum, binary search
    # for the left sum that minimizes |left + right|.

    # 3^15 = 14348907. Generating these in Python might take ~30-60 seconds.
    # Let's try it with optimization.

    def gen_left_sums():
        """Generate all signed sums for left half (indices 1..15)."""
        sums = {}
        left_w = [w[i] for i in left_indices]
        num = len(left_w)
        # Recursive generation to avoid 3**num memory
        def rec(pos, current_sum, coeffs):
            if time.time() - start_time > deadline:
                return
            if pos == num:
                if current_sum not in sums:
                    sums[current_sum] = coeffs[:]
                return
            # -1
            coeffs.append(-1)
            rec(pos + 1, current_sum - left_w[pos], coeffs)
            coeffs.pop()
            # 0
            coeffs.append(0)
            rec(pos + 1, current_sum, coeffs)
            coeffs.pop()
            # +1
            coeffs.append(1)
            rec(pos + 1, current_sum + left_w[pos], coeffs)
            coeffs.pop()

        rec(0, 0, [])
        return sums

    def gen_right_sums():
        """Generate all signed sums for right half (indices 16..30)."""
        sums = {}
        right_w = [w[i] for i in right_indices]
        num = len(right_w)

        def rec(pos, current_sum, coeffs):
            if time.time() - start_time > deadline:
                return
            if pos == num:
                if current_sum not in sums:
                    sums[current_sum] = coeffs[:]
                return
            # -1
            coeffs.append(-1)
            rec(pos + 1, current_sum - right_w[pos], coeffs)
            coeffs.pop()
            # 0
            coeffs.append(0)
            rec(pos + 1, current_sum, coeffs)
            coeffs.pop()
            # +1
            coeffs.append(1)
            rec(pos + 1, current_sum + right_w[pos], coeffs)
            coeffs.pop()

        rec(0, 0, [])
        return sums

    # Try to generate left sums
    left_sums = gen_left_sums()

    if time.time() - start_time > deadline:
        # Fallback: return a simple nonzero solution
        fallback = [0] * n
        fallback[0] = 1
        return fallback

    # Sort left sums by value for binary search
    sorted_left = sorted(left_sums.keys())

    # Now iterate over right sums and find best match
    best_abs = None
    best_left_sum = None
    best_right_sum = None
    best_left_coeffs = None
    best_right_coeffs = None

    right_w = [w[i] for i in right_indices]
    num_right = len(right_w)

    def rec_right(pos, current_sum, coeffs):
        nonlocal best_abs, best_left_sum, best_right_sum
        nonlocal best_left_coeffs, best_right_coeffs

        if time.time() - start_time > deadline:
            return

        if pos == num_right:
            # Find left sum that minimizes |current_sum + left_sum|
            # Binary search for -current_sum in sorted_left
            target = -current_sum
            lo, hi = 0, len(sorted_left) - 1
            while lo <= hi:
                mid_idx = (lo + hi) // 2
                if sorted_left[mid_idx] < target:
                    lo = mid_idx + 1
                else:
                    hi = mid_idx - 1

            # Check candidates around the insertion point
            candidates = []
            if lo < len(sorted_left):
                candidates.append(sorted_left[lo])
            if hi >= 0:
                candidates.append(sorted_left[hi])

            for cand in candidates:
                total = current_sum + cand
                if total == 0:
                    continue
                abs_total = abs(total)
                if best_abs is None or abs_total < best_abs:
                    best_abs = abs_total
                    best_left_sum = cand
                    best_right_sum = current_sum
                    best_left_coeffs = left_sums[cand]
                    best_right_coeffs = coeffs[:]

            return

        # -1
        coeffs.append(-1)
        rec_right(pos + 1, current_sum - right_w[pos], coeffs)
        coeffs.pop()
        # 0
        coeffs.append(0)
        rec_right(pos + 1, current_sum, coeffs)
        coeffs.pop()
        # +1
        coeffs.append(1)
        rec_right(pos + 1, current_sum + right_w[pos], coeffs)
        coeffs.pop()

    rec_right(0, 0, [])

    # If we found a valid solution
    if best_left_coeffs is not None and best_right_coeffs is not None:
        result = [0] * n
        # Map left coeffs to indices 1..15
        for j, idx in enumerate(left_indices):
            result[idx - 1] = best_left_coeffs[j]
        # Map right coeffs to indices 16..30
        for j, idx in enumerate(right_indices):
            result[idx - 1] = best_right_coeffs[j]
        return result

    # Fallback
    fallback = [0] * n
    fallback[0] = 1
    return fallback
