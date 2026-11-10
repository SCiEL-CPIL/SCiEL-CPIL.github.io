import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin before 90s limit

    # Phase 1: Compute sigma up to N using NumPy sieve
    N = 10_000_000
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        if i % 100000 == 0:
            if time.time() > deadline - 40:
                N = i
                break

    if time.time() > deadline - 35:
        return [11, 6]

    # Phase 2: Sort indices by sigma value and find best pair in each group
    sigma_vals = sigma[1:N+1]
    sorted_indices = np.argsort(sigma_vals, kind='stable')
    sorted_sigmas = sigma_vals[sorted_indices]

    best_a, best_b = 11, 6
    best_num = abs(2*11 - 3*6)  # = 4
    best_den = 2*6  # = 12

    i = 0
    while i < N:
        if time.time() > deadline - 10:
            break
        s = sorted_sigmas[i]
        j = i + 1
        while j < N and sorted_sigmas[j] == s:
            j += 1
        if j - i >= 2:
            nums = sorted_indices[i:j] + 1
            nums_sorted = np.sort(nums)
            nlen = len(nums_sorted)
            for k in range(nlen):
                b = int(nums_sorted[k])
                target_a = (3 * b) // 2
                lo, hi = 0, nlen - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if nums_sorted[mid] < target_a:
                        lo = mid + 1
                    else:
                        hi = mid
                for idx in (lo, lo - 1):
                    if 0 <= idx < nlen:
                        a = int(nums_sorted[idx])
                        if a == b:
                            continue
                        num = abs(2*a - 3*b)
                        den = 2*b
                        if num * best_den < best_num * den:
                            best_num = num
                            best_den = den
                            best_a, best_b = a, b
        i = j

    # Phase 3: Targeted search using ratio constraint 2a - 3b = k
    # For small |k|, the error |a/b - 3/2| = |k|/(2b) is small when b is large.
    # We search for pairs (a, b) with a <= 10^7, b <= 10^7, sigma(a) = sigma(b),
    # and 2a - 3b = k for small k values.
    if time.time() < deadline - 5:
        # For each small k, find candidate b values and check if a = (3b+k)/2 is integer
        # and within range, then verify sigma(a) == sigma(b)
        for k in range(-20, 21):
            if time.time() > deadline - 5:
                break
            # a = (3b + k) / 2 must be integer, so 3b + k must be even
            # 3b + k even => b + k even (since 3b has same parity as b)
            # So b must have same parity as k
            start_b = 1
            if k % 2 != 0:
                start_b = 2  # b must be even if k is odd? No: 3b+k even means b+k even
                # Actually 3b+k even: if k odd, b must be odd; if k even, b must be even
                # Wait: 3b+k even. 3b has same parity as b. So b+k even.
                # If k is odd, b must be odd. If k is even, b must be even.
            # b range: a = (3b+k)/2 <= 10^7 => 3b+k <= 20000000 => b <= (20000000-k)/3
            # Also a >= 1 => 3b+k >= 2 => b >= (2-k)/3
            # b >= 1
            max_b = min(10_000_000, (20_000_000 - k) // 3)
            min_b = max(1, (2 - k + 1) // 3)  # ceil((2-k)/3)
            
            # Adjust min_b for parity
            if k % 2 != 0:
                # b must be odd
                if min_b % 2 == 0:
                    min_b += 1
                step = 2
            else:
                # b must be even
                if min_b % 2 != 0:
                    min_b += 1
                step = 2
            
            # For each b in range, compute a and check sigma
            # This could be slow if range is large, so limit the number of checks
            checks = 0
            for b in range(min_b, max_b + 1, step):
                if time.time() > deadline - 5:
                    break
                if checks > 50000:
                    break
                checks += 1
                a = (3 * b + k) // 2
                if a < 1 or a > 10_000_000:
                    continue
                if a == b:
                    continue
                # Check if both are within our computed sigma range
                if a <= N and b <= N:
                    if sigma[a] == sigma[b]:
                        num = abs(2*a - 3*b)
                        den = 2*b
                        if num * best_den < best_num * den:
                            best_num = num
                            best_den = den
                            best_a, best_b = a, b

    return [int(best_a), int(best_b)]
