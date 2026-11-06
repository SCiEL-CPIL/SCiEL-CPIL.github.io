"""Number-theoretic pair-collision search for powerful-number sum representations."""
import numpy as np
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form."""
    powerful = set()
    powerful.add(1)
    max_b = int(round(limit ** (1/3))) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = int(math.isqrt(limit // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    # Powerful numbers must be at most N - 1 <= 10^18 - 1, but we only need
    # b such that there exists a <= b with a + b <= 10^18 and a >= 1.
    # So b <= 10^18 - 1. However, generating up to 10^18 is infeasible.
    # Key insight: a <= b, so a <= (10^18 - b) and a <= b => a <= min(b, 10^18 - b).
    # For a to exist, we need b <= 10^18 - 1. But practically, b up to ~10^18 is huge.
    # However, the number of powerful numbers up to 10^18 is only ~2.6 * 10^12? No.
    # Powerful numbers up to X: count is ~ pi * sqrt(X) / 3 + O(X^(1/3)).
    # For X = 10^18: sqrt = 10^9, so ~ 10^9 / 3 ~ 3.3 * 10^8. That's too many to enumerate.
    #
    # Revised strategy: We cannot enumerate all powerful numbers up to 10^18.
    # Instead, use the FFT approach but with a much larger limit, OR use a
    # different parameterization.
    #
    # Actually, the current FFT approach with limit=100M finds N up to 200M.
    # But N can be up to 10^18! The current approach is fundamentally limited.
    #
    # New structural approach: Use the fact that powerful numbers have the form
    # a^2 * b^3. For a given N, we want to count pairs (a1^2*b1^3, a2^2*b2^3)
    # summing to N. This is hard to do directly.
    #
    # Alternative: Since the number of powerful numbers up to 10^18 is ~3.3*10^8,
    # we can't store them all. But we can use a probabilistic / sampling approach
    # or a smarter enumeration.
    #
    # Actually, let's reconsider. The count of powerful numbers up to X is
    # asymptotic to (pi/sqrt(3)) * sqrt(X) + O(X^(1/3)).
    # For X = 10^18: (pi/sqrt(3)) * 10^9 ≈ 1.814 * 10^9. That's ~2 billion.
    # Too many to enumerate in Python in 150s.
    #
    # So we need a different approach entirely. Let's use the FFT approach
    # but with a much larger limit. The issue is memory.
    #
    # With float32, a limit of 200M uses 200M * 4 = 800MB for the indicator,
    # plus FFT overhead. That might be too much.
    #
    # Let's try a hybrid: generate powerful numbers up to a moderate limit,
    # then use a meet-in-the-middle or direct pair enumeration for the
    # top candidates.
    #
    # Actually, the best structural change: instead of FFT over the full range,
    # enumerate pairs directly but with a smart bound. For each powerful a,
    # the corresponding b = N - a must also be powerful. If we fix N, we can
    # check all a. But we don't know N.
    #
    # Let's use the pair-collision approach but with a bounded range.
    # Generate powerful numbers up to L, then for each pair (a, b) with a <= b
    # and a + b <= 10^18, bucket by sum. But the number of pairs is ~ (2L)^2/2
    # which is too large.
    #
    # OK, new plan: Use the FFT approach with the largest feasible limit,
    # but also add a targeted search for large N values by checking
    # specific candidate N values that are known to have many representations.
    #
    # For now, let's just improve the FFT limit and add a fallback.

    # Determine feasible limit based on available memory/time
    # Try progressively larger limits
    for limit in [150_000_000, 100_000_000, 60_000_000]:
        try:
            powerful = _generate_powerful(limit)
            break
        except Exception:
            continue
    else:
        powerful = _generate_powerful(30_000_000)

    powerful_set = set(powerful)
    K = len(powerful)

    if K < 2:
        return {'N': 2, 'pairs': [[1, 1]]}

    # Use FFT convolution
    max_sum = 2 * limit
    n_fft = 1
    while n_fft < max_sum + 1:
        n_fft *= 2

    f = np.zeros(n_fft, dtype=np.float32)
    for val in powerful:
        if val < n_fft:
            f[val] = 1.0

    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=n_fft)
    conv_int = np.round(conv[:max_sum + 1]).astype(np.int64)

    # Compute pairs with a <= b
    even_s = np.arange(0, max_sum + 1, 2)
    half_s = even_s // 2
    is_powerful_half = (f[half_s] == 1.0)
    adjusted = conv_int.copy()
    adjusted[even_s[is_powerful_half]] += 1
    pairs_count = adjusted // 2
    if max_sum >= 2:
        pairs_count[:2] = 0

    # Find top candidates
    top_k = 200
    if len(pairs_count) > top_k:
        top_indices = np.argpartition(pairs_count, -top_k)[-top_k:]
    else:
        top_indices = np.arange(len(pairs_count))
    top_indices = top_indices[np.argsort(pairs_count[top_indices])[::-1]]

    best_N = 2
    best_count = 1
    best_pairs = [[1, 1]]

    for idx in top_indices:
        if time.time() > deadline - 10:
            break

        N = int(idx)
        if N < 2:
            continue

        count = 0
        exact_pairs = []
        for a in powerful:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set:
                count += 1
                exact_pairs.append([a, b])

        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = exact_pairs

        if pairs_count[idx] < best_count:
            break

    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
