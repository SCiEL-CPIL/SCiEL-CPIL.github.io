import time
import random

def construct():
    """
    Constructs a set of distinct integers in 1..256 such that subset sums
    with different cardinalities are distinct.
    Maximizes the set size.
    """
    deadline = time.time() + 50  # Safety margin for 60s limit
    
    def is_valid_ap(a, d, n):
        """Check if AP {a, a+d, ..., a+(n-1)d} is valid and within 1..256."""
        if a < 1 or a + (n - 1) * d > 256:
            return False
        # Check condition: min sum of k > max sum of k-1
        # M_k = k*a + d*k*(k-1)//2
        # M'_k-1 = (k-1)*a + d * sum_{j=n-k+1}^{n-1} j
        # sum_{j=n-k+1}^{n-1} j = (n-k+1 + n-1)*(k-1)//2 = (2n-k)*(k-1)//2
        for k in range(1, n + 1):
            m_k = k * a + d * k * (k - 1) // 2
            if k == 1:
                m_prev = 0
            else:
                m_prev = (k - 1) * a + d * (2 * n - k) * (k - 1) // 2
            if m_k <= m_prev:
                return False
        return True

    def check_set(S):
        """Check if a given set S is valid."""
        S = sorted(S)
        n = len(S)
        # Prefix sums for min sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + S[i]
        
        # Suffix sums for max sums
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + S[i]
            
        # M_k = prefix[k]
        # M'_k-1 = suffix[n-(k-1)] = suffix[n-k+1]
        for k in range(1, n + 1):
            min_sum_k = prefix[k]
            if k == 1:
                max_sum_k_minus_1 = 0
            else:
                max_sum_k_minus_1 = suffix[n - (k - 1)]
            if min_sum_k <= max_sum_k_minus_1:
                return False
        return True

    # 1. Try Arithmetic Progressions for N from 60 down to 1
    for n in range(60, 0, -1):
        if time.time() > deadline:
            break
        found = False
        # d=1 is the most likely to work for large n
        for d in range(1, 10):
            # a must be such that a + (n-1)d <= 256
            # and a >= 1
            # Also from analysis a > d*(k-1)(n-k) approx
            # Let's just iterate a
            max_a = 256 - (n - 1) * d
            if max_a < 1:
                continue
            # We want large a to satisfy the sum gap
            # Start from max_a downwards
            for a in range(max_a, 0, -1):
                if is_valid_ap(a, d, n):
                    return [a + i * d for i in range(n)]
        # If no AP found for this n, try a limited random search?
        # Given time, we might skip if n is small, but for n=32,33 it's worth a shot
        if n >= 32:
            # Random search for non-AP
            # Heuristic: pick numbers from the upper half
            for _ in range(50):
                if time.time() > deadline:
                    break
                # Pick n distinct numbers from 1..256
                # Bias towards larger numbers
                candidates = list(range(1, 257))
                random.shuffle(candidates)
                # Sort to ensure distinct and ordered
                # Just take first n
                S = sorted(candidates[:n])
                if check_set(S):
                    return S
    
    # Fallback: Return a known valid small set
    return list(range(128, 136))

# EVOLVE-BLOCK-END
