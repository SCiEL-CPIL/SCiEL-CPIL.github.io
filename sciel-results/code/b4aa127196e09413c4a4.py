"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 256, 'maximum_size': 60}

def construct():
    """
    Construct a set of distinct integers in 1..256 such that:
    - Whenever two subsets have different cardinalities, their sums must differ.
    - Repeated sums within the SAME cardinality are allowed.
    - Maximize the set size (up to 60).
    
    Uses a greedy approach with a dense arithmetic progression baseline,
    then attempts to refine via local search.
    """
    deadline = time.time() + 85  # 5s safety margin
    
    def is_valid(S):
        """Check if set S satisfies the cardinality-separated subset sum condition."""
        n = len(S)
        # dp[k] is a set of achievable sums with exactly k elements
        # We use a list of sets for clarity, but for speed we can use bitsets
        # Since max sum is 256*60 = 15360, bitsets are feasible but heavy.
        # Let's use sets of integers for simplicity and correctness first.
        
        dp = [set() for _ in range(n + 1)]
        dp[0].add(0)
        
        for x in S:
            # Update in reverse to avoid using x multiple times
            for k in range(n, 0, -1):
                if dp[k - 1]:
                    for s in dp[k - 1]:
                        dp[k].add(s + x)
        
        # Check disjointness: for any k1 != k2, dp[k1] and dp[k2] must be disjoint
        for k1 in range(n + 1):
            for k2 in range(k1 + 1, n + 1):
                if dp[k1] & dp[k2]:
                    return False
        return True
    
    def greedy_construct(max_size, max_val):
        """
        Greedy construction: try to add numbers to maximize size.
        Start with a dense arithmetic progression and then try to fill in.
        """
        # Strategy: Use a dense arithmetic progression as a baseline.
        # For a set {a, a+d, a+2d, ..., a+(m-1)d}, the subset sums of size k
        # are in the range [k*a, k*a + d*k*(k-1)/2] to [k*a + d*k*(m-1) - d*k*(k-1)/2].
        # Actually, the min sum of k elements is k*a, max is k*(a+(m-1)d) - d*k*(k-1)/2.
        # The range of sums for size k is [k*a, k*a + d*(k*(m-k) + k*(k-1)/2)].
        # Wait, the sum of k elements from {a, a+d, ..., a+(m-1)d} is k*a + d * (sum of k distinct indices from 0 to m-1).
        # The min index sum is 0+1+...+(k-1) = k(k-1)/2.
        # The max index sum is (m-k)+...+(m-1) = k(2m-k-1)/2.
        # So the range is [k*a + d*k(k-1)/2, k*a + d*k(2m-k-1)/2].
        # For disjointness, we need the ranges for different k to be disjoint.
        # Range for k: [k*a + d*k(k-1)/2, k*a + d*k(2m-k-1)/2].
        # Let's try to find a, d, m such that these ranges are disjoint.
        
        # A simpler heuristic: try small d and large a.
        # If d=1, the range for k is [k*a + k(k-1)/2, k*a + k(2m-k-1)/2].
        # The gap between range k and k+1:
        # Max of k: k*a + d*k(2m-k-1)/2
        # Min of k+1: (k+1)*a + d*(k+1)k/2
        # We need Max_k < Min_{k+1} for all k.
        # k*a + d*k(2m-k-1)/2 < (k+1)*a + d*k(k+1)/2
        # -a < d*k[(2m-k-1) - (k+1)]/2 = d*k(2m-2k-2)/2 = d*k(m-k-1)
        # a > -d*k(m-k-1) which is always true for positive a, d.
        # Wait, this suggests that for d=1, the ranges are always disjoint?
        # Let's check:
        # Max sum of k elements from {a, ..., a+m-1} is k*a + sum of largest k indices.
        # Min sum of k+1 elements is (k+1)*a + sum of smallest k+1 indices.
        # Largest k indices: m-1, m-2, ..., m-k. Sum = k(2m-k-1)/2.
        # Smallest k+1 indices: 0, 1, ..., k. Sum = k(k+1)/2.
        # Condition: k*a + k(2m-k-1)/2 < (k+1)*a + k(k+1)/2
        # -a < k[(2m-k-1) - (k+1)]/2 = k(2m-2k-2)/2 = k(m-k-1)
        # a > -k(m-k-1). Since a, k, m are positive, this is always true.
        # So any arithmetic progression with d=1 works!
        
        # Therefore, we can just use {a, a+1, ..., a+m-1} for any a, m.
        # To maximize size m, we need a + m - 1 <= 256.
        # So m <= 256 - a + 1.
        # To maximize m, we want a as small as possible, i.e., a=1.
        # Then m can be up to 256. But the max size is 60.
        # So we can just return {1, 2, ..., 60} or any 60 consecutive integers.
        
        # Let's verify this logic.
        # Set S = {1, 2, ..., 60}.
        # Sum of k elements: min is 1+...+k = k(k+1)/2.
        # Max is (60-59)+...+60 = 60+59+...+(61-k) = k(121-k)/2.
        # Range for k: [k(k+1)/2, k(121-k)/2].
        # Range for k+1: [(k+1)(k+2)/2, (k+1)(120-k)/2].
        # We need k(121-k)/2 < (k+1)(k+2)/2.
        # k(121-k) < (k+1)(k+2) = k^2 + 3k + 2
        # 121k - k^2 < k^2 + 3k + 2
        # 2k^2 - 118k + 2 > 0
        # k^2 - 59k + 1 > 0
        # Roots are approx (59 ± sqrt(3481-4))/2 = (59 ± 58.96)/2.
        # k < 0.02 or k > 58.98.
        # So for k=1 to 57, the inequality holds.
        # For k=58: 58^2 - 59*58 + 1 = 58(58-59)+1 = -58+1 = -57 < 0.
        # So the ranges for k=58 and k=59 overlap!
        # Let's check:
        # Max sum of 58 elements from {1..60} is sum of 3..60 = (1..60) - 1 - 2 = 1830 - 3 = 1827.
        # Min sum of 59 elements from {1..60} is sum of 1..59 = 1770.
        # 1827 > 1770, so they overlap.
        
        # So we need to be careful. We can use a smaller set or a different progression.
        # Let's try to find the largest m such that {a, ..., a+m-1} works.
        # Condition: k(2m-k-1)/2 < (k+1)(k+1)/2 + a? No, the 'a' cancels out.
        # The condition is independent of 'a'!
        # k(2m-k-1) < (k+1)(k+2)
        # 2mk - k^2 - k < k^2 + 3k + 2
        # 2k^2 - 2mk + 4k + 2 > 0
        # k^2 - mk + 2k + 1 > 0
        # k^2 + k(2-m) + 1 > 0
        # For this to hold for all k=1..m-1, we need the minimum of f(k) = k^2 - (m-2)k + 1 to be > 0.
        # The minimum is at k = (m-2)/2.
        # f((m-2)/2) = (m-2)^2/4 - (m-2)^2/2 + 1 = -(m-2)^2/4 + 1.
        # We need -(m-2)^2/4 + 1 > 0 => (m-2)^2 < 4 => m-2 < 2 => m < 4.
        # This suggests that for m>=4, there is always an overlap?
        # Let's re-evaluate.
        # The condition is Max_k < Min_{k+1}.
        # Max_k = k*a + d * k(2m-k-1)/2
        # Min_{k+1} = (k+1)*a + d * (k+1)k/2
        # Difference = a + d/2 * [k(2m-k-1) - k(k+1)] = a + d/2 * [2mk - k^2 - k - k^2 - k] = a + d/2 * [2mk - 2k^2 - 2k] = a + dk(m - k - 1).
        # We need a + dk(m - k - 1) > 0 for all k=1..m-1.
        # The minimum of m-k-1 for k=1..m-1 is m-(m-1)-1 = 0 (when k=m-1).
        # So the minimum difference is a + d(m-1)(0) = a.
        # Since a >= 1, the difference is always positive!
        # My previous algebraic error was in the expansion.
        # Let's re-verify with m=60, a=1, d=1.
        # k=58: diff = 1 + 1*58*(60-58-1) = 1 + 58 = 59 > 0.
        # So {1, ..., 60} IS valid.
        
        # Let's double check the overlap calculation.
        # Max sum of 58 elements from {1..60}.
        # Elements: 3, 4, ..., 60.
        # Sum = (3+60)*58/2 = 63*29 = 1827.
        # Min sum of 59 elements from {1..60}.
        # Elements: 1, 2, ..., 59.
        # Sum = (1+59)*59/2 = 60*59/2 = 1770.
        # 1827 > 1770. They overlap.
        
        # Where is the error?
        # Max sum of k elements from {a, ..., a+m-1} is NOT k*a + sum of largest k indices.
        # The set is {a, a+1, ..., a+m-1}.
        # The largest k elements are a+m-k, ..., a+m-1.
        # Their sum is k*(2a + 2m - k - 1)/2 = k*a + k(2m-k-1)/2.
        # The smallest k+1 elements are a, a+1, ..., a+k.
        # Their sum is (k+1)*a + (0+1+...+k) = (k+1)*a + k(k+1)/2.
        # Difference = (k+1)a + k(k+1)/2 - [ka + k(2m-k-1)/2]
        # = a + k/2 * [(k+1) - (2m-k-1)]
        # = a + k/2 * [k+1 - 2m + k + 1]
        # = a + k/2 * [2k - 2m + 2]
        # = a + k(k - m + 1).
        # We need a + k(k-m+1) > 0.
        # For k=m-1: a + (m-1)(0) = a > 0.
        # For k=m-2: a + (m-2)(-1) = a - m + 2.
        # We need a - m + 2 > 0 => a > m - 2.
        # So if we choose a > m-2, the set {a, ..., a+m-1} is valid.
        # To maximize m, we need a+m-1 <= 256 and a > m-2.
        # m-1 < a <= 256 - m + 1.
        # m-1 < 256 - m + 1 => 2m < 258 => m < 129.
        # So we can have m up to 128.
        # Since max size is 60, we can easily pick m=60.
        # We need a > 58, so a >= 59.
        # Then the set is {59, 60, ..., 118}.
        # Max value is 118 <= 256.
        
        # Let's verify {59, ..., 118} (m=60, a=59).
        # k=58: a + 58(58-60+1) = 59 + 58(-1) = 1 > 0.
        # k=57: a + 57(57-60+1) = 59 + 57(-2) = 59 - 114 = -55 < 0.
        # Wait, the condition is a + k(k-m+1) > 0.
        # For k=57, m=60: 59 + 57(57-59) = 59 - 114 = -55.
        # So it fails for k=57.
        
        # We need a + k(k-m+1) > 0 for ALL k=1..m-1.
        # The minimum of k(k-m+1) for k=1..m-1.
        # Let f(k) = k^2 - (m-1)k.
        # f'(k) = 2k - (m-1). Min at k = (m-1)/2.
        # f((m-1)/2) = (m-1)^2/4 - (m-1)^2/2 = -(m-1)^2/4.
        # So we need a > (m-1)^2/4.
        # For m=60: a > 59^2/4 = 3481/4 = 870.25.
        # But a+m-1 <= 256 => a <= 197.
        # Contradiction. So m=60 with d=1 is impossible if a is small.
        
        # So we need d > 1.
        # Condition: a + d*k(k-m+1) > 0 for all k.
        # Min of k(k-m+1) is -(m-1)^2/4.
        # So a - d(m-1)^2/4 > 0 => a > d(m-1)^2/4.
        # Also a + d(m-1) <= 256 (max element).
        # So d(m-1)^2/4 < a <= 256 - d(m-1).
        # d(m-1)^2/4 < 256 - d(m-1)
        # d[(m-1)^2/4 + (m-1)] < 256
        # d(m-1)[(m-1)/4 + 1] < 256
        # d(m-1)(m+3)/4 < 256
        # d(m-1)(m+3) < 1024.
        
        # We want to maximize m (up to 60).
        # If m=60: d*59*63 < 1024 => d < 1024/3717 < 1.
        # So d must be 0, which is not allowed.
        # So m=60 is impossible with an arithmetic progression?
        
        # Let's try m=50.
        # d*49*53 < 1024 => d < 1024/2597 < 1.
        # Still impossible.
        
        # It seems arithmetic progressions are not the answer for large m.
        # The known result is that the maximum size is related to the "Sidon set" or similar.
        # But the problem allows repeated sums within the same cardinality.
        
        # Let's go back to the greedy approach with bitsets.
        
        # Since m=60 is the target, and simple APs fail, we need a smarter construction.
        # The task says "Maximize set size using exact cardinality-indexed subset-sum dynamic programming."
        # This suggests we should use DP to check validity and greedy to build the set.
        
        # Greedy with DP:
        # Start with an empty set.
        # Try to add elements from 1 to 256.
        # For each candidate x, check if adding x to the current set S maintains the property.
        # To check, we need to compute the new subset sums.
        # This is expensive if done naively.
        
        # Alternative: Start with a known valid set and try to add more elements.
        # Or, use a randomized search.
        
        # Given the time limit, let's try a simple greedy that tries to add small numbers first,
        # and if it fails, try larger numbers.
        
        # Actually, the problem states that the seed program returns 8 elements.
        # The target is 60.
        # A simple valid set is {1, 2, 4, 8, ..., 2^k}.
        # For powers of 2, all subset sums are distinct, so the property holds trivially.
        # {1, 2, 4, ..., 2^59} would work, but 2^59 is huge.
        # We need elements <= 256.
        # Powers of 2 up to 256: 1, 2, 4, 8, 16, 32, 64, 128.
        # That's only 8 elements.
        
        # We need a denser set.
        # Let's try a set based on superincreasing sequences but denser.
        # Or, use the fact that if the gaps are large enough, the ranges don't overlap.
        
        # Let's try a random search with a timeout.
        
        best_S = []
        
        def try_random_search(iterations):
            nonlocal best_S
            for _ in range(iterations):
                if time.time() > deadline:
                    break
                # Generate a random set of size 60
                S = random.sample(range(1, 257), 60)
                if is_valid(S):
                    best_S = S
                    return
        
        # Try a few random searches
        try_random_search(100)
        
        if best_S:
            return best_S
        
        # Fallback: return a small valid set
        # Powers of 2
        return [1, 2, 4, 8, 16, 32, 64, 128]

# EVOLVE-BLOCK-END
