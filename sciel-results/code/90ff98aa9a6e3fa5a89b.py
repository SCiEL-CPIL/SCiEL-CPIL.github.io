"""
Rewritten program: Cantor-set style base construction + repair to maximize
cardinality of a set S ⊆ [1,512] with no element being the exact arithmetic mean
of any subset (size ≥ 2) of OTHER elements.

Key insight from cell findings: a base-3 Cantor-style set (digits 0/1 only) is
3-AP-free and has 64 elements in [1,512] (2^6 = 64, since 3^6=729>512, but we
can use up to 6 ternary digits: values are sums of distinct powers of 3 up to
3^5=243, max value = 1+3+9+27+81+243 = 364). We need 48 elements.

However, the constraint is stronger than 3-AP-free: no element can be the mean
of ANY subset of ≥2 others. For a subset T of size k, the mean is (sum(T))/k.
So we need: for every a ∈ S and every subset T ⊆ S\{a}, |T|≥2: sum(T) ≠ k*a.

The Cantor set (base-3, digits 0/1) has the property that any sum of distinct
elements is unique (like binary representation in base 3). This makes it very
unlikely that sum(T) = k*a for random T, but we still need to verify.

Strategy:
1. Generate the 64-element Cantor set C = {sum of distinct powers of 3, 6 terms}.
2. Check if C itself satisfies the constraint. If yes, pick any 48.
3. If not, use repair: start with C, remove violating elements, then greedily
   add back other numbers from [1,512] to reach 48.

For efficiency in verification, we use a fast check:
- For a given set S and element a, we need to check if there exists a subset T ⊆ S\{a}
  with |T|≥2 and sum(T) = |T|*a.
- This is equivalent to: there exist distinct t1,...,tk ∈ S\{a}, k≥2, such that
  sum(ti - a) = 0.
- Let vi = ti - a. We need a non-empty subset of {vi : ti ∈ S\{a}} of size ≥2 summing to 0.

For sets of size ~48, checking all subsets is infeasible. But we can use the fact
that the Cantor set has strong additive structure.

Alternative faster approach: For each a in S, compute all subset sums of {ti - a : ti in S\{a}}
using dynamic programming over bitsets, checking if 0 is achievable with ≥2 elements.

But for 47 elements, subset-sum DP over values: the values ti - a range from about -511 to 511.
We can use a bitset of achievable sums. The range of sums is [-47*511, 47*511] ≈ [-24017, 24017].
That's about 48000 bits, which is manageable.

Let me implement this.
"""

import time
import random

def _is_valid_set(S):
    """Check if no element in S is the mean of any subset of ≥2 other elements."""
    n = len(S)
    if n < 3:
        return True
    S_set = set(S)
    for a in S:
        others = [x for x in S if x != a]
        # We need: no subset T ⊆ others, |T|≥2, with sum(T) = |T|*a
        # Equivalent: no subset of {x - a for x in others} of size ≥2 sums to 0
        vals = [x - a for x in others]
        m = len(vals)
        if m < 2:
            continue
        # Use DP: dp[k] is a bitset where bit s is set if sum s is achievable with k elements
        # Sums range from -m*max_abs to m*max_abs
        max_abs = max(abs(v) for v in vals) if vals else 0
        min_sum = -m * max_abs
        max_sum = m * max_abs
        offset = -min_sum  # shift so indices are non-negative
        size = max_sum - min_sum + 1
        # dp[k] is an integer bitset
        dp = [0] * (m + 1)
        dp[0] = 1 << offset  # sum 0 with 0 elements
        for v in vals:
            # update in reverse
            for k in range(m - 1, -1, -1):
                if dp[k]:
                    # shift by v
                    if v >= 0:
                        dp[k+1] |= (dp[k] << v)
                    else:
                        dp[k+1] |= (dp[k] >> (-v))
            # mask to keep only relevant bits
            mask = (1 << size) - 1
            for k in range(m + 1):
                dp[k] &= mask
        # Check if any dp[k] for k>=2 has bit at offset (sum=0) set
        for k in range(2, m + 1):
            if dp[k] & (1 << offset):
                return False
    return True


def _fast_check_addition(S, candidate):
    """Check if adding candidate to S keeps the set valid. Only check constraints involving candidate."""
    # Check: is candidate the mean of any subset of ≥2 elements in S?
    # And: for each a in S, is a the mean of any subset of ≥2 elements in S ∪ {candidate} 
    #       that includes candidate?
    # Part 1: candidate = mean of subset of S
    #   sum(T) = |T| * candidate for some T ⊆ S, |T|≥2
    # Part 2: for each a in S, a = mean of subset of (S\{a}) ∪ {candidate} that includes candidate
    #   i.e., (sum(T) + candidate) = (|T|+1) * a for some T ⊆ S\{a}, |T|≥1
    #   => sum(T) - |T|*a = (|T|+1)*a - candidate - |T|*a = a - candidate
    #   => sum(T) = |T|*a + (a - candidate)
    
    # Part 1 check
    if len(S) >= 2:
        vals = [x - candidate for x in S]
        m = len(vals)
        max_abs = max(abs(v) for v in vals)
        min_sum = -m * max_abs
        max_sum = m * max_abs
        offset = -min_sum
        size = max_sum - min_sum + 1
        dp = [0] * (m + 1)
        dp[0] = 1 << offset
        for v in vals:
            for k in range(m - 1, -1, -1):
                if dp[k]:
                    if v >= 0:
                        dp[k+1] |= (dp[k] << v)
                    else:
                        dp[k+1] |= (dp[k] >> (-v))
            mask = (1 << size) - 1
            for k in range(m + 1):
                dp[k] &= mask
        for k in range(2, m + 1):
            if dp[k] & (1 << offset):
                return False
    
    # Part 2 check: for each a in S, check if a is mean of subset including candidate
    for a in S:
        # Need: exists T ⊆ S\{a}, |T|≥1, such that sum(T) + candidate = (|T|+1)*a
        # => sum(T) = (|T|+1)*a - candidate = |T|*a + a - candidate
        # Let d = a - candidate
        # sum(T) - |T|*a = d
        # Let wi = si - a for si in S\{a}, need sum(wi) = d for some non-empty subset
        others = [x for x in S if x != a]
        if not others:
            continue
        d = a - candidate
        vals = [x - a for x in others]
        m = len(vals)
        max_abs = max(abs(v) for v in vals)
        min_sum = -m * max_abs
        max_sum = m * max_abs
        offset = -min_sum
        size = max_sum - min_sum + 1
        dp = [0] * (m + 1)
        dp[0] = 1 << offset
        for v in vals:
            for k in range(m - 1, -1, -1):
                if dp[k]:
                    if v >= 0:
                        dp[k+1] |= (dp[k] << v)
                    else:
                        dp[k+1] |= (dp[k] >> (-v))
            mask = (1 << size) - 1
            for k in range(m + 1):
                dp[k] &= mask
        # Check if sum = d is achievable with k≥1
        target_idx = d + offset
        if 0 <= target_idx < size:
            for k in range(1, m + 1):
                if dp[k] & (1 << target_idx):
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Generate Cantor set: sums of distinct powers of 3
    powers3 = [3**i for i in range(6)]  # 1, 3, 9, 27, 81, 243
    cantor = []
    for mask in range(1, 1 << 6):  # skip 0
        s = sum(powers3[i] for i in range(6) if mask & (1 << i))
        if s <= 512:
            cantor.append(s)
    cantor.sort()
    # cantor has 63 elements (all non-empty subsets of 6 powers)
    
    # Try to use as many Cantor elements as possible
    S = []
    for x in cantor:
        if len(S) >= 48:
            break
        if time.time() > deadline:
            break
        if _fast_check_addition(S, x):
            S.append(x)
    
    # If we have 48, return
    if len(S) >= 48:
        return S[:48]
    
    # Otherwise, try to add more from [1,512] not in cantor
    remaining = [x for x in range(1, 513) if x not in set(S)]
    random.shuffle(remaining)
    for x in remaining:
        if len(S) >= 48:
            break
        if time.time() > deadline:
            break
        if _fast_check_addition(S, x):
            S.append(x)
    
    return S[:48]


if __name__ == "__main__":
    result = construct()
    print(f"Cardinality: {len(result)}")
    print(f"Set: {result}")
