"""
Program for the suite task: construct a large set in 1..128 with no triple a,b,c
such that a+b, a+c, b+c are all also in the set.

Approach: Hybrid construction with three parts:
  - A = set of odd numbers in [1, 31] (sum-free: sum of two odds is even, not in A)
  - B = [64, 128] (large interval where pair sums exceed 128)
  - G = {63} (a single carefully chosen element in the gap)

  Cross-checks:
  - For a, b in A: a+b is even and at most 31+29=60 < 64, so a+b not in B.
    Since a+b is even, a+b not in A. Also 63 is odd, so a+b != 63 (a+b is even).
    Thus a+b not in S.
  - For a, b in B: a+b >= 64+65=129 > 128, so a+b not in S.
  - For a in A, b in B: a+b in [65, 159]. If a+b in S, it must be in B (since > 31 and != 63).
    But we need three distinct members with all three pair sums in S.
    
  Let's verify all triple configurations:
  - All three in A: sums even, not in A. Safe.
  - All three in B: sums > 128, not in S. Safe.
  - All three in G: only one element, can't form triple. Safe.
  - Two in A (a, b), one in B (c):
    a+b: even, in [2, 60]. Not in A (odd), not in B (starts at 64), not 63 (even). Not in S. Safe.
  - Two in A (a, b), one in G (63):
    a+b: even, in [2, 60]. Not in S. Safe.
  - One in A (a), two in B (b, c):
    b+c: in [129, 254]. Not in S. Safe.
  - One in A (a), one in B (b), one in G (63):
    a+b: in [65, 159]. If <= 128, in B. If > 128, not in S.
    a+63: in [64, 94]. If in S, must be in B or = 63. a+63=63 implies a=0, impossible.
    So a+63 in B if a+63 >= 64, i.e., a >= 1. Always true. So a+63 in B.
    b+63: in [127, 191]. If <= 128, in B. If > 128, not in S.
    
    For the triple to be forbidden, we need a+b, a+63, b+63 all in S.
    a+63 is always in B (since a >= 1, a+63 >= 64, and a <= 31, a+63 <= 94 <= 128).
    So we need a+b in S and b+63 in S.
    a+b in S means a+b in B (since a+b >= 65 and a+b != 63 as it's >= 65).
    So a+b >= 64, which is always true (a >= 1, b >= 64, so a+b >= 65).
    And a+b <= 128? a <= 31, b <= 128, so a+b <= 159. If a+b <= 128, then a+b in B.
    b+63 in S means b+63 in B (since b+63 >= 127, and if <= 128, in B; if > 128, not in S).
    b+63 <= 128 means b <= 65.
    
    So the forbidden condition holds when:
      a+b <= 128 AND b <= 65
    i.e., b <= 65 and a <= 128 - b.
    
    For b = 64: a <= 64. Since a <= 31, always true. So a+64, a+63, 64+63=127 all in S?
    Wait: 64+63 = 127, which is in B. a+64 in B (since a+64 in [65, 95]). a+63 in B (in [64, 94]).
    So {a, 64, 63} is a forbidden triple for any a in A!
    
    This means we CANNOT include 63 if we include 64 in B.
    
    Let me reconsider. We need to either remove 64 from B or not include 63.
    
    Alternative: Remove 64 from B. Then B = [65, 128].
    Re-check triple (a, b, 63) with b in [65, 128]:
    b+63 in [128, 191]. b+63 <= 128 only if b=65. So b+63=128 in B.
    a+b in [66, 159]. a+b <= 128 if a <= 128-b.
    For b=65: a <= 63. Always true. So a+65 in B, a+63 in B, 65+63=128 in B.
    So {a, 65, 63} is forbidden for any a in A.
    
    So including 63 creates problems with small elements in B.
    
    Let me try a different approach: use a greedy algorithm to find a maximal valid set.
"""

def construct():
    # Use a greedy approach to build a large valid set
    # Start with all numbers 1..128, then remove elements that create forbidden triples
    
    # A set S is valid if there do NOT exist three distinct a, b, c in S
    # such that a+b, a+c, b+c are all in S.
    
    # We'll use a backtracking/greedy approach:
    # Try to include each number from 1 to 128 if it doesn't create a violation.
    
    # Actually, let's think about this differently.
    # The constraint is: no three distinct a, b, c with all three pair sums in S.
    
    # A known approach: use a sum-free-like construction.
    # Let's try: S = {x in 1..128 : x in some range or has some property}
    
    # Alternative: use the fact that if we take numbers from two disjoint intervals
    # where cross-sums fall outside both intervals, we can avoid the condition.
    
    # Let me try: A = odds in [1, 31], B = [66, 128]
    # Check: |A| = 16, |B| = 63. Total = 79.
    
    # Can we do better? Let's try a computational approach.
    
    # Greedy: start with empty set, add numbers 1..128 in order if valid.
    # But this might not be optimal.
    
    # Better: start with all numbers, remove minimally.
    
    # Actually, let's use the following insight:
    # If we take S to be a subset of [1, n] such that no three distinct elements
    # have all three pair sums in S, this is related to sum-free sets but stronger.
    
    # Let me try a different construction:
    # S = {x : x in [1, 43] and x is odd} union {x : x in [64, 128]}
    # |A| = 22 (odds 1..43), |B| = 65. Total = 87.
    
    # Check validity:
    # - All in A: sums even, not in A. Safe.
    # - All in B: sums > 128. Safe.
    # - Two in A (a, b), one in B (c):
    #   a+b: even, in [2, 84]. Not in A (odd). In B if >= 64.
    #   So a+b in B if a+b >= 64.
    #   a+c: in [65, 171]. In B if <= 128.
    #   b+c: in [65, 171]. In B if <= 128.
    #   For forbidden triple: need a+b in S, a+c in S, b+c in S.
    #   a+b in S means a+b in B (since a+b even, not in A). So a+b >= 64.
    #   a+c in S means a+c in B. So a+c >= 64 (always) and a+c <= 128.
    #   b+c in S means b+c in B. So b+c <= 128.
    #   So need: a+b >= 64, a+c <= 128, b+c <= 128.
    #   With a, b <= 43 (odd), c >= 64:
    #   a+c <= 128 means c <= 128 - a.
    #   b+c <= 128 means c <= 128 - b.
    #   a+b >= 64.
    #   Example: a=31, b=33, c=64: a+b=64 in B, a+c=95 in B, b+c=97 in B. Forbidden!
    
    # So that doesn't work.
    
    # Let me go back to the original approach and verify it carefully.
    # A = odds in [1, 31], B = [64, 128]. Size = 81.
    # We showed this is valid.
    
    # Can we add elements? Let's try adding 32.
    # S = A union B union {32}.
    # Check triples involving 32:
    # - (32, a, b) with a, b in A:
    #   32+a: even, in [33, 63]. Not in A (odd). In B if >= 64. 32+a <= 63, so not in B. Not in S. Safe.
    # - (32, a, c) with a in A, c in B:
    #   32+a: in [33, 63]. Not in S. Safe.
    # - (32, c, d) with c, d in B:
    #   c+d: in [129, 254]. Not in S. Safe.
    # - (32, a, b) with a, b in A: already checked.
    # So 32 is safe to add!
    
    # Can we add 33?
    # S = A union B union {32, 33}.
    # Check (33, a, b) with a, b in A:
    #   33+a: even? 33 is odd, a is odd, sum is even. In [34, 64].
    #   33+a in B if = 64. 33+a=64 means a=31. So 33+31=64 in B.
    #   33+b: similarly, 33+b in B if b=31.
    #   a+b: even, in [2, 60]. Not in S.
    #   So for triple (33, a, b): a+b not in S. Safe.
    # Check (33, a, c) with a in A, c in B:
    #   33+a: in [34, 64]. In B if = 64 (a=31).
    #   33+c: in [97, 161]. In B if <= 128.
    #   a+c: in [65, 159]. In B if <= 128.
    #   For forbidden: need 33+a in S, 33+c in S, a+c in S.
    #   33+a in S only if a=31 (gives 64 in B).
    #   So a=31. Then 33+c in S if c <= 95. a+c=31+c in S if c <= 97.
    #   So if c <= 95, all three in S. Forbidden!
    # So 33 is NOT safe if we have 31 in A and elements in B up to 95.
    
    # So we can add 32 but not 33.
    
    # Let's also try 30:
    # (30, a, b) with a, b in A:
    #   30+a: even, in [31, 61]. Not in A. In B if >= 64. 30+a <= 61, not in B. Not in S. Safe.
    # (30, a, c) with a in A, c in B:
    #   30+a: in [31, 61]. Not in S. Safe.
    # (30, c, d) with c, d in B:
    #   c+d > 128. Safe.
    # So 30 is safe.
    
    # Similarly, 28, 26, etc. (even numbers in [2, 32]) should be safe.
    # Let's verify: for even x in [2, 32]:
    # (x, a, b) with a, b in A: x+a even, in [3, 63]. Not in A. In B if >= 64. x+a <= 32+31=63. Not in B. Not in S.
    # (x, a, c) with a in A, c in B: x+a in [3, 63]. Not in S.
    # (x, c, d) with c, d in B: c+d > 128. Not in S.
    # So all even numbers in [2, 32] are safe to add!
    
    # Even numbers in [2, 32]: 2, 4, 6, ..., 32. That's 16 numbers.
    
    # So S = A (odds 1..31) union evens [2, 32] union B [64, 128]
    # = [1, 32] union [64, 128]
    # Size = 32 + 65 = 97.
    
    # Let's verify this is valid:
    # S = {1, 2, ..., 32} union {64, 65, ..., 128}
    
    # Check all triple configurations:
    # 1. All three in [1, 32]:
    #    a+b, a+c, b+c in [2, 64].
    #    For all three sums to be in S, they must be in [1, 32] or [64, 128].
    #    a+b, a+c, b+c are each in [2, 64].
    #    If any sum is in [33, 63], it's not in S.
    #    So we need a+b, a+c, b+c all in [1, 32] or all in [64, 128] (but max is 64, so only 64 possible).
    #    Case 1: all sums in [1, 32]. Then a+b <= 32, a+c <= 32, b+c <= 32.
    #      This implies a, b, c <= 16 (since a+b <= 32 and a, b >= 1).
    #      But we also need the sums to be in S, which they are if <= 32.
    #      So if a, b, c are all <= 16, the sums are in [2, 32] subset of S.
    #      Example: a=1, b=2, c=3: sums 3, 4, 5 all in S. Forbidden!
    
    # So [1, 32] is NOT sum-free in this sense. We have forbidden triples within [1, 32].
    
    # So we can't just take all of [1, 32].
    
    # Back to drawing board. The original construction with A=odds[1,31], B=[64,128] gives 81.
    # We can add even numbers [2, 32] but that creates triples within [1, 32].
    
    # Let me think about what subsets of [1, 32] are "triple-safe" in the sense that
    # no three distinct elements have all three pair sums also in the set.
    
    # This is equivalent to: the set T = [1, 32] part must not contain a, b, c with
    # a+b, a+c, b+c all in T.
    
    # A simple choice: T = odds in [1, 31]. Size 16. We know this works.
    # Can we do better?
    
    # Try T = [1, 8] union [17, 32]? Let me check size: 8 + 16 = 24.
    # But need to verify no forbidden triples.
    
    # This is getting complex. Let me just use the original construction with the verified 81.
    
    A = [x for x in range(1, 32, 2)]
    B = list(range(64, 129))
    S = sorted(set(A) | set(B))
    return [int(x) for x in S]
