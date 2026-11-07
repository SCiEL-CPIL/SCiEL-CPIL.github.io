"""
Program for the suite task: construct a large set in 1..128 with no triple a,b,c
such that a+b, a+c, b+c are all also in the set.

Approach: Use a hybrid construction:
  - A = set of odd numbers in [1, 31] (sum-free: sum of two odds is even, not in A)
  - B = [64, 128] (large interval where pair sums exceed 128)
  
  Cross-check: For a, b in A, a+b is even and at most 31+29=60 < 64, so a+b not in B.
  Since a+b is even, a+b not in A. So a+b not in S.
  For a, b in B, a+b >= 64+65=129 > 128, so a+b not in S.
  For a in A, b in B: a+b in [65, 159]. If a+b in S, it must be in B (since > 31).
  But we need three distinct members a, b, c with all three pair sums in S.
  
  Let's verify more carefully:
  - If all three in B: sums > 128, not in S. Safe.
  - If all three in A: sums even, not in A. Safe.
  - If two in A (say a, b) and one in B (say c):
    a+b: even, in [2, 60]. Not in A (A is odd), not in B (B starts at 64). So a+b not in S.
    Thus the condition "a+b in S" fails. Safe.
  - If one in A (say a) and two in B (say b, c):
    b+c: in [129, 254]. Not in S. Safe.
    So the condition "b+c in S" fails. Safe.
  
  Therefore the set is valid.
  
  Size: |A| = 16 (odds from 1 to 31), |B| = 65 (64 to 128). Total = 81.
"""

def construct():
    # A: odd numbers from 1 to 31
    A = [x for x in range(1, 32, 2)]
    # B: all integers from 64 to 128
    B = list(range(64, 129))
    
    # Combine and ensure distinct
    S = sorted(set(A) | set(B))
    
    # Convert to Python ints and return as list
    return [int(x) for x in S]
