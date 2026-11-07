"""
Program for the suite task: construct a large set in 1..128 with no triple a,b,c
such that a+b, a+c, b+c are all also in the set.

Approach: Use a hybrid construction:
  - A = set of odd numbers in [1, 31] (sum-free: sum of two odds is even, not in A)
  - B = [64, 128] (large interval where pair sums exceed 128)
  - C = {32} (single element in the gap [32, 63])

  Verification:
  1. Triples within A: Sums are even, not in A. Safe.
  2. Triples within B: Sums > 128, not in S. Safe.
  3. Triples within C: |C|=1, no triple. Safe.
  4. Two in A, one in B: Sum of two A is even <= 60, not in A or B. Safe.
  5. Two in B, one in A: Sum of two B > 128. Safe.
  6. One in A, one in B, one in C (x=32):
     Let a in A, b in B, x=32.
     Sums: a+b, a+32, b+32.
     a+32 is in [33, 63]. Not in A (odds <= 31) and not in B (>= 64). So a+32 not in S.
     Thus condition fails. Safe.
     
  Size: |A| = 16, |B| = 65, |C| = 1. Total = 82.
"""

def construct():
    # A: odd numbers from 1 to 31
    A = [x for x in range(1, 32, 2)]
    # B: all integers from 64 to 128
    B = list(range(64, 129))
    # C: add 32
    C = [32]
    
    # Combine and ensure distinct
    S = sorted(set(A) | set(B) | set(C))
    
    # Convert to Python ints and return as list
    return [int(x) for x in S]
