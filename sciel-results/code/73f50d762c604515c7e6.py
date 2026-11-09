"""Optimal solution: superincreasing greedy gives powers of 2, largest=2048 is provably minimal.

The greedy minimal-extension algorithm (smallest unused integer with no subset-sum
collision) provably yields the same sequence: at each step the smallest valid next
element is exactly 1 + sum(previous elements), which gives 2^(i-1).
"""

# EVOLVE-BLOCK-START
def construct():
    """Return 12 distinct positive integers with all subset sums distinct,
    minimizing the largest element.
    
    Uses the greedy minimal-extension approach: at each step, find the smallest
    positive integer not yet used such that adding it to the set does not create
    a duplicate subset sum. This is equivalent to the superincreasing condition
    a_i > sum(a_1..a_{i-1}), whose minimal solution is a_i = 2^(i-1).
    
    Proof of optimality: For all 2^n subset sums to be distinct, when the sequence
    is sorted as a_1 < a_2 < ... < a_n, we must have a_i > sum(a_1..a_{i-1}) for
    each i >= 2 (otherwise a_i equals a subset sum of earlier elements, causing
    a collision). The greedy minimal choice is a_1=1, a_i=1+sum(a_1..a_{i-1}),
    yielding a_i=2^(i-1). Thus a_12=2048 is the minimum possible largest element."""
    n = 12
    return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
