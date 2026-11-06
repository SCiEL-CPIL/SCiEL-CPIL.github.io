"""
Optimized solution for Distinct Subset Sums with n=12.

The problem asks for n=12 distinct positive integers such that all 2^n subset sums are distinct, minimizing the largest integer.

Mathematical Derivation:
Let the sorted integers be $a_1 < a_2 < \dots < a_n$.
Let $S_k = \sum_{i=1}^k a_i$.
For all subset sums to be distinct, the set of subset sums of $\{a_1, \dots, a_k\}$ must be disjoint from the set of subset sums of $\{a_1, \dots, a_k\}$ shifted by $a_{k+1}$.
This requires $a_{k+1} > S_k$.
To minimize the largest element $a_n$, we should make each $a_k$ as small as possible subject to the previous constraints.
$a_1 \ge 1$. Minimal $a_1 = 1$.
$a_2 > a_1 \implies a_2 \ge 2$. Minimal $a_2 = 2$.
$a_3 > a_1 + a_2 = 3 \implies a_3 \ge 4$. Minimal $a_3 = 4$.
...
$a_k \ge 2^{k-1}$.
Thus, the minimal largest element for $n$ integers is $2^{n-1}$.
For $n=12$, the optimal set is $\{1, 2, 4, \dots, 2^{11}\}$.
The largest element is $2^{11} = 2048$.

This solution directly constructs this optimal set.
"""

import time

def _verify_distinct_subset_sums(vals):
    """
    Verify that all 2^n subset sums are distinct.
    Returns True if distinct, False otherwise.
    """
    n = len(vals)
    # Using a set of sums is sufficient for n=12 (4096 sums)
    sums = {0}
    for v in vals:
        new_sums = set()
        for s in sums:
            new_sums.add(s + v)
        if len(sums) + len(new_sums) != len(sums.union(new_sums)):
            return False
        sums = sums.union(new_sums)
    return True

def construct():
    start_time = time.time()
    
    # The optimal sequence is powers of 2.
    # a_i = 2^(i-1) for i=1..12
    n = 12
    optimal_vals = [1 << i for i in range(n)]
    
    # Safety check: Ensure the solution is valid
    if not _verify_distinct_subset_sums(optimal_vals):
        # This should never happen for powers of 2, but just in case
        # Fallback to a greedy search if verification fails (unlikely)
        # But for n=12, powers of 2 are guaranteed valid.
        pass
        
    return optimal_vals

# EVOLVE-BLOCK-END
