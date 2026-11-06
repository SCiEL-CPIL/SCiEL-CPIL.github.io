"""Rewritten program for the reciprocal-subsum-free set task.

Strategy:
  * The condition is: for every selected element x, 1/x != sum_{y in S'} 1/y for any
    subset S' of the OTHER selected elements.
  * Equivalently, x * (sum of reciprocals of some subset of S\{x}) != 1.
  * We search for large subsets of [1..24] satisfying this.
  * Use greedy construction with multiple orderings + local improvement (swap/add/remove).
  * Validate exactly using Fraction arithmetic to avoid float errors.
  * Meet-in-the-middle for subset-sum check when needed, but for n<=24 we can use
    a smarter check: for each target x, check if 1/x is in the set of all subset sums
    of reciprocals of S\{x}. Since |S\{x}| <= 23, subset sums are 2^23 ~ 8M which is
    too many to enumerate fully. Instead, we use a greedy/necessary condition or
    a bounded DFS with pruning.
  * Actually, for each x in S, we need to check if there exists a nonempty subset T of
    S\{x} such that sum_{y in T} 1/y = 1/x. This is a subset-sum problem on rationals.
  * For small sets (say <= 20 elements), we can do a DFS with memoization or just
    iterate all subsets. For larger sets, we need pruning.
  * Alternative: use the fact that if we sort elements, the largest reciprocal is 1/1=1,
    smallest is 1/24. The sum of all reciprocals of [2..24] is H_24 - 1 ~ 3.77.
    So 1/x for x>=1 is at most 1, and we're checking if 1/x equals a subset sum.
  * Key insight: if the set is "superincreasing" in reciprocal space (each reciprocal
    is larger than the sum of all smaller reciprocals in the set), then no subset sum
    can equal any single reciprocal. But that's very restrictive.
  * Better: use randomized local search. Start with a valid set, try to add elements,
    remove elements, swap elements to increase size.

Let me think about what sets are valid. The trivial valid set is {1..24} minus some elements.
The seed returns [13..24] (12 elements). Can we do better?

Let's think: if we include 1, then 1/1 = 1. We need no subset of other elements to sum to 1.
The sum of 1/2+1/3+...+1/24 is H_24 - 1 ≈ 3.77, so it's possible. But we need an EXACT
subset to sum to 1.

For a random large set, it's unlikely that 1/x equals a subset sum of other reciprocals,
because the subset sums are dense but the targets are sparse. So we might be able to get
close to 24 elements.

Let me implement:
1. A validator that checks a set S: for each x in S, check if 1/x is representable as a
   subset sum of {1/y : y in S, y != x}.
2. For the subset sum check, use a DFS with pruning: sort the reciprocals, and use the
   fact that if the remaining sum can't reach the target, prune.
3. Greedy construction: start empty, add elements in some order if valid.
4. Local search: try to add any missing element; if can't, try swaps.
"""

import time
import random
from fractions import Fraction

DEADLINE = 75.0
_START = None

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE


def _can_form_target(target, vals):
    """Check if target can be formed as a subset sum of vals (nonempty subset).
    vals is a list of Fractions. Use DFS with pruning.
    """
    if not vals:
        return target != 0
    # Sort in descending order for better pruning
    vals = sorted(vals, reverse=True)
    total = sum(vals)
    if total < target:
        return False
    # If target is 0, we need empty subset, but we require nonempty, so return False
    # Actually the problem says "subset of the OTHER selected elements" - does it allow empty?
    # "no reciprocal of a selected element may equal the sum of reciprocals of a subset of the OTHER selected elements"
    # A subset can be empty, sum = 0. Since 1/x > 0, empty subset never matches. So we only care about nonempty.
    
    n = len(vals)
    # For small n, just do DFS
    # For larger n, we can use a meet-in-the-middle or just DFS with pruning
    # Since n <= 23, and we have pruning, DFS should work for most cases
    
    # Use a recursive helper with memoization on (index, remaining) - but remaining is Fraction, hard to memo
    # Just do DFS with pruning
    
    def dfs(idx, remaining):
        if remaining == 0:
            return True
        if idx >= n:
            return False
        if remaining < 0:
            return False
        # Pruning: if sum of all remaining vals < remaining, can't reach
        # Precompute suffix sums
        if idx < n:
            if suffix_sums[idx] < remaining:
                return False
        # Option 1: skip vals[idx]
        if dfs(idx + 1, remaining):
            return True
        # Option 2: take vals[idx]
        if dfs(idx + 1, remaining - vals[idx]):
            return True
        return False
    
    # Precompute suffix sums
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + vals[i]
    
    return dfs(0, target)


def is_valid(S):
    """Check if set S is reciprocal-subsum-free."""
    if not S:
        return True
    S_set = set(S)
    for x in S:
        target = Fraction(1, x)
        others = [Fraction(1, y) for y in S_set if y != x]
        if _can_form_target(target, others):
            return False
    return True


def construct():
    global _START
    _START = time.time()
    
    best = list(range(13, 25))  # seed fallback
    best_size = len(best)
    
    # Try greedy construction with different orderings
    elements = list(range(1, 25))
    
    orderings = [
        elements,
        list(reversed(elements)),
        sorted(elements, key=lambda x: random.random()),
        list(range(24, 0, -1)),
        list(range(1, 25, 2)) + list(range(2, 25, 2)),  # odds then evens
        list(range(2, 25, 2)) + list(range(1, 25, 2)),  # evens then odds
    ]
    
    for ordering in orderings:
        if _deadline_hit():
            break
        S = []
        for x in ordering:
            S_test = S + [x]
            if is_valid(S_test):
                S = S_test
        if len(S) > best_size:
            best = S
            best_size = len(S)
    
    # Local search: try to add any missing element
    if not _deadline_hit():
        S = list(best)
        improved = True
        while improved and not _deadline_hit():
            improved = False
            for x in elements:
                if x in S:
                    continue
                S_test = S + [x]
                if is_valid(S_test):
                    S = S_test
                    best = S
                    best_size = len(S)
                    improved = True
                    break  # restart loop with new S
    
    # Try randomized local search with swaps
    if not _deadline_hit():
        random.seed(42)
        S = list(best)
        for _ in range(200):
            if _deadline_hit():
                break
            # Try to add a random missing element
            missing = [x for x in elements if x not in S]
            if missing:
                x = random.choice(missing)
                S_test = S + [x]
                if is_valid(S_test):
                    S = S_test
                    if len(S) > best_size:
                        best = S
                        best_size = len(S)
            # Try swap: remove one, add another
            if len(S) > 1:
                a = random.choice(S)
                missing = [x for x in elements if x not in S]
                if missing:
                    b = random.choice(missing)
                    S_test = [x for x in S if x != a] + [b]
                    if is_valid(S_test):
                        S = S_test
                        if len(S) > best_size:
                            best = S
                            best_size = len(S)
    
    # Final: try to add any remaining elements
    if not _deadline_hit():
        S = list(best)
        improved = True
        while improved and not _deadline_hit():
            improved = False
            for x in elements:
                if x in S:
                    continue
                S_test = S + [x]
                if is_valid(S_test):
                    S = S_test
                    best = S
                    best_size = len(S)
                    improved = True
                    break
    
    return best


if __name__ == "__main__":
    print(construct())
