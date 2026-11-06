"""Rewritten program for the reciprocal-subsum-free set task.

Strategy:
  * The seed returns [13, 14, ..., 24] (12 elements).
  * We try to build larger sets by:
      1. Greedy-add in several orders (descending, ascending, random shuffles)
         with exact rational validation using meet-in-the-middle subset-sum.
      2. Greedy-removal from the full set [1..24].
      3. Local search (swap / add / remove) to improve cardinality.
  * All validation uses Fraction arithmetic to avoid floating-point errors.
  * A wall-clock deadline guard ensures we return the best valid set found
    before the 90 s kill.
"""
import time
import random
from fractions import Fraction
from itertools import combinations

DEADLINE = 75.0
_START = None


def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE


def _subset_sum_possible(vals, target):
    """Check if any non-empty subset of vals sums to target (Fraction).
    Uses meet-in-the-middle for speed when len(vals) is moderate."""
    n = len(vals)
    if n == 0:
        return False
    if n <= 20:
        # Brute force via iterative set of achievable sums
        sums = {Fraction(0)}
        for v in vals:
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns == target:
                    return True
                new_sums.add(ns)
            sums |= new_sums
        return False
    else:
        # Meet in the middle
        mid = n // 2
        left = vals[:mid]
        right = vals[mid:]

        left_sums = {Fraction(0)}
        for v in left:
            new_sums = set()
            for s in left_sums:
                new_sums.add(s + v)
            left_sums |= new_sums

        right_sums = {Fraction(0)}
        for v in right:
            new_sums = set()
            for s in right_sums:
                new_sums.add(s + v)
            right_sums |= new_sums

        for ls in left_sums:
            needed = target - ls
            if needed in right_sums:
                # Must ensure the subset is non-empty overall
                # If ls == 0 and needed == 0, that's the empty subset
                if ls == Fraction(0) and needed == Fraction(0):
                    continue
                return True
        return False


def is_valid(S):
    """Check that no reciprocal of an element in S equals the sum of
    reciprocals of a non-empty subset of the OTHER elements in S."""
    S = list(S)
    n = len(S)
    if n == 0:
        return True
    recip = [Fraction(1, x) for x in S]
    for i in range(n):
        others = recip[:i] + recip[i+1:]
        if not others:
            continue
        if _subset_sum_possible(others, recip[i]):
            return False
    return True


def _greedy_add(order, start=None):
    """Greedily add elements in the given order, validating at each step."""
    if _deadline_hit():
        return start if start else []
    S = list(start) if start else []
    recip_S = [Fraction(1, x) for x in S]
    for x in order:
        if x in S:
            continue
        # Check if adding x violates the condition
        # For each existing element y in S: 1/y != sum of subset of (others ∪ {x})
        # For x itself: 1/x != sum of subset of S
        # We need to check both directions.
        # Simpler: just validate the new set.
        new_S = S + [x]
        if is_valid(new_S):
            S = new_S
            recip_S.append(Fraction(1, x))
    return S


def _greedy_remove():
    """Start with [1..24], remove elements that cause violations."""
    S = list(range(1, 25))
    changed = True
    while changed and not _deadline_hit():
        changed = False
        for i in range(len(S)):
            # Check if S is valid
            if is_valid(S):
                return S
            # Try removing element i
            new_S = S[:i] + S[i+1:]
            if is_valid(new_S):
                S = new_S
                changed = True
                break
    return S


def _local_search(S, iters=500):
    """Try to improve S by adding, removing, or swapping elements."""
    S = list(S)
    best = S
    best_len = len(S)
    for _ in range(iters):
        if _deadline_hit():
            break
        improved = False
        # Try adding
        for x in range(1, 25):
            if x in S:
                continue
            new_S = S + [x]
            if is_valid(new_S):
                S = new_S
                best = S
                best_len = len(S)
                improved = True
                break
        if improved:
            continue
        # Try swapping
        for i in range(len(S)):
            for x in range(1, 25):
                if x in S:
                    continue
                new_S = S[:i] + [x] + S[i+1:]
                if len(new_S) != len(S):
                    continue
                if is_valid(new_S):
                    # Prefer swaps that maintain size; only accept if we can
                    # later grow. For now, accept if valid.
                    S = new_S
                    best = S
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try removing one element to allow future additions? 
            # Actually, removal reduces size, so only do if it enables growth.
            # Skip for now.
            break
    return best


def construct():
    global _START
    _START = time.time()
    random.seed(42)

    best = list(range(13, 25))  # seed answer, 12 elements
    if not is_valid(best):
        best = []

    # 1. Greedy add in descending order
    S1 = _greedy_add(range(24, 0, -1))
    if len(S1) > len(best):
        best = S1

    # 2. Greedy add in ascending order
    S2 = _greedy_add(range(1, 25))
    if len(S2) > len(best):
        best = S2

    # 3. Greedy add from top half
    S3 = _greedy_add(range(13, 25))
    if len(S3) > len(best):
        best = S3

    # 4. Greedy add from bottom half
    S4 = _greedy_add(range(1, 13))
    if len(S4) > len(best):
        best = S4

    # 5. Several random shuffles
    for _ in range(20):
        if _deadline_hit():
            break
        order = list(range(1, 25))
        random.shuffle(order)
        Sr = _greedy_add(order)
        if len(Sr) > len(best):
            best = Sr

    # 6. Greedy removal from full set
    if not _deadline_hit():
        Sg = _greedy_remove()
        if len(Sg) > len(best):
            best = Sg

    # 7. Local search on best
    if not _deadline_hit():
        best = _local_search(best, iters=200)

    # 8. More random restarts if time permits
    while not _deadline_hit():
        order = list(range(1, 25))
        random.shuffle(order)
        Sr = _greedy_add(order)
        Sr = _local_search(Sr, iters=100)
        if len(Sr) > len(best):
            best = Sr

    if not best:
        # Fallback: return a known small valid set
        best = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        if not is_valid(best):
            best = [24]

    return best


# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is kept for interface compatibility.)
# EVOLVE-BLOCK-END
