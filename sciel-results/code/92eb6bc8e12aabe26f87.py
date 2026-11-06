"""Rewritten program: construct() returns up to 48 distinct ints in [1,512]
such that no element is the exact arithmetic mean of any subset of >=2 OTHER elements.
Strategy: greedy construction with fast subset-sum DP validity checks, plus a final
polish pass that tries to swap out elements to increase density. Time-budgeted.
"""
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill

    N = 512

    def is_valid_set(selected):
        """Check if no element in selected is the mean of any subset of >=2 other elements."""
        n = len(selected)
        if n < 2:
            return True
        sel = sorted(selected)
        for i, a in enumerate(sel):
            others = [x for j, x in enumerate(sel) if j != i]
            m = len(others)
            if m < 2:
                continue
            # For each subset size k >= 2, check if any subset of others sums to k*a
            # Use DP: dp[s] = set of subset sizes that can achieve sum s
            # But we need to check per element a, and for each k whether sum k*a is achievable
            # Max sum of others: sum(others) <= 512*47 ~ 24064
            # We'll do: for each a, build DP over others to find achievable sums,
            # then check if k*a is achievable for some k in [2, m].
            # Optimization: precompute achievable sums once per a.
            max_sum = sum(others)
            if max_sum == 0:
                continue
            # dp[s] = True if sum s is achievable by some subset of others
            dp = [False] * (max_sum + 1)
            dp[0] = True
            for x in others:
                for s in range(max_sum, x - 1, -1):
                    if dp[s - x]:
                        dp[s] = True
            # Now check: for k in 2..m, is k*a achievable?
            for k in range(2, m + 1):
                target = k * a
                if target <= max_sum and dp[target]:
                    return False
        return True

    def is_valid_fast(selected, new_elem):
        """Check if adding new_elem to selected keeps the set valid.
        Only need to check:
        1. new_elem is not mean of any subset of >=2 elements in selected
        2. No existing element in selected becomes mean of a subset that includes new_elem
        """
        sel = selected
        m = len(sel)
        # Check 1: new_elem not mean of subset of sel
        if m >= 2:
            max_sum = sum(sel)
            dp = [False] * (max_sum + 1)
            dp[0] = True
            for x in sel:
                for s in range(max_sum, x - 1, -1):
                    if dp[s - x]:
                        dp[s] = True
            for k in range(2, m + 1):
                target = k * new_elem
                if target <= max_sum and dp[target]:
                    return False
        # Check 2: for each a in sel, a not mean of subset of (sel \ {a}) that may include new_elem
        for a in sel:
            others = [x for x in sel if x != a] + [new_elem]
            m2 = len(others)
            if m2 < 2:
                continue
            max_sum2 = sum(others)
            dp2 = [False] * (max_sum2 + 1)
            dp2[0] = True
            for x in others:
                for s in range(max_sum2, x - 1, -1):
                    if dp2[s - x]:
                        dp2[s] = True
            for k in range(2, m2 + 1):
                target = k * a
                if target <= max_sum2 and dp2[target]:
                    return False
        return True

    # Greedy construction: try to add elements in order that maximizes density
    # Start with a good seed: use powers or structured set, then fill greedily
    selected = []
    # Try adding elements from 1 to 512, but use a heuristic order
    # Better: try all candidates and pick the one that allows most future additions
    # For speed, do a simple greedy first, then improve

    # Phase 1: Greedy from 1 to 512
    for x in range(1, N + 1):
        if len(selected) >= 48:
            break
        if time.time() > deadline - 10:
            break
        if is_valid_fast(selected, x):
            selected.append(x)

    if len(selected) >= 48:
        return selected[:48]

    # Phase 2: Try to improve by swapping elements
    # For each position, try replacing with a different value
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        sel_set = set(selected)
        for i in range(len(selected)):
            if time.time() > deadline - 5:
                break
            for x in range(1, N + 1):
                if x in sel_set:
                    continue
                # Try swapping selected[i] with x
                new_sel = selected[:i] + [x] + selected[i+1:]
                new_sel.sort()
                # Quick check: is new_sel valid?
                # Use is_valid_fast by removing selected[i] and adding x
                remaining = [v for j, v in enumerate(selected) if j != i]
                if is_valid_fast(remaining, x):
                    # Also need to check that x doesn't break validity with remaining
                    # is_valid_fast already checks both directions
                    selected = new_sel
                    sel_set = set(selected)
                    improved = True
                    break
            if time.time() > deadline - 5:
                break

    # Phase 3: Try to add more elements if we're under 48
    if len(selected) < 48 and time.time() < deadline - 3:
        sel_set = set(selected)
        for x in range(1, N + 1):
            if len(selected) >= 48:
                break
            if x in sel_set:
                continue
            if is_valid_fast(selected, x):
                selected.append(x)
                sel_set.add(x)

    # Final validation
    if not is_valid_set(selected):
        # Fallback: rebuild greedily
        selected = []
        for x in range(1, N + 1):
            if len(selected) >= 48:
                break
            if is_valid_fast(selected, x):
                selected.append(x)

    return selected[:48]
