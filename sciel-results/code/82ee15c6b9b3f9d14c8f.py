"""
Construct a maximum-cardinality subset of [1,100] with no subset summing to 200.

Strategy:
- Use a greedy heuristic that selects integers in an order that minimizes the
  number of "forbidden" subset-sum collisions (hyperedges summing to 200).
- We maintain the set of achievable subset sums of the chosen elements.
- For each candidate x in 1..100, we compute the new sums that would be added.
  If 200 is NOT among them, we can include x without violating the constraint.
- To maximize cardinality, we prefer elements that add fewer new achievable sums
  (i.e., have lower "conflict density"), and break ties by preferring larger
  values (which tend to be less involved in many 200-sums).

Additionally, we use a deadline guard to ensure we finish well within the 90s limit.
"""

import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    # We'll try multiple heuristic orderings and keep the best (largest) valid set.
    best_set = set()
    best_size = 0
    
    # Precompute: for each x in 1..100, how many subsets of [1,100]\{x} sum to 200-x?
    # This is a proxy for "conflict density" — elements involved in many 200-sums are riskier.
    # But we can't precompute all subsets of 99 elements. Instead, we use a dynamic approach.
    
    # Strategy 1: Greedy with dynamic sum tracking, trying multiple orderings.
    
    def greedy_order(order):
        """Try to build a set by processing elements in the given order."""
        chosen = set()
        reachable = {0}
        for x in order:
            if time.time() > deadline:
                break
            # Compute new sums if we add x
            new_sums = set()
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    new_sums.add(ns)
            if 200 not in new_sums:
                chosen.add(x)
                reachable |= new_sums
        return chosen
    
    # Generate several candidate orderings
    orderings = []
    
    # Ordering 1: Descending (original approach)
    orderings.append(list(range(100, 0, -1)))
    
    # Ordering 2: Ascending
    orderings.append(list(range(1, 101)))
    
    # Ordering 3: Descending even, then descending odd
    evens_desc = [x for x in range(100, 0, -2) if x % 2 == 0]
    odds_desc = [x for x in range(100, 0, -2) if x % 2 == 1]
    orderings.append(evens_desc + odds_desc)
    
    # Ordering 4: By "risk" — elements x where 200-x is also in [1,100] are riskier
    # Pair risk: x is risky if 200-x exists in [1,100] and x != 200-x
    def risk_score(x):
        # x contributes to a 2-sum with 200-x if 200-x is in [1,100] and != x
        r = 0
        if 1 <= 200 - x <= 100 and 200 - x != x:
            r += 100
        # Also, x can be part of many k-sums; larger x is generally safer
        # Use a heuristic: elements > 100 are impossible, so all are in range.
        # Prefer larger elements (they have fewer partners to sum to 200)
        return r + (100 - x) * 0.1
    
    # Sort by risk (ascending) — safer elements first
    orderings.append(sorted(range(1, 101), key=lambda x: risk_score(x)))
    
    # Ordering 5: Randomized (a few seeds)
    rng = random.Random(42)
    for seed in [42, 123, 777, 2024]:
        r = random.Random(seed)
        orderings.append(r.sample(range(1, 101), 100))
    
    # Ordering 6: Greedy by marginal gain — at each step, pick the element that
    # adds the fewest new reachable sums (while not hitting 200).
    # This is more expensive but potentially better.
    
    def greedy_marginal():
        chosen = set()
        reachable = {0}
        remaining = set(range(1, 101))
        while remaining and time.time() < deadline:
            best_x = None
            best_new_count = float('inf')
            best_new_sums = None
            for x in remaining:
                new_sums = set()
                for s in reachable:
                    ns = s + x
                    if ns <= 200:
                        new_sums.add(ns)
                if 200 in new_sums:
                    continue
                # Count how many new sums are added
                added = new_sums - reachable
                if len(added) < best_new_count:
                    best_new_count = len(added)
                    best_x = x
                    best_new_sums = new_sums
                    if best_new_count == 0:
                        break
            if best_x is None:
                break
            chosen.add(best_x)
            reachable |= best_new_sums
            remaining.discard(best_x)
        return chosen
    
    # Try all orderings
    for i, order in enumerate(orderings):
        if time.time() > deadline - 5.0:
            break
        result = greedy_order(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    # Try the marginal greedy
    if time.time() < deadline - 5.0:
        result = greedy_marginal()
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    # If best_set is empty (shouldn't happen), fall back to a simple valid set
    if not best_set:
        # Just take all numbers > 200... but there are none in [1,100].
        # Take numbers > 100? None. 
        # Any single element is valid.
        best_set = {100}
    
    return sorted(best_set)
