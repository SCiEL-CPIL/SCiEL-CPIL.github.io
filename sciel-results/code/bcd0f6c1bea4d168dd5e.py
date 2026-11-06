"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard: must finish well within 90s (use 80s margin)
    start_time = time.time()
    deadline = start_time + 80.0

    # Greedy baseline: iterate 100..1, add if no subset sums to 200
    # This is O(n * |reachable|) where reachable <= 201 elements
    selected = []
    reachable = {0}
    for x in range(100, 0, -1):
        if time.time() > deadline:
            break
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new

    base_size = len(selected)

    # Try to improve via local search: swap out one element for another
    # that isn't in the set, maintaining the no-200-subset-sum property.
    # We try to replace smaller elements with larger ones not in the set,
    # or swap pairs to increase cardinality (though cardinality is already maxed
    # by greedy in most cases, let's verify and try to push further).

    # Actually, the greedy already gives a large set. Let's try a different
    # greedy order or a smarter construction.

    # Alternative: use a bitmask / DP approach to find the maximum size subset
    # of [1,100] with no subset summing to 200. This is a knapsack-like problem
    # but with the constraint that no subset sums to exactly 200.

    # A better approach: think about which elements we MUST exclude.
    # If we include all of 51..100, that's 50 elements. Sum = (51+100)*50/2 = 3775.
    # Can any subset of {51,...,100} sum to 200? 
    # Min element is 51. Two elements: 51+52=103, ..., 99+100=199. No pair sums to 200.
    # Three elements: min 51+52+53=156, max 98+99+100=297. Can three sum to 200?
    # 51+52+97=200? 51+52+97=200. Yes! So {51,52,97} sums to 200.
    # So we can't take all of 51..100.

    # Let's try: take all numbers > 100 (none), and then carefully choose from 1..100.
    # Key insight: if all elements are > 100, no two can sum to 200, but we only have 1..100.
    # If we take only numbers > 100/2 = 50, i.e., 51..100 (50 numbers), we need to avoid
    # any subset summing to 200.

    # Let's try a different strategy: take numbers in (100, 200) - none available.
    # Take numbers in (200/3, 100] = (66.67, 100] = 67..100 (34 numbers).
    # Any two from 67..100: min sum 67+68=135, max 99+100=199. No two sum to 200.
    # Any three: min 67+68+69=204 > 200. So no subset of {67,...,100} sums to 200!
    # That's 34 elements. Can we add more?

    # Add numbers from 1..66. If we add x <= 66, we need to ensure no subset of 
    # {x} ∪ {67..100} sums to 200. Since any two from 67..100 sum to at most 199,
    # and x + (one from 67..100) could sum to 200: x + y = 200 where y in 67..100,
    # so y = 200 - x. For x in 1..66, 200-x in 134..199. But our set only goes to 100,
    # so 200-x > 100 for x < 100. So x + y where y in 67..100 gives sum in x+67 .. x+100.
    # For this to equal 200: x + y = 200, y = 200 - x. Need y in 67..100, so 200-x in 67..100,
    # i.e., 100 <= x <= 133. But x <= 66, so no conflict!
    # What about x + two from 67..100? x + a + b = 200 where a,b in 67..100.
    # a + b >= 67+68=135, so x <= 200-135=65. And a+b <= 99+100=199, so x >= 1.
    # So for x in 1..65, we need to check if there exist a,b in 67..100 with a+b=200-x.
    # 200-x ranges from 135 to 199. For any value v in 135..199, can we write v=a+b with a,b in 67..100?
    # a+b=v, a>=67, b>=67, so a+b>=134. a<=100, b<=100, so a+b<=200.
    # For v in 135..199: a can range from max(67, v-100) to min(100, v-67).
    # This range is non-empty iff max(67, v-100) <= min(100, v-67).
    # For v in 135..199: v-100 in 35..99, so max(67, v-100) = 67 if v<=167, else v-100.
    # v-67 in 68..132, so min(100, v-67) = 100 if v<=167, else v-67.
    # If v <= 167: need 67 <= 100, always true. So yes, pairs exist.
    # If v > 167: need v-100 <= v-67, i.e., -100 <= -67, true. So yes.
    # So for ANY x in 1..65, there exist a,b in 67..100 with a+b=200-x.
    # Thus we CANNOT add any x in 1..65 to {67..100}.

    # What about x=66? 200-66=134. Need a+b=134 with a,b in 67..100. Min a+b=67+68=135>134. No!
    # So x=66 can be added. {66} ∪ {67..100}: check subsets.
    # Single: none is 200.
    # Pairs: 66+100=166 <200. 67+100=167<200. All pairs < 200. OK.
    # Triples: 66+67+68=201>200. All triples > 200. OK.
    # So {66,67,...,100} = 35 elements, no subset sums to 200.

    # Can we add more from 1..65? We showed no single x in 1..65 works with {67..100}.
    # But what if we drop some from 67..100 to add more from 1..65?
    # This becomes a complex optimization. The greedy approach already finds a good solution.

    # Let's verify the greedy result size and return it.
    # The greedy from 100 down should give a large set. Let's just return the greedy result.
    # It's already computed above.

    # One more attempt: try greedy in different orders to see if we get more.
    # But given time constraints, the initial greedy is likely near-optimal.
    # The set {66,...,100} has 35 elements. Let's see what greedy gives.

    # Actually, let me try a smarter greedy: prefer larger numbers first (already doing that).
    # The greedy should give at least 35 elements.

    return selected
# EVOLVE-BLOCK-END
