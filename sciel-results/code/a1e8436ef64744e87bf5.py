"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The task: find a nonempty subset S of [1,100] with no subset summing to 200, maximizing |S|.
    #
    # Strategy:
    # 1. Construct a large base set that is known to be valid.
    #    Let L = {66, 67, ..., 100}. The minimum sum of 3 distinct elements in L is 66+67+68 = 201.
    #    The maximum sum of 2 distinct elements in L is 100+99 = 199.
    #    Thus, no subset of L sums to 200. |L| = 35.
    #
    # 2. Add small elements from [1, 65] that do not create a subset sum of 200.
    #    If we add a set A of small elements, we must ensure:
    #    - No subset of A sums to 200. (Easy, since sum(A) < 200 if we pick small enough)
    #    - No subset of A sums to 200 - l for any l in L. i.e., sum(A) avoids {100, 101, ..., 134}.
    #    - No subset of A sums to 200 - l1 - l2 for any l1, l2 in L.
    #      The minimum sum of two elements in L is 66+67=133. So 200-133 = 67.
    #      The maximum sum of two elements in L is 100+99=199. So 200-199 = 1.
    #      So we need sum(A) to avoid [1, 67].
    #      Wait, if sum(A) is in [1, 67], then there exists l1, l2 in L such that sum(A) + l1 + l2 = 200?
    #      Not necessarily. We need the specific subset of A that sums to 200-l1-l2.
    #      If A = {1, 2, ..., k}, the reachable sums are 0..k(k+1)/2.
    #      If k=13, sum=91. Reachable sums 0..91.
    #      Forbidden sums for A (to avoid 200 with 1 or 2 elements from L):
    #      - 1 element from L: 200 - l for l in [66,100] -> [100, 134].
    #      - 2 elements from L: 200 - (l1+l2) for l1,l2 in [66,100].
    #        l1+l2 ranges from 133 to 199.
    #        So 200-(l1+l2) ranges from 1 to 67.
    #      So if A has any subset summing to 1..67, it might be dangerous.
    #      Specifically, if A contains 1, then 1 + 66 + 133? No, 133 is not in L.
    #      If A contains 1, and L contains 66, do we have 1 + 66 + 133 = 200? No, 133 not in L.
    #      Do we have 1 + l1 + l2 = 200? l1+l2 = 199. 99+100=199. Yes!
    #      So if A contains 1, and L contains 99 and 100, then 1+99+100=200.
    #      So A cannot contain 1 if L contains 99 and 100.
    #      Similarly, if A contains 2, 2+99+99? No, distinct. 2+98+100=200.
    #      So A must avoid small numbers if L has large numbers.
    #
    #    Let's refine the base set L.
    #    If we remove 99 and 100 from L, we lose 2 elements.
    #    L' = {66, ..., 98}. Size 33.
    #    Max 2-sum in L' is 97+98=195. Min 2-sum is 66+67=133.
    #    Forbidden sums for A:
    #    - 1 elem from L': [200-98, 200-66] = [102, 134].
    #    - 2 elem from L': [200-195, 200-133] = [5, 67].
    #    So A must have no subset sum in [5, 134] (union of [5,67] and [102,134]).
    #    This is very restrictive.
    #
    #    Let's try a different L.
    #    The current greedy finds L={66..100} and stops. Size 35.
    #    Can we add ANY small element to {66..100}?
    #    If we add 1: 1+99+100=200. No.
    #    If we add 2: 2+98+100=200. No.
    #    ...
    #    If we add 34: 34+66+100=200. No.
    #    It seems {66..100} is "saturated" with respect to small additions because for any x in [1,34],
    #    200-x is in [166, 199], which is a sum of two elements in {66..100}.
    #    E.g., 200-1=199=99+100. 200-2=198=98+100. ... 200-34=166=66+100.
    #    So we cannot add any x in [1,34] to {66..100}.
    #
    #    What if we modify L to allow some small elements?
    #    Suppose we want to add A={1,2,...,k}.
    #    We need to remove elements from L to break all sums 200.
    #    This is a hitting set problem.
    #
    #    Alternative approach:
    #    Use the structure of the problem.
    #    Let S be the solution.
    #    If max(S) >= 100, we can't have 100 and 99 and 1.
    #
    #    Let's look at the set {1..13} U {66..100}.
    #    Size 13 + 35 = 48.
    #    Is it valid?
    #    Check sums involving A={1..13} and L={66..100}.
    #    Sums from A alone: max 91. No 200.
    #    Sums from L alone: min 3-sum 201. No 200.
    #    Sums from A and L:
    #    a + l = 200? a <= 13, l >= 66. Max a+l = 113. No.
    #    a + l1 + l2 = 200?
    #    l1+l2 = 200-a. a in [1,13] -> 200-a in [187, 199].
    #    Does L={66..100} contain pairs summing to 187..199?
    #    Yes. 187=87+100, 188=88+100, ..., 199=99+100.
    #    So if A contains 1, and L contains 99,100, then 1+99+100=200.
    #    So {1..13} U {66..100} is INVALID.
    #
    #    We must remove elements from L to avoid these sums.
    #    For each a in A, we must ensure no pair in L sums to 200-a.
    #    If A={1..13}, we must break pairs summing to 187..199.
    #    Pairs summing to 199: (99,100).
    #    Pairs summing to 198: (98,100), (99,99-no).
    #    Pairs summing to 197: (97,100), (98,99).
    #    ...
    #    Pairs summing to 187: (87,100), (88,99), ..., (93,94).
    #
    #    We need to hit all these pairs.
    #    If we remove 100, we break (99,100), (98,100), ..., (87,100).
    #    Remaining pairs to break:
    #    198: (98,99) -> sum 197? No, 198-100=98. If 100 removed, 198 needs (98,99)? 98+99=197.
    #    Let's list pairs for each target T in 187..199 within L={66..100}:
    #    T=199: (99,100)
    #    T=198: (98,100) [99,99 invalid]
    #    T=197: (97,100), (98,99)
    #    T=196: (96,100), (97,99) [98,98 invalid]
    #    T=195: (95,100), (96,99), (97,98)
    #    ...
    #    T=187: (87,100), (88,99), ..., (93,94)
    #
    #    If we remove 100, we break all pairs involving 100.
    #    Remaining pairs:
    #    T=197: (98,99)
    #    T=196: (97,99)
    #    T=195: (96,99), (97,98)
    #    T=194: (95,99), (96,98)
    #    T=193: (94,99), (95,98), (96,97)
    #    ...
    #    T=187: (88,99), (89,98), ..., (93,94)
    #
    #    If we remove 99, we break (98,99), (97,99), etc.
    #    Remaining pairs (without 100, 99):
    #    T=195: (97,98)
    #    T=194: (96,98)
    #    T=193: (95,98), (96,97)
    #    T=192: (94,98), (95,97)
    #    T=191: (93,98), (94,97), (95,96)
    #    ...
    #    T=187: (89,98), (90,97), (91,96), (92,95), (93,94)
    #
    #    If we remove 98, we break (97,98), (96,98), etc.
    #    Remaining pairs (without 100, 99, 98):
    #    T=193: (96,97)
    #    T=192: (95,97)
    #    T=191: (94,97), (95,96)
    #    T=190: (93,97), (94,96)
    #    T=189: (92,97), (93,96), (94,95)
    #    ...
    #    T=187: (90,97), (91,96), (92,95), (93,94)
    #
    #    It seems removing the top k elements breaks many pairs.
    #    Let's try removing 100, 99, 98, 97, 96, 95.
    #    L_new = {66..94}. Size 29.
    #    Max sum of 2 in L_new: 93+94=187.
    #    Min sum of 2 in L_new: 66+67=133.
    #    Forbidden sums for A: [200-187, 200-133] = [13, 67].
    #    So A must have no subset sum in [13, 67].
    #    If A={1..12}, sum=78.
    #    Reachable sums of {1..12} are 0..78.
    #    This intersects [13, 67].
    #    So A={1..12} is invalid with L_new={66..94}.
    #
    #    We need A to have NO subset sum in [13, 67].
    #    This means all non-empty subset sums of A must be < 13 or > 67.
    #    If A has an element >= 13, say 13, then 13 is in [13,67]. Invalid.
    #    So max(A) <= 12.
    #    If A={1..12}, sum 13 is reachable (13 itself if 13 in A, or 1+2+...+5=15? No, 1+2+3+4+5=15. 1+2+3+4=10. 1+2+3+4+5=15. Can we make 13? 12+1=13. Yes).
    #    So A cannot contain 12 and 1.
    #    Actually, if A={1..12}, 13 is a subset sum. So invalid.
    #    We need A such that no subset sums to 13..67.
    #    This implies sum(A) < 13 or sum(A) > 67 and gaps.
    #    If sum(A) < 13, max A is {1..4} (sum 10) or {1,2,3,4,5} (sum 15 - no).
    #    {1,2,3,4} sum 10. Valid?
    #    Subset sums: 1,2,3,4,3,4,5,5,6,6,7,7,8,9,9,10.
    #    All < 13.
    #    So A={1,2,3,4} is valid with L_new={66..94}.
    #    Total size: 4 + 29 = 33. Worse than 35.
    #
    #    Let's try a different L.
    #    What if we keep L={66..100} and don't add any small numbers? Size 35.
    #    Can we replace some large numbers with more small numbers?
    #    If we remove 100, 99, 98 from L (size 32).
    #    L={66..97}. Max 2-sum 96+97=193. Min 2-sum 133.
    #    Forbidden sums for A: [7, 67].
    #    A must avoid subset sums 7..67.
    #    Max A with sum < 7: {1,2,3} sum 6. Size 3.
    #    Total 32+3=35. Same.
    #
    #    What if we remove 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67?
    #    L={66}. Size 1.
    #    Forbidden sums for A: [134, 134].
    #    A can be anything as long as no subset sums to 134.
    #    Max A from [1,65] with no subset sum 134.
    #    Sum 1..16 = 136.
    #    We can take {1..15} sum 120. Size 15.
    #    Can we add 16? 120+16=136. Subset sum 134?
    #    1..15 sums to 120. 134-120=14. Can we make 14 from 1..15? Yes.
    #    So {1..16} has subset sum 134?
    #    1..15 sum 120. 134 is not in 1..15 sums? Max sum 120.
    #    So {1..16} sums to 136.
    #    Does {1..16} have a subset summing to 134?
    #    Total 136. Remove 2. 136-2=134. Yes.
    #    So we cannot have 16 and 2 in A if sum is 134?
    #    We need to avoid 134.
    #    {1..15} is safe. Size 15.
    #    Total 1+15=16. Worse.
    #
    #    It seems 35 is a strong local optimum for this "block" strategy.
    #    However, the greedy algorithm in the seed code returns 35.
    #    Is there a set larger than 35?
    #
    #    Let's try to construct one.
    #    Consider S = {1..50}.
    #    Sum 1..50 = 1275.
    #    Does it have subset sum 200?
    #    Yes, 100+100? No distinct.
    #    199+1? 199 not in set.
    #    99+101? 101 not in set.
    #    50+49+48+47+6 = 200?
    #    50+49+48+47 = 194. 194+6=200.
    #    Yes.
    #
    #    Consider S = {51..100}. Size 50.
    #    Min sum 51+52=103.
    #    51+52+53+44? No.
    #    51+52+53+54 = 210.
    #    Can we make 200?
    #    100+99+1? No 1.
    #    100+51+49? No 49.
    #    100+50? No 50.
    #    99+51+50? No.
    #    98+52+50? No.
    #    97+53+50? No.
    #    It seems {51..100} might be valid?
    #    Let's check.
    #    Elements are 51..100.
    #    Any subset sum 200?
    #    If 1 element: no (51..100).
    #    If 2 elements: x+y=200. x,y in [51,100].
    #    x=100, y=100 (no).
    #    x=99, y=101 (no).
    #    So no 2-element sum.
    #    If 3 elements: x+y+z=200.
    #    Min 51+52+53=156.
    #    Max 98+99+100=297.
    #    Is there a solution?
    #    100+51+49 (49 no).
    #    99+52+49 (49 no).
    #    ...
    #    67+66+67 (no distinct).
    #    Let's check if 200 is reachable.
    #    This is equivalent to: is there a subset of {51..100} summing to 200?
    #    Let y_i = x_i - 51. y_i in [0, 49].
    #    Sum (y_i + 51) = 200.
    #    Sum y_i + 51*k = 200.
    #    If k=1: y=149. No.
    #    If k=2: y1+y2 = 200-102=98.
    #    Max y1+y2 = 49+48=97. No.
    #    If k=3: y1+y2+y3 = 200-153=47.
    #    Min 0+1+2=3. Max 47+48+49=144.
    #    Can we make 47 from 3 distinct numbers in [0,49]?
    #    Yes, e.g., 44+2+1=47.
    #    So x1=44+51=95, x2=2+51=53, x3=1+51=52.
    #    95+53+52=200.
    #    So {51..100} is INVALID.
    #
    #    We need to remove elements to break this.
    #    Remove 95, 53, 52?
    #    If we remove 95, we break this sum.
    #    But there are many sums.
    #
    #    Let's go back to the greedy result of 35.
    #    Is it possible to get 36?
    #    The seed code does local search but it's slow and might not find improvements.
    #    The prompt asks to "design a new program... from scratch... do not reuse current designs".
    #    But the current design IS the greedy + local search.
    #    The directive says "keep this program's approach and improve it within that approach".
    #    Wait, the second directive says "design a new program... do not reuse the current designs".
    #    And "Use a bitset... and greedily select elements in descending order".
    #    So I must use the greedy descending approach.
    #
    #    The greedy descending approach yields {66..100} (size 35).
    #    Can we tweak the greedy to do better?
    #    Maybe skip some large numbers to allow more small numbers?
    #    But greedy is descending. It picks 100, 99, ... 66.
    #    Then it tries 65. 65+66+69=200. Reject.
    #    ...
    #    It rejects all 1..65.
    #    So result is {66..100}.
    #
    #    To improve, we might need to modify the greedy to be "lookahead" or "swap-based".
    #    But the directive says "greedily select elements in descending order only if the target sum 200 is not newly reachable".
    #    This implies a strict greedy.
    #
    #    However, the "improvement" might come from a better initial state or a different tie-breaking?
    #    No, the set is unique for strict greedy.
    #
    #    Is there a different descending order?
    #    "Descending order" usually means 100, 99, ..., 1.
    #
    #    Maybe the "approach" allows for a smarter greedy?
    #    "Greedily select elements in descending order...".
    #
    #    Let's check if {66..100} is truly optimal.
    #    If I can find a set of size 36, I win.
    #
    #    Let's try to construct one manually.
    #    Take {67..100} (size 34).
    #    Min 3-sum 67+68+69=204.
    #    Max 2-sum 100+99=199.
    #    Can we add small numbers?
    #    Forbidden sums for A:
    #    1 elem from L: [100, 133].
    #    2 elem from L: [1, 67].
    #    So A must avoid subset sums 1..133.
    #    This means A must be empty or sum(A) > 133 and no small sums.
    #    If A={134}, sum 134.
    #    Is 134 in [1,133]? No.
    #    Is 134 in [100,133]? No.
    #    So A={134} is valid?
    #    Wait, 134 is not in [1,100].
    #    We can only pick from [1,100].
    #    So max A sum is 100.
    #    So A must have no subset sum in [1, 100] (since [1,67] U [100,133] covers [1,100] except 68..99).
    #    Wait, [1,67] and [100,133].
    #    Gaps: 68..99.
    #    So A can have subset sums in 68..99.
    #    But A must NOT have subset sums in 1..67 or 100..133.
    #    If A has any element x <= 67, then x is a subset sum in [1,67]. Invalid.
    #    So min(A) >= 68.
    #    If A={68}, sum 68. Valid?
    #    68 is not in [1,67] or [100,133].
    #    So A={68} is valid.
    #    Total S = {67..100} U {68}. But 68 is already in L.
    #    So we can't add 68.
    #    Next candidate 69? Already in L.
    #    ...
    #    100? Already in L.
    #    So no elements from [1,100] \ {67..100} are valid?
    #    [1,100] \ {67..100} = {1..66}.
    #    All x in {1..66} are <= 67. So they are forbidden.
    #    So {67..100} is maximal? Size 34.
    #    {66..100} is size 35.
    #
    #    What if we remove 66 from L?
    #    L={67..100}.
    #    Forbidden sums: 1..67, 100..133.
    #    Same as above.
    #
    #    What if we remove 100 from L?
    #    L={66..99}.
    #    Forbidden sums:
    #    1 elem: [101, 134].
    #    2 elem: [2, 68]. (200-198=2, 200-133=67? 66+67=133. 200-133=67. 66+99=165. 200-165=35. 98+99=197. 200-197=3. Min 2-sum 133. Max 197. So [3, 67]).
    #    Union: [3, 67] U [101, 134].
    #    Gaps: 1, 2, 68..100, 135..
    #    A can have subset sums 1, 2, 68..100.
    #    If A={1}, sum 1. Valid.
    #    If A={2}, sum 2. Valid.
    #    If A={1,2}, sum 3. 3 is in [3,67]. Invalid.
    #    So A can be {1} or {2} or {68} or {69}...
    #    If A={1}, S={1} U {66..99}. Size 1+34=35.
    #    If A={68}, S={68} U {66..99}. 68 is in L.
    #    So A={1} gives size 35.
    #
    #    It seems 35 is the max for this structure.
    #
    #    I will return the greedy result, which is {66..100}.
    #    To satisfy "improve", I will implement the greedy efficiently and return it.
    #    If there is a better set, it's not found by this approach.
    #
    #    Wait, the previous response forecasted 48. That was wrong.
    #    I must produce a valid program.
    #    The greedy program is valid.
    #
    #    I will rewrite the greedy loop cleanly.

    selected = []
    reachable = 1  # Bit 0 set
    mask = (1 << 201) - 1  # Bits 0 to 200
    
    for x in range(100, 0, -1):
        # Check if adding x makes 200 reachable
        if (reachable >> (200 - x)) & 1:
            # 200-x is reachable, so adding x makes 200 reachable
            pass
        else:
            # Safe to add
            selected.append(x)
            reachable |= (reachable << x)
            reachable &= mask
            
    return selected
# EVOLVE-BLOCK-END
