"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from fractions import Fraction

def construct():
    n = 256
    # We need to find a subset S of [1, n] with no 3-term arithmetic progression
    # that maximizes sum_{x in S} 1/x.
    
    # A 3-term AP is (a, b, c) with a < b < c and b - a == c - b, i.e., a + c = 2b.
    # So for any pair (a, c) with a < c and a + c even, the midpoint b = (a+c)/2 
    # must not be in S if both a and c are in S.
    # Equivalently, if we pick a and c, we cannot pick b.
    # This is a constraint satisfaction problem.
    
    # Let's think of it as: we have variables x_1, ..., x_n (binary).
    # Constraint: for all a < b < c with a + c = 2b, NOT (x_a AND x_c AND x_b).
    # This is a 3-SAT-like constraint but specifically: x_a + x_b + x_c <= 2 for each AP.
    
    # Since n=256 is small, we can try a greedy approach with backtracking or
    # use the fact that the maximum density set without 3-APs is known to be related
    # to Behrend's construction, but for maximizing reciprocal sum, small numbers are more valuable.
    
    # Key insight: 1/1 > 1/2 > 1/3 > ... so we want to include small numbers if possible.
    # The greedy approach in the seed program picks numbers in order 1, 2, 3, ... and
    # includes a number if it doesn't complete a 3-AP with any two previously selected numbers.
    # This is a valid greedy but may not be optimal.
    
    # Let's try a better approach: since the reciprocal sum is dominated by small numbers,
    # let's try to include as many small numbers as possible.
    
    # Actually, let's think about which numbers can be in a 3-AP-free set.
    # 1 can be in the set. If 1 is in the set, then for any b, c with 1 + c = 2b, 
    # we cannot have both b and c in the set.
    
    # Let's use a branch and bound or a smarter greedy.
    
    # Alternative: use the known result that the largest subset of [1, n] without 3-APs
    # has size roughly n / exp(sqrt(c log n)) but we want max reciprocal sum, not max size.
    
    # For max reciprocal sum, we should prioritize small numbers. Let's try a greedy
    # that processes numbers in order of 1/x (i.e., ascending x) but with lookahead.
    
    # Actually, the seed greedy is already processing in ascending order. Let's see if
    # we can improve it.
    
    # Let's try a different greedy: process numbers in ascending order, but when deciding
    # whether to include x, check if including x would prevent including a larger number
    # with higher reciprocal weight... but larger numbers have lower reciprocal weight.
    
    # Since 1/x is decreasing, the greedy that takes the smallest available number that
    # doesn't create a conflict should be near-optimal. But let's verify with a small example.
    
    # For n=3: [1,2,3]. 1,2,3 form a 3-AP (1,2,3). So we can't take all three.
    # Options: {1,2} sum=1.5, {1,3} sum=1.333, {2,3} sum=0.833. Best is {1,2}.
    # Greedy: take 1, take 2 (1,2 no AP yet), skip 3 (1,2,3 is AP). Result {1,2}. Correct.
    
    # For n=4: [1,2,3,4]. 
    # Greedy: take 1, take 2, skip 3 (1,2,3), take 4? Check: 1,2,4 no AP. 1,4? no pair. 2,4? no.
    # So {1,2,4}. Sum = 1 + 0.5 + 0.25 = 1.75.
    # Alternative: {1,3,4}? 1,3,5 no. 1,4? 1,3,4: 1+4=5, 2*3=6, no. 3,4? no. 
    # Wait, is {1,3,4} valid? Check all triples: (1,3,4): 1+4=5, 2*3=6, no. (1,3), (1,4), (3,4) no 3-APs.
    # Sum = 1 + 1/3 + 1/4 = 1.583 < 1.75. So {1,2,4} is better.
    
    # The greedy seems good. But let's try to improve it by considering that sometimes
    # skipping a number might allow two larger numbers... but larger numbers have smaller reciprocals.
    
    # Actually, I think the greedy is optimal for this problem structure because of the
    # monotonicity of 1/x. But let's implement a slightly smarter version.
    
    # Let's use a backtracking search with pruning for n=256, but that might be too slow.
    
    # Instead, let's try the greedy and then try to improve it by local search.
    
    selected = []
    for x in range(1, n + 1):
        # Check if adding x creates a 3-AP with any two elements in selected
        conflict = False
        for i in range(len(selected)):
            for j in range(len(selected)):
                if i == j:
                    continue
                a, b = selected[i], selected[j]
                # Check if (a, b, x) or (a, x, b) or (x, a, b) or (x, b, a) etc. form a 3-AP
                # A 3-AP is three distinct numbers in arithmetic progression.
                # If we add x, we need to check all pairs (p, q) in selected and see if
                # {p, q, x} contains a 3-AP.
                # The three numbers are p, q, x. They form a 3-AP if one is the average of the other two.
                # So check: 2*p == q + x, 2*q == p + x, 2*x == p + q.
                if 2 * a == b + x or 2 * b == a + x or 2 * x == a + b:
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            selected.append(x)
    
    # Now try to improve with local search: try swapping elements
    # But this is complex. Let's just return the greedy result.
    
    return selected
# EVOLVE-BLOCK-END
