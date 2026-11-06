"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 256
    # Weights are 1/x
    weights = [Fraction(0)] + [Fraction(1, i) for i in range(1, n + 1)]
    
    # Precompute conflicts: for each x, list of (a, b) such that a, b, x form a 3-AP
    # Specifically, if we have a < b < x, and 2b - a = x, then {a, b, x} is a 3-AP.
    # Also, if we have a < x < b, and 2x - a = b, then {a, x, b} is a 3-AP.
    # And if we have x < a < b, and 2a - x = b, then {x, a, b} is a 3-AP.
    
    # To make B&B efficient, we process vertices in descending order of weight (i.e., ascending x).
    # Order: 1, 2, 3, ..., 256
    
    # Conflict lists: 
    # conflicts[x] will contain a list of tuples (a, b) or similar?
    # Actually, standard MWIS on hypergraph:
    # A set S is independent if no hyperedge is fully contained in S.
    # Hyperedges are all (a, b, c) with a < b < c and b-a = c-b.
    
    # Efficient check:
    # When adding x to a set S, we must ensure no 3-AP is formed.
    # Any 3-AP containing x is of the form:
    # 1. (x-2d, x-d, x) for d >= 1 such that x-2d >= 1
    # 2. (x-d, x, x+d) for d >= 1 such that x+d <= n
    # 3. (x, x+d, x+2d) for d >= 1 such that x+2d <= n
    
    # So, x conflicts with S if:
    # - There exist a, b in S such that {a, b, x} is a 3-AP.
    # This happens if:
    #   a = x-2d, b = x-d (both in S)
    #   a = x-d, b = x+d (both in S, note b > x so if we process in increasing order, b not in S yet? 
    #   Wait, if we process in increasing order 1..n, then when considering x, S only contains elements < x.
    #   So case 2 and 3 involve elements > x, which are not in S yet.
    #   Therefore, we only need to check case 1: (x-2d, x-d, x).
    
    # Let's verify this logic.
    # If we build the set S by adding elements in order 1, 2, ..., n.
    # When we decide to include x, S contains a subset of {1, ..., x-1}.
    # A 3-AP involving x and two elements from S must be of the form (a, b, x) with a < b < x.
    # The condition is 2b - a = x.
    # So for each d >= 1, let b = x-d, a = x-2d.
    # If a >= 1 and a in S and b in S, then we cannot add x.
    
    # This simplifies the conflict check significantly!
    # We don't need to precompute all hyperedges. We just need to check pairs in S.
    
    # However, for B&B, we need to bound the potential weight.
    # The upper bound is current_weight + sum of weights of all remaining vertices that COULD be added.
    # A simple bound is current_weight + sum(weights[i] for i in range(x+1, n+1)).
    # This is a very loose bound.
    
    # A tighter bound: 
    # We can use the fact that the set must be 3-AP-free.
    # But calculating the exact MWIS of the remaining subgraph is hard.
    # Let's stick to the simple sum bound first, but use Fraction for exactness.
    # The simple sum bound is: sum_{i=x+1}^n 1/i.
    
    # Let's optimize the search:
    # 1. Order vertices by weight (which is just 1, 2, ..., n).
    # 2. Branch on include/exclude.
    # 3. Use Fraction for weights and bounds.
    
    # Precompute suffix sums of weights for quick upper bound calculation.
    # suffix_sum[i] = sum_{j=i}^n 1/j
    suffix_sum = [Fraction(0)] * (n + 2)
    for i in range(n, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
    
    best_set = []
    best_weight = Fraction(0)
    
    # Recursive B&B
    # State: current index x (1-based, next to consider), current set S (as a set of ints for fast lookup), current weight W
    # To speed up conflict checking, we maintain S as a Python set.
    
    def branch(x, S, W):
        nonlocal best_set, best_weight
        
        # Check time
        if time.time() > deadline:
            return
        
        # Pruning:
        # Upper bound = W + suffix_sum[x]
        # If this is <= best_weight, prune.
        if W + suffix_sum[x] <= best_weight:
            return
        
        # Base case: x > n
        if x > n:
            if W > best_weight:
                best_weight = W
                best_set = list(S)
            return
        
        # Option 1: Exclude x
        # This is usually the safer branch to explore first? Or include?
        # Including x might lead to a better solution faster if weights are high.
        # But excluding x allows more future inclusions.
        # Let's try Include first, as we want to find a good solution early to prune.
        
        # Check if x can be added to S
        can_add = True
        # Check for 3-APs (a, b, x) with a, b in S
        # a = x - 2d, b = x - d
        # d can range from 1 to (x-1)//2
        for d in range(1, (x // 2) + 1):
            b = x - d
            a = x - 2 * d
            if a < 1:
                break
            if a in S and b in S:
                can_add = False
                break
        
        if can_add:
            S.add(x)
            branch(x + 1, S, W + weights[x])
            S.remove(x)
            
            # Check time again
            if time.time() > deadline:
                return
        
        # Option 2: Exclude x
        branch(x + 1, S, W)
        
    # Start search
    # We can use a greedy initialization to set a good initial best_weight
    greedy_set = set()
    greedy_w = Fraction(0)
    for i in range(1, n + 1):
        can_add = True
        for d in range(1, (i // 2) + 1):
            b = i - d
            a = i - 2 * d
            if a < 1:
                break
            if a in greedy_set and b in greedy_set:
                can_add = False
                break
        if can_add:
            greedy_set.add(i)
            greedy_w += weights[i]
    
    best_set = list(greedy_set)
    best_weight = greedy_w
    
    # Run B&B
    # We start with x=1, S=empty, W=0
    # Note: The greedy solution is a valid starting point.
    # The B&B will explore the space to see if we can do better.
    
    # Optimization: 
    # The simple bound W + suffix_sum[x] is very weak.
    # For example, if x=100, suffix_sum[100] is roughly ln(256/100) ~ 0.9.
    # But we can't add all of them.
    # However, for small n=256, the search space might be manageable with good pruning.
    
    # Let's add a stronger bound if possible.
    # A known bound for 3-AP-free sets is related to the size, but we are maximizing weight.
    # Since weights are decreasing, the "best" set will likely consist of small numbers.
    # But 1, 2, 3 is a 3-AP. 1, 2, 4, 5... 
    # Actually, the set of numbers with only 0s and 1s in base 3 is 3-AP-free.
    # This set is sparse.
    
    # Let's just run the B&B. If it's too slow, we might need to adjust.
    # Given 90s, and n=256, pure B&B might be too slow without better bounds.
    # However, the "Include" branch is constrained by the 3-AP condition.
    # The number of 3-AP-free subsets is much smaller than 2^256.
    
    # Let's try to improve the bound.
    # If we are at x, the remaining elements are x+1...n.
    # The maximum weight we can add is bounded by the weight of the largest 3-AP-free subset of {x+1...n}.
    # This is still hard.
    
    # Alternative: Use the greedy result as a strong lower bound.
    # And rely on the structure.
    
    # Let's run it.
    branch(1, set(), Fraction(0))
    
    return best_set

# EVOLVE-BLOCK-END
