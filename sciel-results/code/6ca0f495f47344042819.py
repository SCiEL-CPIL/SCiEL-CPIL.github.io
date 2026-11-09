"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import sys

# Increase recursion limit to be safe
sys.setrecursionlimit(10000)

def construct():
    """
    Find the largest subset of [1, 24] such that no reciprocal of a selected element
    equals the sum of reciprocals of a subset of the OTHER selected elements.
    
    This is a constraint satisfaction problem where we seek the maximum cardinality subset.
    We use a backtracking search (DFS) with forward checking.
    
    Key Insight:
    Let S be the selected set. The condition is: for all x in S, 1/x is NOT in Sum(S \ {x}).
    If we build S incrementally by adding elements in a specific order, we can prune:
    1. When adding x to S_old to form S_new:
       - Check if 1/x is in Sum(S_old). If yes, invalid.
       - Check if adding x breaks the condition for any y in S_old.
         The condition for y is: 1/y not in Sum(S_new \ {y}).
         S_new \ {y} = (S_old \ {y}) U {x}.
         Sum(S_new \ {y}) = Sum(S_old \ {y}) U { 1/x + s | s in Sum(S_old \ {y}) } U {1/x}.
         We know 1/y is not in Sum(S_old \ {y}) by induction.
         So we just need to check if 1/y is in { 1/x + s | s in Sum(S_old \ {y}) } U {1/x}.
         1/y == 1/x implies y=x (impossible).
         So we check if 1/y - 1/x is in Sum(S_old \ {y}).
    
    To make this efficient, we maintain:
    - `sums`: Set of all subset sums of S_old.
    - `sums_without`: A dictionary mapping each y in S_old to Sum(S_old \ {y}).
    
    However, maintaining `sums_without` for all y is expensive (O(N * 2^N) memory/time per step).
    Given N=24, the set of sums is bounded. The maximum sum is sum(1/i for i in 1..24) ~ 3.8.
    The number of distinct fractional sums is manageable.
    
    Alternative:
    Since N is small (24), we can use a different strategy.
    We process elements in descending order (24 down to 1).
    Why descending?
    If we add large numbers first, their reciprocals are small.
    The sums of reciprocals of large numbers are small.
    It is less likely that a small reciprocal (like 1/24) is a sum of other reciprocals from the set {24, 23, ...}.
    Actually, 1/24 is smaller than any sum of two other reciprocals from {24..13}.
    This suggests that the "conflict" check might be simpler or less frequent.
    
    Let's stick to the rigorous check but optimize the implementation.
    We will use a recursive function `dfs(x, current_set, sums, sums_without)`.
    x: current element to consider (24 down to 1).
    
    To optimize, we can precompute the reciprocals.
    
    Let's try a simpler pruning:
    If we add x, and 1/x is in `sums`, we prune.
    For the "breaking existing" check, we iterate over y in current_set.
    We need `sums_without[y]`.
    We can compute `sums_without[y]` on the fly if we store it, or recompute it.
    Recomputing `Sum(S_old \ {y})` from scratch for each y is O(2^|S|).
    If |S| is up to 15, 2^15 = 32768.
    15 * 32768 ~ 500k operations per step.
    24 steps * 500k = 12M operations. This is acceptable in Python if the inner loop is fast.
    But `sums` is a set of Fractions. Operations on Fractions are slow.
    
    Let's try to maintain `sums_without` incrementally.
    When we add x to S_old to get S_new:
    New `sums` = sums_old U {s + rx | s in sums_old} U {rx}.
    New `sums_without`:
    For the new element x: sums_without[x] = sums_old.
    For existing y:
    sums_without_new[y] = Sum(S_new \ {y}) = Sum((S_old \ {y}) U {x})
    = sums_without_old[y] U { s + rx | s in sums_without_old[y] } U {rx}.
    
    This update is O(|sums_without_old[y]|) for each y.
    Total cost per step: Sum_{y in S_old} |sums_without_old[y]| + |sums_old|.
    This is roughly O(N * 2^N) in the worst case, but in practice, the sums sets might be smaller due to collisions or bounds.
    
    Let's implement this.
    """
    N = 24
    recips = {i: Fraction(1, i) for i in range(1, N + 1)}
    
    best_size = 0
    best_set = []
    
    def dfs(x, current_set, sums, sums_without):
        nonlocal best_size, best_set
        
        # Pruning: if even taking all remaining elements can't beat best_size
        # Remaining elements are 1 to x. Count is x.
        if len(current_set) + x <= best_size:
            return
            
        if x == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Option 1: Skip x
        dfs(x - 1, current_set, sums, sums_without)
        
        # Option 2: Take x
        rx = recips[x]
        
        # Check 1: Is rx in sums?
        if rx in sums:
            return  # Cannot add x
        
        # Check 2: Does adding x break any existing y?
        valid = True
        for y in current_set:
            ry = recips[y]
            # Check if ry - rx is in sums_without[y]
            if ry - rx in sums_without[y]:
                valid = False
                break
        
        if not valid:
            return
            
        # Update state
        new_sums = set(sums)
        for s in sums:
            new_sums.add(s + rx)
        new_sums.add(rx)
        
        new_sums_without = {y: sums_without[y] for y in sums_without}
        # For new x
        new_sums_without[x] = sums
        # For existing y, update sums_without[y]
        for y in current_set:
            old_sums_wo_y = sums_without[y]
            new_sums_wo_y = set(old_sums_wo_y)
            for s in old_sums_wo_y:
                new_sums_wo_y.add(s + rx)
            new_sums_wo_y.add(rx)
            new_sums_without[y] = new_sums_wo_y
            
        current_set.append(x)
        dfs(x - 1, current_set, new_sums, new_sums_without)
        current_set.pop()
        
    # Start with x=24, empty set
    dfs(24, [], set(), {})
    
    if not best_set:
        return [1]
        
    return best_set

# EVOLVE-BLOCK-END
