"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction

def construct():
    """
    Find the largest subset of [1, 24] such that no reciprocal of a selected element
    equals the sum of reciprocals of a subset of the OTHER selected elements.
    
    Approach:
    Process elements in decreasing order (24 down to 1).
    Key Insight: If S contains elements all greater than x, then for any y in S, 1/y < 1/x.
    Adding x cannot create a sum equal to 1/y because 1/y - 1/x < 0, while sums of reciprocals are positive.
    Thus, we only need to ensure that 1/x is not already a sum of a subset of the previously selected (larger) elements.
    
    This reduces the problem to a standard backtracking search where we maintain the set of all subset sums of reciprocals of the currently selected elements.
    """
    N = 24
    elements = list(range(N, 0, -1))  # 24, 23, ..., 1
    recips = {x: Fraction(1, x) for x in elements}
    
    best_size = 0
    best_set = []
    
    # Memoization or caching might help, but the state is (idx, frozenset(sums)).
    # The sums set can be large, so simple memoization on idx is not enough.
    # Given N=24, simple DFS with pruning should suffice.
    
    def backtrack(idx, current_set, sums):
        nonlocal best_size, best_set
        
        # Pruning: Even if we take all remaining elements, we can't beat best_size
        remaining = N - idx
        if len(current_set) + remaining <= best_size:
            return
        
        if idx == N:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        x = elements[idx]
        rx = recips[x]
        
        # Option 1: Try to include x
        # Check if rx is in sums. If it is, we cannot include x.
        if rx not in sums:
            # Calculate new sums
            # new_sums = sums U {s + rx for s in sums} U {rx}
            # Optimization: We can update in place or create new set.
            # Creating new set is safer for backtracking.
            new_sums = set(sums)
            # Add rx
            new_sums.add(rx)
            # Add rx + s for all s in original sums
            for s in sums:
                new_sums.add(s + rx)
            
            current_set.append(x)
            backtrack(idx + 1, current_set, new_sums)
            current_set.pop()
            
        # Option 2: Skip x
        backtrack(idx + 1, current_set, sums)

    # Initial call
    # sums starts empty
    backtrack(0, [], set())
    
    if not best_set:
        return [1]
        
    return best_set

# EVOLVE-BLOCK-END
