import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s kill
    n = 256
    
    # Precompute conflict graph: conflict[a][c] = True if a,c can't both be in S
    # a,c conflict if (a+c) is even and 1 <= (a+c)//2 <= n and (a+c)//2 != a and (a+c)//2 != c
    # Since a != c (we consider distinct pairs), midpoint is always different from both.
    # (a+c)//2 == a iff c == a, so for distinct a,c it's always valid midpoint.
    # So conflict: a+c is even and 1 <= (a+c)//2 <= n
    # Since a,c in [1,n], (a+c)//2 is always in [1,n] when a+c is even.
    # So conflict iff (a+c) % 2 == 0, i.e., a and c have the same parity.
    # Wait, that's not right. Let me re-check.
    # A 3AP is (a, b, c) with a + c = 2b. For a given pair {a,c}, they form a 3AP
    # if there exists b = (a+c)/2 that is an integer in [1,n].
    # b is integer iff (a+c) is even.
    # b is in [1,n] iff 1 <= (a+c)/2 <= n iff 2 <= a+c <= 2n.
    # Since a,c >= 1, a+c >= 2 always. Since a,c <= n, a+c <= 2n always.
    # So for distinct a,c in [1,n], they conflict iff (a+c) is even, i.e., same parity.
    
    # Wait, that means ALL pairs of same parity conflict! That can't be right for a 3AP-free set.
    # Let me re-think. A 3AP (a, b, c) requires a, b, c to be DISTINCT elements of the set.
    # If a and c have the same parity, then b = (a+c)/2 is an integer. But we also need b != a and b != c.
    # b = a iff (a+c)/2 = a iff c = a. Since we consider distinct pairs, b != a and b != c always.
    # So yes, any two elements of the same parity form a 3AP with their midpoint.
    # This means a 3AP-free set can contain at most one element of each parity class?
    # No wait - it means that if a and c are both in S and have the same parity, then b=(a+c)/2 must NOT be in S.
    # But the constraint is: S contains no 3-term AP. So {a, b, c} ⊄ S for any 3AP.
    # Having a and c in S is fine as long as b is NOT in S.
    # The conflict is: we can't have all three {a, b, c} in S.
    # So the "conflict graph" for independent set formulation needs to be different.
    
    # Actually, for the independent set formulation:
    # Node x conflicts with node y if {x, y} cannot both be in S.
    # But {x, y} CAN both be in S - the constraint is that we can't have x, (x+y)/2, y all in S.
    # This is a 3-uniform hypergraph constraint, not a pairwise one!
    
    # So the simple MWIS formulation doesn't directly apply. Let me reconsider.
    
    # Alternative: use the branch-and-bound but with much faster conflict checking.
    # The key bottleneck is the `can_add` function which iterates over all elements in S.
    
    # Better approach: precompute for each x, the set of pairs that would create a 3AP with x.
    # If x is added to S, it conflicts with existing pairs {a,c} in S where {a,x,c} or {a,c,x} etc. form a 3AP.
    # Specifically, adding x to S creates a 3AP iff:
    #   - There exist a, c in S such that {a, x, c} is a 3AP (x is middle: a+c=2x)
    #   - There exist a, c in S such that {a, c, x} is a 3AP (x is end: a+x=2c or c+x=2a)
    
    # Let me use a different strategy: greedy + local search with swaps.
    
    # First, build a good initial solution greedily.
    # For each x from 1 to n, add x if it doesn't create a 3AP.
    
    def has_3ap_with(S, x):
        """Check if adding x to S creates any 3AP."""
        for a in S:
            # x as middle: need c = 2x - a in S, c != x, c != a
            c = 2 * x - a
            if 1 <= c <= n and c in S and c != x and c != a:
                return True
            # x as first term: need c = 2a - x in S, c != x, c != a  
            c = 2 * a - x
            if 1 <= c <= n and c in S and c != x and c != a:
                return True
            # x as last term: need c = (a+x)//2 in S if (a+x) even, c != x, c != a
            if (a + x) % 2 == 0:
                c = (a + x) // 2
                if 1 <= c <= n and c in S and c != x and c != a:
                    return True
        return False
    
    # Greedy: try adding elements in order of decreasing weight (smallest first)
    S = set()
    for x in range(1, n + 1):
        if not has_3ap_with(S, x):
            S.add(x)
    
    best_set = set(S)
    best_weight_float = sum(1.0 / x for x in S)
    
    # Local search: try adding elements, removing elements, swapping
    # Phase 1: Try to add any element that fits
    improved = True
    while improved:
        improved = False
        for x in range(1, n + 1):
            if x not in S and not has_3ap_with(S, x):
                S.add(x)
                best_set = set(S)
                best_weight_float = sum(1.0 / x for x in S)
                improved = True
    
    # Phase 2: Try swaps - remove one element, add one or more
    # For each element in S, try removing it and then greedily adding
    for x_remove in list(S):
        if time.time() > deadline - 5.0:
            break
        S_temp = set(S)
        S_temp.discard(x_remove)
        # Try to add elements that were previously blocked
        for x_add in range(1, n + 1):
            if x_add not in S_temp and not has_3ap_with(S_temp, x_add):
                S_temp.add(x_add)
        w_temp = sum(1.0 / x for x in S_temp)
        if w_temp > best_weight_float:
            S = S_temp
            best_set = set(S)
            best_weight_float = w_temp
    
    # Phase 3: Try adding pairs - remove one, add two
    for x_remove in list(S):
        if time.time() > deadline - 5.0:
            break
        S_temp = set(S)
        S_temp.discard(x_remove)
        # Try to add two elements
        added = []
        for x_add in range(1, n + 1):
            if x_add not in S_temp and not has_3ap_with(S_temp, x_add):
                S_temp.add(x_add)
                added.append(x_add)
        w_temp = sum(1.0 / x for x in S_temp)
        if w_temp > best_weight_float:
            S = S_temp
            best_set = set(S)
            best_weight_float = w_temp
    
    # Compute exact weight
    exact_weight = sum(Fraction(1, x) for x in best_set)
    
    return sorted(best_set)
