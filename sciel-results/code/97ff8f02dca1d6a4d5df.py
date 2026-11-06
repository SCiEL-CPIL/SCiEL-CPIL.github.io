import time
from fractions import Fraction

def construct():
    N = 256
    deadline = time.time() + 80  # 80 seconds safety margin (90s limit)
    
    # Precompute all 3-AP conflicts
    # conflict[x] = set of y such that x and y cannot both be in the set
    # Actually, we need hyperedges: triples (a, a+d, a+2d) that can't all be present
    # For MWIS on 3-uniform hypergraph, we can use a different approach
    
    # Let's precompute for each pair (i,j) whether they form part of a 3-AP
    # If i and j are in the set, then certain other elements are forbidden
    
    # Better approach: precompute for each element x, the set of elements that conflict with it
    # Two elements conflict if there exists a third element z such that {x, y, z} forms a 3-AP
    # Actually, the constraint is: no three elements form a 3-AP
    # So we need to track which pairs, when both selected, forbid certain third elements
    
    # Let's use a different formulation:
    # For each potential 3-AP (a, b, c) where a < b < c and b-a = c-b,
    # we cannot have all three in the set
    
    # Precompute all 3-APs
    ap_list = []
    for a in range(1, N + 1):
        for d in range(1, N - a + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                ap_list.append((a, b, c))
    
    # For each element, precompute which APs it participates in
    element_aps = {i: [] for i in range(1, N + 1)}
    for ap in ap_list:
        for elem in ap:
            element_aps[elem].append(ap)
    
    # Greedy initial solution
    S = set()
    in_set = [False] * (N + 1)
    
    def can_add(x, current_set, in_set_arr):
        """Check if x can be added without creating a 3-AP"""
        if in_set_arr[x]:
            return False
        for y in current_set:
            # Check if x, y, and some z form a 3-AP
            # Case 1: y, x, z where z = 2x - y
            if y < x:
                z = 2 * x - y
                if z <= N and in_set_arr[z]:
                    return False
            # Case 2: x, y, z where z = 2y - x (y is middle)
            if y > x:
                z = 2 * y - x
                if z <= N and in_set_arr[z]:
                    return False
            # Case 3: y, m, x where m = (x+y)/2
            if (x + y) % 2 == 0:
                m = (x + y) // 2
                if 1 <= m <= N and in_set_arr[m]:
                    return False
        return True
    
    # Greedy: add elements in order of decreasing weight (1/x), i.e., increasing x
    for x in range(1, N + 1):
        if can_add(x, S, in_set):
            S.add(x)
            in_set[x] = True
    
    best_set = S.copy()
    best_weight = sum(Fraction(1, x) for x in S)
    
    # Local search with branch and bound
    # We'll try to improve by considering swaps and additions
    
    # Precompute weights as Fractions for exact comparison
    weights = {x: Fraction(1, x) for x in range(1, N + 1)}
    
    # Try all possible single additions
    for x in range(1, N + 1):
        if not in_set[x] and can_add(x, S, in_set):
            S.add(x)
            in_set[x] = True
            new_weight = best_weight + weights[x]
            if new_weight > best_weight:
                best_weight = new_weight
                best_set = S.copy()
    
    # Try swaps: remove x, add y
    for x in list(S):
        if time.time() > deadline:
            break
        # Remove x
        S.remove(x)
        in_set[x] = False
        old_weight = best_weight
        best_weight -= weights[x]
        
        # Try adding each y
        for y in range(1, N + 1):
            if y == x or in_set[y]:
                continue
            if can_add(y, S, in_set):
                S.add(y)
                in_set[y] = True
                new_weight = best_weight + weights[y]
                if new_weight > best_weight:
                    best_weight = new_weight
                    best_set = S.copy()
                S.remove(y)
                in_set[y] = False
        
        # Restore x
        S.add(x)
        in_set[x] = True
        best_weight += weights[x]
    
    # Convert back to list
    return list(best_set)
