import time
import random

N = 256

def construct():
    # 1. Generate all 3-APs
    # A 3-AP is (a, b, c) with a < b < c and b-a = c-b.
    # Represent as tuple (a, b, c) or frozenset.
    # We need to check conflicts.
    
    all_3aps = []
    # d is difference
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            all_3aps.append((a, b, c))
            
    num_aps = len(all_3aps)
    
    # 2. Build Conflict Graph
    # Edge if AP_i U AP_j contains a 4-AP.
    # Optimization: Two 3-APs form a 4-AP in their union if:
    # Case 1: They share 2 elements and are consecutive segments.
    #   e.g., (a, a+d, a+2d) and (a+d, a+2d, a+3d).
    #   Union: a, a+d, a+2d, a+3d. This is a 4-AP.
    # Case 2: Do they form a 4-AP otherwise?
    #   If they don't share 2 elements, their union has 4 or 5 or 6 elements.
    #   If they share 1 element, union has 5 elements.
    #   If they share 0 elements, union has 6 elements.
    #   A 4-AP in the union must be a subset of the union.
    #   Given the structure of 3-APs, the only 4-APs that can be formed by the union of two 3-APs are those where the 3-APs are "aligned" to form the 4-AP.
    #   Specifically, if P1 and P2 are 3-APs, P1 U P2 contains a 4-AP iff P1 and P2 are the two 3-APs of that 4-AP.
    #   Proof: Let Q be a 4-AP in P1 U P2. Q contains two 3-APs Q1, Q2.
    #   Q1 subset P1 U P2. Q2 subset P1 U P2.
    #   If Q1 is not subset P1 and not subset P2, it must pick elements from both.
    #   But Q1 is a 3-AP.
    #   It turns out that for 3-APs, the only way to form a 4-AP is the consecutive segment case.
    #   Let's verify.
    #   P1 = {1, 3, 5} (d=2). P2 = {2, 4, 6} (d=2). Union {1,2,3,4,5,6}.
    #   Contains {1,2,3,4} (d=1).
    #   Is {1,2,3,4} formed by P1 and P2 being consecutive segments of it?
    #   Segments of {1,2,3,4} are {1,2,3} and {2,3,4}.
    #   P1 is not {1,2,3}. P2 is not {2,3,4}.
    #   So my "Case 1 only" hypothesis is WRONG.
    #   P1={1,3,5} and P2={2,4,6} DO form a 4-AP in their union.
    #   So we must check if P1 U P2 contains ANY 4-AP.
    
    # Efficient check for conflict:
    # P1 U P2 has at most 6 elements.
    # Check all 4-subsets? No, check if any 4-AP is a subset.
    # A 4-AP is defined by start a and diff d.
    # We can just iterate through all 4-APs in [1, N] and see if they are subset of P1 U P2?
    # That's too slow for building the graph (16k * 16k * 4APs).
    
    # Better:
    # For each pair of APs, check if their union contains a 4-AP.
    # Union size <= 6.
    # We can generate all 4-APs that could possibly be in the union.
    # Or simply:
    # Sort the union elements.
    # Check for 4-APs in the sorted list.
    
    # Precompute 4-APs as sets of elements?
    # There are ~ N^2/12 4-APs. ~ 5000.
    # For each AP_i, store the set of 4-APs that contain elements of AP_i?
    # No, we need "Union contains 4-AP".
    
    # Let's use a different strategy for Conflict:
    # Two APs conflict if they cannot both be in a set whose union has no 4-AP.
    # This is hard to precompute pairwise if the "cross" effect is significant.
    
    # Given the time limit and complexity, I will simplify the conflict check:
    # Conflict if P1 U P2 contains a 4-AP.
    # Implementation:
    # For each pair (i, j), compute union.
    # Check if union has a 4-AP.
    # To speed up, note that if P1 and P2 have different differences, they might still form a 4-AP (like the d=2 example).
    
    # Number of APs is ~16000. Pairs ~ 128M. Too many for Python in 60s.
    # We need a faster heuristic.
    
    # Revised Strategy:
    # 1. Greedy Construction of A directly, but using "3-AP density" as a guide?
    # No, directive says MIS on 3-APs.
    
    # I will implement a **Sparse** Conflict Graph.
    # Only check conflicts for APs that are "close" in index or difference?
    # Or, just use the "Consecutive Segment" conflict (Case 1) which is fast to compute, and rely on the Repair step to fix the "Cross" 4-APs.
    # This is a valid "Heuristic MIS".
    
    # Conflict if P1 and P2 are consecutive segments of a 4-AP.
    # P1 = (a, a+d, a+2d), P2 = (a+d, a+2d, a+3d).
    # Or P1 = (a+d, a+2d, a+3d), P2 = (a, a+d, a+2d).
    
    # This graph is sparse.
    # Build this graph.
    # Run Greedy MIS.
    # Union elements.
    # Repair: Remove elements to break ANY 4-AP.
    # Improve: Add elements to increase 3-APs.
    
    # This should work and be fast.
    
    # 1. Generate 3-APs
    ap_to_idx = {}
    all_3aps = []
    idx = 0
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            ap = (a, b, c)
            ap_to_idx[ap] = idx
            all_3aps.append(ap)
            idx += 1
            
    num_aps = len(all_3aps)
    
    # 2. Build Adjacency List for "Consecutive Segment" Conflicts
    # For each 4-AP (a, a+d, a+2d, a+3d), the two 3-APs are (a, a+d, a+2d) and (a+d, a+2d, a+3d).
    # They conflict.
    adj = [[] for _ in range(num_aps)]
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap1 = (a, a + d, a + 2 * d)
            ap2 = (a + d, a + 2 * d, a + 3 * d)
            
            i1 = ap_to_idx.get(ap1)
            i2 = ap_to_idx.get(ap2)
            
            if i1 is not None and i2 is not None:
                adj[i1].append(i2)
                adj[i2].append(i1)
                
    # 3. Greedy Independent Set
    # Sort APs by degree (ascending) or random?
    # Low degree nodes are easier to pick.
    # Or high degree?
    # Usually, picking low degree nodes first leaves more options.
    
    # Let's try random shuffle and pick if no neighbors picked.
    # This is a standard greedy MIS.
    
    order = list(range(num_aps))
    random.shuffle(order)
    
    selected = [False] * num_aps
    for i in order:
        # Check if any neighbor is selected
        conflict = False
        for j in adj[i]:
            if selected[j]:
                conflict = True
                break
        if not conflict:
            selected[i] = True
            
    # 4. Construct A from selected APs
    A_set = set()
    for i in range(num_aps):
        if selected[i]:
            a, b, c = all_3aps[i]
            A_set.add(a)
            A_set.add(b)
            A_set.add(c)
            
    # 5. Repair: Remove elements to break 4-APs
    # Check all 4-APs. If all 4 elements are in A_set, remove one.
    # Which one to remove? The one that destroys the most 3-APs? Or just the last one?
    # Let's remove the element that is part of the most 3-APs in A_set?
    # Or just remove the largest element?
    
    # Precompute 3-APs for each element?
    # For repair, we just need to break 4-APs.
    
    # Iterate through all 4-APs
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            elems = [a, a + d, a + 2 * d, a + 3 * d]
            if all(e in A_set for e in elems):
                # Remove one element.
                # To minimize loss of 3-APs, remove the one with lowest "3-AP contribution".
                # But that's expensive.
                # Just remove the last one?
                A_set.remove(elems[3])
                
    # 6. Improve: Add elements
    # Try adding elements not in A_set.
    # If adding x doesn't create a 4-AP, add it.
    # Check if adding x creates a 4-AP:
    # x is part of a 4-AP if there are 3 other elements in A_set that form a 4-AP with x.
    
    # Precompute for each x, the 4-APs it belongs to?
    # Or just check on the fly.
    
    # For each x in 1..N not in A_set:
    #   Check if adding x creates a 4-AP.
    #   A 4-AP containing x can be:
    #   x, x+d, x+2d, x+3d
    #   x-d, x, x+d, x+2d
    #   x-2d, x-d, x, x+d
    #   x-3d, x-2d, x-d, x
    #   If any of these 4-tuples has the other 3 in A_set, then adding x creates a 4-AP.
    
    def can_add(x, A_set):
        if x in A_set:
            return False
        # Check all 4-APs containing x
        # d > 0
        for d in range(1, N // 3 + 1):
            # x is first
            if x + 3 * d <= N:
                if (x + d in A_set) and (x + 2 * d in A_set) and (x + 3 * d in A_set):
                    return False
            # x is second
            if x - d >= 1 and x + 2 * d <= N:
                if (x - d in A_set) and (x + d in A_set) and (x + 2 * d in A_set):
                    return False
            # x is third
            if x - 2 * d >= 1 and x + d <= N:
                if (x - 2 * d in A_set) and (x - d in A_set) and (x + d in A_set):
                    return False
            # x is fourth
            if x - 3 * d >= 1:
                if (x - 3 * d in A_set) and (x - 2 * d in A_set) and (x - d in A_set):
                    return False
        return True
        
    # Greedy Add
    for x in range(1, N + 1):
        if x not in A_set:
            if can_add(x, A_set):
                A_set.add(x)
                
    return sorted(list(A_set))

BET: {mechanism: "MIS on 3-APs provides a dense initial core of 3-APs that are pairwise compatible (mostly), which is then repaired and expanded. This avoids the local optima of simple element-wise greedy search by starting from a globally structured set of APs.", q: 0.6, delta: [0.1, 0.5]}
