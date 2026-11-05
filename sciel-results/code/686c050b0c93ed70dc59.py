"""Optimized construction of two Sidon sets with disjoint difference sets."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    N = 256
    MAX_SIZE = 64
    
    # Precompute: for each x in [1..256], and for each set S,
    # we need to check if x can be added.
    # We'll use a fast incremental approach.
    
    def _greedy_build(order, deadline):
        """
        Greedily build two sets following the given order.
        For each x in order, try to add to set A or set B (whichever allows it).
        Returns (A, B, sumsA, sumsB, diffsA, diffsB).
        """
        A = []
        B = []
        sumsA = set()  # pair sums of A (including a+a)
        sumsB = set()  # pair sums of B
        diffsA = set()  # nonzero diffs of A
        diffsB = set()  # nonzero diffs of B
        # Union of all diffs for cross-check
        all_diffs = set()
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Try to add x to A
            can_add_A = True
            if can_add_A:
                # Check pair sums: x + a for all a in A, and x+x
                for a in A:
                    if (a + x) in sumsA:
                        can_add_A = False
                        break
                if can_add_A and (x + x) in sumsA:
                    can_add_A = False
                # Check diffs: x - a and a - x for all a in A
                if can_add_A:
                    for a in A:
                        d1 = x - a
                        d2 = a - x
                        if d1 in diffsA or d2 in diffsA:
                            can_add_A = False
                            break
                        if d1 in all_diffs or d2 in all_diffs:
                            can_add_A = False
                            break
            
            # Try to add x to B
            can_add_B = True
            if can_add_B:
                for b in B:
                    if (b + x) in sumsB:
                        can_add_B = False
                        break
                if can_add_B and (x + x) in sumsB:
                    can_add_B = False
                if can_add_B:
                    for b in B:
                        d1 = x - b
                        d2 = b - x
                        if d1 in diffsB or d2 in diffsB:
                            can_add_B = False
                            break
                        if d1 in all_diffs or d2 in all_diffs:
                            can_add_B = False
                            break
            
            # Choose which set to add to (prefer the smaller one for balance, or random)
            if can_add_A and can_add_B:
                # Add to the set with fewer elements to balance
                if len(A) <= len(B):
                    # Add to A
                    for a in A:
                        sumsA.add(a + x)
                        diffsA.add(x - a)
                        diffsA.add(a - x)
                        all_diffs.add(x - a)
                        all_diffs.add(a - x)
                    sumsA.add(x + x)
                    A.append(x)
                else:
                    # Add to B
                    for b in B:
                        sumsB.add(b + x)
                        diffsB.add(x - b)
                        diffsB.add(b - x)
                        all_diffs.add(x - b)
                        all_diffs.add(b - x)
                    sumsB.add(x + x)
                    B.append(x)
            elif can_add_A:
                for a in A:
                    sumsA.add(a + x)
                    diffsA.add(x - a)
                    diffsA.add(a - x)
                    all_diffs.add(x - a)
                    all_diffs.add(a - x)
                sumsA.add(x + x)
                A.append(x)
            elif can_add_B:
                for b in B:
                    sumsB.add(b + x)
                    diffsB.add(x - b)
                    diffsB.add(b - x)
                    all_diffs.add(x - b)
                    all_diffs.add(b - x)
                sumsB.add(x + x)
                B.append(x)
        
        return A, B
    
    # Strategy: multiple random orderings with different biases
    random.seed(12345)
    best_total = 0
    best_A, best_B = [], []
    
    # First: natural order
    A0, B0 = _greedy_build(range(1, N+1), deadline)
    total0 = len(A0) + len(B0)
    if total0 > best_total:
        best_total = total0
        best_A, best_B = A0, B0
    
    # Multiple random shuffles
    num_trials = 0
    while time.time() < deadline - 2:
        order = list(range(1, N+1))
        random.shuffle(order)
        A_t, B_t = _greedy_build(order, deadline)
        total_t = len(A_t) + len(B_t)
        if total_t > best_total:
            best_total = total_t
            best_A, best_B = A_t, B_t
        num_trials += 1
    
    # Local search: try swapping elements or adding more
    # Try to add elements that were skipped
    if time.time() < deadline - 5:
        # Try a second pass with a different strategy:
        # Build one set first (maximally), then build the other
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Build A greedily (Sidon only, ignore B)
            A_greedy = []
            sumsA = set()
            diffsA = set()
            for x in range(1, N+1):
                can_add = True
                for a in A_greedy:
                    if (a + x) in sumsA:
                        can_add = False
                        break
                if can_add and (x + x) in sumsA:
                    can_add = False
                if can_add:
                    for a in A_greedy:
                        d = x - a
                        if d in diffsA:
                            can_add = False
                            break
                if can_add:
                    for a in A_greedy:
                        sumsA.add(a + x)
                        diffsA.add(x - a)
                        diffsA.add(a - x)
                    sumsA.add(x + x)
                    A_greedy.append(x)
            
            # Now build B avoiding A's diffs
            B_greedy = []
            sumsB = set()
            diffsB = set()
            forbidden_diffs = set(diffsA)
            for x in range(1, N+1):
                if x in A_greedy:
                    # Allow overlap but still check constraints
                    pass
                can_add = True
                for b in B_greedy:
                    if (b + x) in sumsB:
                        can_add = False
                        break
                if can_add and (x + x) in sumsB:
                    can_add = False
                if can_add:
                    for b in B_greedy:
                        d1 = x - b
                        d2 = b - x
                        if d1 in diffsB or d2 in diffsB:
                            can_add = False
                            break
                        if d1 in forbidden_diffs or d2 in forbidden_diffs:
                            can_add = False
                            break
                if can_add:
                    for b in B_greedy:
                        sumsB.add(b + x)
                        diffsB.add(x - b)
                        diffsB.add(b - x)
                        forbidden_diffs.add(x - b)
                        forbidden_diffs.add(b - x)
                    sumsB.add(x + x)
                    B_greedy.append(x)
            
            total = len(A_greedy) + len(B_greedy)
            if total > best_total:
                best_total = total
                best_A, best_B = A_greedy, B_greedy
    
    # Ensure nonempty
    if not best_A and not best_B:
        best_A = [1]
        best_B = [2]
    elif not best_A:
        best_A = [1]
    elif not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
