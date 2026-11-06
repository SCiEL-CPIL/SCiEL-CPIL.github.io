import time
import random
import numpy as np

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 82.0  # Safety margin for 90s kill

    # Precompute all 3-APs: (a, b, c) where a < b < c, b-a = c-b = d > 0
    # a in [1, n], a+d <= n, a+2d <= n
    aps_3 = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            aps_3.append((a, b, c))
    
    num_aps3 = len(aps_3)
    
    # Precompute for each element x, the list of 3-AP indices it participates in
    # And for each 3-AP, which elements it contains
    elem_to_aps = {x: [] for x in range(1, n + 1)}
    for idx, (a, b, c) in enumerate(aps_3):
        elem_to_aps[a].append(idx)
        elem_to_aps[b].append(idx)
        elem_to_aps[c].append(idx)
    
    # Precompute 4-AP conflicts: A 3-AP (a, b, c) with diff d conflicts with adding:
    # - a-d (forms a-d, a, b, c)
    # - c+d (forms a, b, c, c+d)
    # Also, two 3-APs can conflict if their union contains a 4-AP.
    # But actually, the constraint is on the SET S: S must not contain any 4-term AP.
    # A 4-term AP is (x, x+d, x+2d, x+3d).
    # So if we have 3-APs, we need to ensure no 4-AP is fully contained in the union of selected elements.
    
    # Let's define the problem differently.
    # We want to select a subset S of [1, n].
    # Constraint: No 4-term AP in S.
    # Objective: Maximize number of 3-term APs in S.
    
    # This is a Max-CSP or similar.
    # Let's use a bitmask for S. n=256, so 256-bit integer.
    # Python ints are arbitrary precision, so this is fine.
    
    # Precompute 4-APs as masks? No, too many.
    # Better: For each candidate element x, determine which existing elements y in S would create a 4-AP with x.
    
    # Let's precompute for each x, the "forbidden pairs" or just check on the fly.
    # Given n=256, checking 4-APs is O(n) per insertion if we're smart, or O(n^2) if naive.
    # Let's precompute for each x, a list of (y, z) such that {x, y, z, w} is a 4-AP?
    # No, a 4-AP is defined by (start, d).
    # If we add x, we violate 4-AP constraint if there exists d such that:
    # 1. x-3d, x-2d, x-d in S
    # 2. x-2d, x-d, x+d in S
    # 3. x-d, x+d, x+2d in S
    # 4. x+d, x+2d, x+3d in S
    
    # To make this fast, let's precompute for each x and each d, the mask of the other 3 elements.
    # But we need to check if those 3 are in S.
    
    # Let's use a greedy + local search approach with efficient scoring.
    
    # Scoring: count 3-APs in S.
    # We can maintain a count incrementally.
    # When adding x, new 3-APs are those containing x and two other elements in S.
    # Number of such 3-APs = sum over y in S of (1 if 2x-y in S else 0) ... wait.
    # A 3-AP containing x is (x-d, x, x+d) or (x, x+d, x+2d) or (x-2d, x-d, x).
    # Let's just precompute for each x, the list of 3-AP indices.
    # When adding x, gain = number of 3-APs in elem_to_aps[x] where the other two elements are already in S.
    
    # Let's represent S as a set for fast membership, and also track the count.
    
    def can_add(x, S):
        """Check if x can be added to S without creating a 4-AP."""
        # Check all 4-AP patterns involving x
        # Pattern 1: x is 4th term: x-3d, x-2d, x-d
        # Pattern 2: x is 3rd term: x-2d, x-d, x+d
        # Pattern 3: x is 2nd term: x-d, x+d, x+2d
        # Pattern 4: x is 1st term: x+d, x+2d, x+3d
        
        # Max d such that elements are in [1, n]
        # For pattern 1: x-3d >= 1 => d <= (x-1)//3
        # For pattern 4: x+3d <= n => d <= (n-x)//3
        # We need to check all d where the other 3 elements are in range.
        
        max_d = (n - 1) // 3 + 1
        for d in range(1, max_d + 1):
            # Pattern 1
            if x - 3*d >= 1:
                if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                    return False
            # Pattern 2
            if x + d <= n:
                if (x - 2*d >= 1) and (x - d in S) and (x + d in S) and (x - 2*d in S):
                    # Wait, pattern 2 is x-2d, x-d, x, x+d.
                    # We are adding x. So we need x-2d, x-d, x+d in S.
                    if (x - 2*d in S) and (x - d in S) and (x + d in S):
                        return False
            # Pattern 3
            if x + 2*d <= n:
                # Pattern 3 is x-d, x, x+d, x+2d.
                # We need x-d, x+d, x+2d in S.
                if (x - d >= 1) and (x - d in S) and (x + d in S) and (x + 2*d in S):
                    return False
            # Pattern 4
            if x + 3*d <= n:
                # Pattern 4 is x, x+d, x+2d, x+3d.
                # We need x+d, x+2d, x+3d in S.
                if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                    return False
        return True

    def count_gain(x, S):
        """Count number of new 3-APs formed by adding x to S."""
        gain = 0
        for idx in elem_to_aps[x]:
            a, b, c = aps_3[idx]
            # Check if the other two are in S
            if x == a:
                if b in S and c in S:
                    gain += 1
            elif x == b:
                if a in S and c in S:
                    gain += 1
            elif x == c:
                if a in S and b in S:
                    gain += 1
        return gain

    def greedy_construction(order):
        S = set()
        current_score = 0
        for x in order:
            if can_add(x, S):
                gain = count_gain(x, S)
                S.add(x)
                current_score += gain
        return S, current_score

    # Local Search
    def local_search(S, current_score):
        improved = True
        S = set(S)
        while improved and time.time() < deadline - 5:
            improved = False
            # Phase 1: Add
            for x in range(1, n + 1):
                if x in S:
                    continue
                if can_add(x, S):
                    gain = count_gain(x, S)
                    if gain > 0:
                        S.add(x)
                        current_score += gain
                        improved = True
                    elif gain == 0:
                        # Maybe add anyway? No, 0 gain doesn't help score, might block others.
                        # But sometimes 0 gain now allows high gain later?
                        # Greedy add with 0 gain is risky. Let's skip for now.
                        pass
                # Check time
                if time.time() > deadline - 5:
                    return S, current_score
            
            # Phase 2: Swap
            # Try removing y and adding x
            # To make this efficient, we can iterate over x not in S, and for each, find a y in S
            # such that swapping improves score.
            # This is expensive. Let's do a limited version.
            
            # For each x not in S:
            #   If can_add(x, S - {y}) for some y in S?
            #   Actually, removing y might make x addable.
            
            # Let's try a simpler swap:
            # For each y in S:
            #   For each x not in S:
            #     If can_add(x, S - {y}) and count_gain(x, S - {y}) - count_loss(y, S) > 0:
            #        Swap.
            
            # count_loss(y, S) = number of 3-APs containing y and two other elements in S.
            # This is similar to count_gain but for removal.
            
            def count_loss(y, S):
                loss = 0
                for idx in elem_to_aps[y]:
                    a, b, c = aps_3[idx]
                    if y == a:
                        if b in S and c in S:
                            loss += 1
                    elif y == b:
                        if a in S and c in S:
                            loss += 1
                    elif y == c:
                        if a in S and b in S:
                            loss += 1
                return loss

            # Optimization: Precompute loss for all y in S?
            # And for each x not in S, precompute gain if added to S?
            
            # Let's just do a few random swaps or specific candidates to save time.
            # Or iterate x, and for each x, try to find a y to remove.
            
            for x in range(1, n + 1):
                if x in S:
                    continue
                # Check if x can be added to S (without removal)
                # If not, maybe removing some y allows it.
                # Which y blocks x?
                # A y blocks x if adding x to S creates a 4-AP involving y.
                # Let's find all y in S such that removing y makes x addable.
                
                # This is complex. Let's just try random swaps if time permits.
                # Or skip swaps for now and rely on multiple restarts.
                pass
                
        return S, current_score

    # Run multiple random restarts
    best_S = set()
    best_score = 0
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S, score = greedy_construction(order)
        S, score = local_search(S, score)
        if score > best_score:
            best_score = score
            best_S = S

    return sorted(best_S)

# EVOLVE-BLOCK-END
