import time
import random

def construct():
    t_start = time.time()
    deadline = t_start + 55.0  # Safety margin
    
    N = 256
    
    def is_valid_swap(S, y, x):
        """
        Check if (S - {y}) U {x} is 3-AP free, given S is 3-AP free.
        y in S, x not in S.
        """
        # We only need to check if x forms a 3-AP with any pair in S - {y}
        # Pairs (a, b) in S - {y}.
        # 3-AP conditions involving x:
        # 1. x, a, b => 2a = x + b => b = 2a - x
        # 2. a, x, b => 2x = a + b => b = 2x - a
        # 3. a, b, x => 2b = a + x => a = 2b - x
        
        # Iterate a in S - {y}
        # For each a, check if partner b is in S - {y}
        
        # Optimization: S is a set.
        # We can just iterate.
        for a in S:
            if a == y:
                continue
            # Case 1: x, a, b -> b = 2a - x
            b = 2 * a - x
            if b in S and b != y:
                return False
            # Case 2: a, x, b -> b = 2x - a
            b = 2 * x - a
            if b in S and b != y:
                return False
            # Case 3: a, b, x -> a = 2b - x => b = (a+x)/2
            # This is covered by iterating b?
            # If we iterate a, and check b = 2a-x and b=2x-a, do we cover all?
            # Let the AP be {u, v, w}.
            # If x=u, then v, w in S. 2v = u+w => w = 2v-x. Covered by Case 1 with a=v.
            # If x=v, then u, w in S. 2v = u+w => w = 2x-u. Covered by Case 2 with a=u.
            # If x=w, then u, v in S. 2v = u+w => u = 2v-x. Covered by Case 1 with a=v?
            # Wait. Case 1 checks if there exists a in S such that 2a-x in S.
            # If x is the largest (w), then u, v in S.
            # The AP is u, v, x.
            # 2v = u + x => u = 2v - x.
            # So if we iterate a=v, we check if b = 2v-x = u is in S.
            # Yes, this is covered by Case 1 where a is the middle element.
            
        return True

    def greedy_construct(skip_set=None):
        if skip_set is None:
            skip_set = set()
        selected = set()
        for x in range(1, N + 1):
            if x in skip_set:
                continue
            # Check if x can be added to selected
            # x forms AP with a, b in selected?
            # Same logic as is_valid_swap but adding x to selected
            valid = True
            for a in selected:
                # x, a, b => b = 2a - x
                b = 2 * a - x
                if b in selected:
                    valid = False
                    break
                # a, x, b => b = 2x - a
                b = 2 * x - a
                if b in selected:
                    valid = False
                    break
                # a, b, x => a = 2b - x => b = (a+x)/2
                # This is covered?
                # If x is largest, a, b in selected.
                # 2b = a + x.
                # If we iterate a, we check b = 2a-x and b = 2x-a.
                # We missed the case where a is the smallest?
                # Let's re-verify the "add x" check.
                # We need to ensure no a, b in selected such that {a,b,x} is AP.
                # Iterate a in selected.
                # 1. x is smallest: x, a, b => 2a = x+b => b = 2a-x.
                # 2. x is middle: a, x, b => 2x = a+b => b = 2x-a.
                # 3. x is largest: a, b, x => 2b = a+x => a = 2b-x.
                #    In this case, if we iterate a (which is the smallest),
                #    we don't directly check b.
                #    However, if we iterate b (which is the middle),
                #    we check a = 2b-x.
                #    So if we iterate ALL elements in selected, we cover all cases
                #    by checking both directions?
                #    Let's just check:
                #    for a in selected:
                #       if (2*a - x) in selected: return False
                #       if (2*x - a) in selected: return False
                #    Is this sufficient?
                #    Suppose AP is u, v, w with x=w.
                #    Then u, v in selected.
                #    2v = u + w => u = 2v - w = 2v - x.
                #    When we iterate a=v, we check if (2v - x) is in selected.
                #    Yes, u is in selected. So it returns False.
                #    So yes, iterating a and checking 2a-x and 2x-a is sufficient.
            
            if valid:
                selected.add(x)
        return selected

    def hill_climb(S):
        S = set(S)
        W = sum(1.0 / x for x in S)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            best_gain = 0
            best_swap = None
            S_list = list(S)
            # Sort by descending value to try removing large ones first?
            # Not strictly necessary, but might find good swaps faster.
            for y in S_list:
                # Only consider x < y to ensure weight increase
                for x in range(1, y):
                    if x in S:
                        continue
                    if is_valid_swap(S, y, x):
                        gain = 1.0 / x - 1.0 / y
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (y, x)
            if best_swap:
                y, x = best_swap
                S.remove(y)
                S.add(x)
                W += best_gain
                improved = True
        return S, W

    best_S = set()
    best_W = 0.0

    # 1. Standard Greedy
    S0 = greedy_construct()
    S_hc, W_hc = hill_climb(S0)
    if W_hc > best_W:
        best_W = W_hc
        best_S = S_hc

    # 2. Perturbed Greedy (skip small numbers)
    # Try skipping 1, 2, 3, etc.
    # The optimal set might exclude 1 to allow more small numbers?
    # Unlikely, 1 is very valuable. But let's try a few.
    for skip in range(1, 10):
        if time.time() > deadline:
            break
        S_skip = greedy_construct(skip_set={skip})
        S_hc, W_hc = hill_climb(S_skip)
        if W_hc > best_W:
            best_W = W_hc
            best_S = S_hc

    # 3. Random Restarts / Kicks
    # If we have time, try to kick the best solution
    if time.time() < deadline - 5:
        for _ in range(100): # Max 100 kicks
            if time.time() > deadline:
                break
            S_cur = set(best_S)
            # Remove 2 random elements
            if len(S_cur) < 2:
                continue
            to_remove = random.sample(list(S_cur), 2)
            for y in to_remove:
                S_cur.remove(y)
            
            # Try to add 2 random valid elements
            # We want to add elements that are small (high weight)
            # But they must be valid.
            # Just try to add random x < min(S_cur) or similar?
            # Better: Try to add any valid x from 1..N
            # We want to improve weight, so prefer small x.
            candidates = [x for x in range(1, N+1) if x not in S_cur]
            random.shuffle(candidates)
            added = 0
            for x in candidates:
                if added >= 2:
                    break
                # Check if x can be added to S_cur
                valid = True
                for a in S_cur:
                    if (2*a - x) in S_cur or (2*x - a) in S_cur:
                        valid = False
                        break
                if valid:
                    S_cur.add(x)
                    added += 1
            
            # Hill climb the kicked solution
            S_hc, W_hc = hill_climb(S_cur)
            if W_hc > best_W:
                best_W = W_hc
                best_S = S_hc

    return list(best_S)
