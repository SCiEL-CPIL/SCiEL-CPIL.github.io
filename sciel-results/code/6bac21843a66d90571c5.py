import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85  # Safety margin for 90s limit
    
    n = 192
    max_k = 48
    
    def can_add(S, x):
        """
        Check if x can be added to set S.
        S is a list or set of integers.
        Returns True if valid.
        """
        S_list = list(S)
        k = len(S_list)
        
        # 1. Check if x divides any triple sum in S
        # If k < 3, no triples
        if k >= 3:
            # Optimization: if min sum > max multiple?
            # Just iterate
            for b, c, d in combinations(S_list, 3):
                if (b + c + d) % x == 0:
                    return False
        
        # 2. Check if any a in S divides (x + b + c) for distinct b, c in S \ {a}
        # We need a | (x + b + c) => b + c ≡ -x (mod a)
        for a in S_list:
            if k - 1 < 2:
                continue
            
            target = (-x) % a
            
            # Get others
            others = [y for y in S_list if y != a]
            
            # Check pairs in others
            # Optimization: sort others?
            # Just double loop
            m = len(others)
            for i in range(m):
                oi = others[i]
                for j in range(i + 1, m):
                    oj = others[j]
                    if (oi + oj) % a == target:
                        return False
        
        return True

    def greedy_build(order):
        selected = []
        for x in order:
            if len(selected) >= max_k:
                break
            if can_add(selected, x):
                selected.append(x)
        return selected

    # Strategy 1: Descending order
    best = []
    order1 = list(range(n, 0, -1))
    best = greedy_build(order1)
    
    if time.time() > deadline:
        return best

    # Strategy 2: Random restarts if best < max_k
    # Or just run a few more orders to be sure
    # Try ascending? Unlikely to be better.
    # Try random orders biased to large numbers
    
    random.seed(42)
    for attempt in range(5):
        if time.time() > deadline - 5:
            break
        
        # Create order: mostly large, some random
        # Just shuffle 1..192 but keep large ones early?
        # Or just random shuffle
        order = list(range(1, n + 1))
        random.shuffle(order)
        # Bias: sort by (random + value)
        # Actually, let's try: 50% descending, 50% random
        if attempt % 2 == 0:
             order = list(range(n, 0, -1))
        else:
             random.shuffle(order)
        
        sol = greedy_build(order)
        if len(sol) > len(best):
            best = sol
            
    # Local Search to try to improve or fix if stuck
    # If we have < 48, try to add more
    # If we have 48, we are done (max possible)
    
    if len(best) < max_k:
        # Try to add any missing element
        used = set(best)
        for x in range(n, 0, -1):
            if x not in used:
                if can_add(best, x):
                    best.append(x)
                    if len(best) >= max_k:
                        break
        
        # If still < 48, try swaps?
        # This might be overkill if greedy worked well.
        # But let's add a simple swap loop if time permits.
        
        improved = True
        while improved and len(best) < max_k and time.time() < deadline - 10:
            improved = False
            used = set(best)
            # Try to add
            for x in range(n, 0, -1):
                if x not in used:
                    if can_add(best, x):
                        best.append(x)
                        improved = True
                        break
                        
            if len(best) >= max_k:
                break
                
            # Try swaps: remove a, add x
            # Only if we can't add directly
            for i in range(len(best)):
                if time.time() > deadline - 10:
                    break
                a = best[i]
                temp = best[:i] + best[i+1:]
                used_temp = set(temp)
                for x in range(n, 0, -1):
                    if x not in used_temp:
                        if can_add(temp, x):
                            # We swapped a for x. Size is same.
                            # This doesn't increase size, but might allow adding more later?
                            # Actually, we want to increase size.
                            # If we can't add directly, swapping might help if x is "safer"
                            # But we just checked all x for direct addition.
                            # So this is only useful if we are stuck at size K < 48.
                            # Swapping doesn't change size.
                            # We need to find a sequence of swaps that allows an addition.
                            # This is complex.
                            pass
                            
    return best
