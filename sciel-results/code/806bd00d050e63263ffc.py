"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85
    T = 200
    MASK = (1 << (T + 1)) - 1  # Mask to keep bits 0..T
    
    def achievable_sums_bitset(elements):
        """Return a bitmask where bit i is set if sum i is achievable."""
        bits = 1  # sum 0 is achievable
        for x in elements:
            if x > T:
                continue
            bits |= (bits << x)
            bits &= MASK
        return bits
    
    def has_sum_200(elements):
        """Check if any subset sums to exactly 200."""
        bits = 1
        for x in elements:
            if x > T:
                continue
            bits |= (bits << x)
            bits &= MASK
            if (bits >> T) & 1:
                return True
        return False
    
    def greedy(order):
        """Greedy: add x if it doesn't create a sum of 200."""
        selected = []
        bits = 1  # achievable sums bitmask
        for x in order:
            if x > T:
                continue
            new_bits = bits | ((bits << x) & MASK)
            if not ((new_bits >> T) & 1):
                selected.append(x)
                bits = new_bits
        return selected, bits
    
    def try_add_back(selected, bits):
        """Try to add back excluded elements."""
        selected_set = set(selected)
        excluded = [x for x in range(1, 101) if x not in selected_set]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in excluded[:]:
                if time.time() >= deadline:
                    break
                if x > T:
                    excluded.remove(x)
                    continue
                new_bits = bits | ((bits << x) & MASK)
                if not ((new_bits >> T) & 1):
                    selected.append(x)
                    selected_set.add(x)
                    excluded.remove(x)
                    bits = new_bits
                    improved = True
        return selected, bits
    
    best = None
    best_len = 0
    
    # Strategy 1: Descending order
    if time.time() < deadline:
        sel, bits = greedy(list(range(100, 0, -1)))
        sel, bits = try_add_back(sel, bits)
        if len(sel) > best_len:
            best, best_len = sel, len(sel)
    
    # Strategy 2: Ascending order
    if time.time() < deadline:
        sel, bits = greedy(list(range(1, 101)))
        sel, bits = try_add_back(sel, bits)
        if len(sel) > best_len:
            best, best_len = sel, len(sel)
    
    # Strategy 3: Random orders
    rng = random.Random(42)
    for _ in range(200):
        if time.time() >= deadline:
            break
        order = list(range(1, 101))
        rng.shuffle(order)
        sel, bits = greedy(order)
        sel, bits = try_add_back(sel, bits)
        if len(sel) > best_len:
            best, best_len = sel, len(sel)
    
    # Strategy 4: Start from full set, remove minimally
    if time.time() < deadline:
        # Find minimum removals using greedy: repeatedly find a subset summing to 200, remove one element
        removed = set()
        for _ in range(100):
            if time.time() >= deadline:
                break
            current = [x for x in range(1, 101) if x not in removed]
            if not has_sum_200(current):
                break
            # Find one subset summing to 200 using DP with reconstruction
            # Use bitset to find which elements are needed
            dp = {0: []}
            for x in current:
                if x > T:
                    continue
                new_dp = {}
                for s, subset in dp.items():
                    ns = s + x
                    if ns <= T:
                        if ns not in dp:
                            new_dp[ns] = subset + [x]
                dp.update(new_dp)
                if 200 in dp:
                    break
            if 200 in dp:
                subset = dp[200]
                # Remove the element that appears in the most subsets (heuristic)
                # For simplicity, remove the largest
                remove_elem = max(subset)
                removed.add(remove_elem)
            else:
                break
        sel = [x for x in range(1, 101) if x not in removed]
        if len(sel) > best_len:
            best, best_len = sel, len(sel)
    
    # Strategy 5: Try specific patterns
    if time.time() < deadline:
        # Try: exclude all multiples of k for various k
        for k in range(2, 20):
            if time.time() >= deadline:
                break
            excluded = set(range(k, 101, k))
            sel = [x for x in range(1, 101) if x not in excluded]
            if not has_sum_200(sel):
                if len(sel) > best_len:
                    best, best_len = sel, len(sel)
        
        # Try: take all numbers > 100 (none) or specific ranges
        # Take [1..50] U [151..200] but we only have [1..100]
        # Take [1..49] U [100]: 49+1 = 50 elements
        for low, high in [(1, 49), (1, 50), (1, 51), (50, 100), (51, 100)]:
            if time.time() >= deadline:
                break
            sel = list(range(low, high + 1))
            if not has_sum_200(sel):
                if len(sel) > best_len:
                    best, best_len = sel, len(sel)
    
    # Strategy 6: More targeted - try to exclude a small set
    if time.time() < deadline and best_len < 99:
        # Try excluding 1 element
        for x in range(1, 101):
            if time.time() >= deadline:
                break
            sel = [i for i in range(1, 101) if i != x]
            if not has_sum_200(sel):
                if len(sel) > best_len:
                    best, best_len = sel, len(sel)
                break
        
        # Try excluding 2 elements  
        if best_len < 99:
            found = False
            for i in range(1, 101):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, 101):
                    if time.time() >= deadline:
                        break
                    sel = [x for x in range(1, 101) if x != i and x != j]
                    if not has_sum_200(sel):
                        if len(sel) > best_len:
                            best, best_len = sel, len(sel)
                        found = True
                        break
                if found:
                    break
    
    if best is None or len(best) == 0:
        best = [1]
    
    return best
# EVOLVE-BLOCK-END
