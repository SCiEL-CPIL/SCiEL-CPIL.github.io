import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256
    
    # Helper to check if x can be added to S
    # S is a set for O(1) lookup, but we iterate over list for speed in some cases?
    # Given small size, list iteration is fine.
    
    def is_valid_addition(x, S_list):
        """
        Check if adding x to S_list maintains validity.
        S_list is a list of elements in the set.
        """
        if not S_list:
            return True
            
        # We need to check all triples (a, b, c) in S_new = S_list U {x}
        # such that a < b <= c and a does not divide b+c.
        
        # Violations must involve x.
        
        # Case 1: x is a
        # Need b, c in S_list such that x < b <= c and x | (b+c)
        # Note: b, c must be > x
        
        # Case 2: x is b
        # Need a, c in S_list such that a < x <= c and a | (x+c)
        # Note: c can be x? No, c is in S_list, x not in S_list.
        # So c > x is required for the "c" part if we treat x as b?
        # Condition: a < b <= c. If b=x, then a < x and x <= c.
        # Since c is in S_list, c != x. So c > x.
        
        # Case 3: x is c
        # Need a, b in S_list such that a < b <= x and a | (b+x)
        # Note: b < x is required since b in S_list and b != x.
        
        # Optimization: Separate S_list into less and greater than x
        less = []
        greater = []
        for y in S_list:
            if y < x:
                less.append(y)
            else:
                greater.append(y)
                
        # Case 1: x is a
        # b, c in greater, b <= c
        if len(greater) >= 2:
            g_sorted = sorted(greater)
            for i in range(len(g_sorted)):
                b = g_sorted[i]
                for j in range(i, len(g_sorted)):
                    c = g_sorted[j]
                    if (b + c) % x == 0:
                        return False
        
        # Case 2: x is b
        # a in less, c in greater (since c > x)
        # Condition: a < x <= c.
        # Check a | (x+c)
        for a in less:
            for c in greater:
                if (x + c) % a == 0:
                    return False
            
            # Also check b=x, c=x? 
            # The problem says "Repeated larger summands b=c are included".
            # So (a, x, x) is a valid triple to check.
            # Condition: a < x <= x. True.
            # Check a | (x+x) = 2x.
            if (2 * x) % a == 0:
                return False

        # Case 3: x is c
        # a, b in less, a < b
        # Condition: a < b <= x.
        # Check a | (b+x)
        if len(less) >= 2:
            l_sorted = sorted(less)
            for i in range(len(l_sorted)):
                a = l_sorted[i]
                for j in range(i + 1, len(l_sorted)):
                    b = l_sorted[j]
                    # b > a guaranteed
                    if (b + x) % a == 0:
                        return False
                    
        return True

    def is_valid_set(S_list):
        """Full check, used for validation."""
        S_sorted = sorted(S_list)
        m = len(S_sorted)
        for i in range(m):
            a = S_sorted[i]
            for j in range(i + 1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # Baseline: [171, 256]
    base_set = list(range(171, 257))
    if not is_valid_set(base_set):
        # Should not happen, but just in case
        base_set = list(range(171, 257))
        
    best_set = base_set[:]
    best_size = len(base_set)
    
    # Local Search
    random.seed(42)
    
    # We will try to improve best_set
    # Strategy: Random Restarts with Greedy Initialization and Local Search
    
    for attempt in range(100):
        if time.time() > deadline - 10:
            break
            
        # Create a perturbed version of the current best or a fresh greedy one
        if attempt == 0:
            S = best_set[:]
        else:
            # Random restart: Greedy from high to low with some randomization
            candidates = list(range(n, 0, -1))
            # Shuffle the first part to introduce variety, keep the tail mostly intact?
            # Or just shuffle all?
            random.shuffle(candidates)
            # Bias towards high numbers?
            # Sort candidates descending, but swap some random ones
            S = []
            for x in candidates:
                if time.time() > deadline - 5:
                    break
                if is_valid_addition(x, S):
                    S.append(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        
        # Local Search Phase on S
        # Try to add any element, or swap
        # We'll do a few rounds of trying to add elements
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try to add any element
            remaining = [x for x in range(1, n + 1) if x not in S]
            random.shuffle(remaining)
            
            for x in remaining:
                if is_valid_addition(x, S):
                    S.append(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = S[:]
                    break # Restart adding loop after an addition to re-evaluate remaining? 
                         # Or just continue? Continuing is faster if we track remaining.
                         # But S changed, so validity of previous 'no' might change? No, adding elements only creates more constraints.
                         # So if x couldn't be added before, it can't be added later unless we remove something.
                         # So we can just break and restart the 'remaining' calculation? 
                         # Actually, if we add x, other elements that were blocked by x might now be blocked by the new set? 
                         # No, adding x makes the set "worse" for others.
                         # So we just continue checking other x in remaining?
                         # But `is_valid_addition` depends on S.
                         # Let's just do one pass of adding.
            
            if not improved:
                # Try swaps: Remove one, add two? Or Remove one, add one (size same, maybe enables future adds)
                # Simple swap: Remove y from S, try to add x not in S.
                # If we can add 2 for every 1 removed, size increases.
                # Let's try: Pick a random y in S. Remove it.
                # Try to add any x. If added, great.
                # This doesn't guarantee size increase.
                
                # More aggressive: 
                # Pick 2 elements to remove, try to add 3?
                
                # Let's try a single swap first to see if we can reach a state where we can add more.
                if time.time() < deadline - 2:
                    S_copy = S[:]
                    if S_copy:
                        y_idx = random.randint(0, len(S_copy) - 1)
                        y = S_copy.pop(y_idx)
                        
                        # Try to add something back
                        remaining = [x for x in range(1, n + 1) if x not in S_copy]
                        random.shuffle(remaining)
                        added = False
                        for x in remaining:
                            if is_valid_addition(x, S_copy):
                                S_copy.append(x)
                                added = True
                                break
                        
                        if added:
                            # Check if we can add MORE now
                            added_more = True
                            while added_more and time.time() < deadline - 2:
                                added_more = False
                                remaining = [x for x in range(1, n + 1) if x not in S_copy]
                                random.shuffle(remaining)
                                for x in remaining:
                                    if is_valid_addition(x, S_copy):
                                        S_copy.append(x)
                                        added_more = True
                                        break
                            
                            if len(S_copy) > len(S):
                                S = S_copy
                                improved = True
                                if len(S) > best_size:
                                    best_size = len(S)
                                    best_set = S[:]
                            # If size is same, we might still accept it to explore?
                            # For now, only accept if size improves or if we are stuck and want to perturb.
                            # Let's accept if size is same to avoid local minima?
                            # Risk: getting stuck in a loop of same-size swaps.
                            # Accept with probability or just if size improves.
                            elif len(S_copy) == len(S) and random.random() < 0.1:
                                S = S_copy

    # Final verification and cleanup
    if not is_valid_set(best_set):
        # Fallback to valid set
        return list(range(171, 257))
        
    return best_set
