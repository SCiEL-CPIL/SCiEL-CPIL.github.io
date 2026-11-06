import time
import random

def _can_add(S, x):
    """
    Check if adding x to set S maintains the validity condition.
    Condition: No element in S U {x} divides the sum of any 3 distinct OTHER elements in S U {x}.
    """
    S_list = list(S)
    m = len(S_list)
    
    # 1) Check if x divides the sum of any 3 distinct elements in S.
    #    If yes, then x is invalid.
    #    We need: (a + b + c) % x != 0 for all distinct a,b,c in S.
    if m >= 3:
        # Optimization: Use residues?
        # Just triple loop for now, m is small.
        for i in range(m - 2):
            a = S_list[i]
            for j in range(i + 1, m - 1):
                b = S_list[j]
                for k in range(j + 1, m):
                    c = S_list[k]
                    if (a + b + c) % x == 0:
                        return False

    # 2) Check for each y in S, if y divides (x + u + v) for any distinct u,v in S \ {y, x}.
    #    Note: u and v must be distinct from y and x, and from each other.
    #    Condition: (x + u + v) % y != 0.
    #    Equiv: (u + v) % y != (-x) % y.
    
    for y in S_list:
        target = (-x) % y
        # Find if any pair u,v in S \ {y, x} sums to target mod y.
        # Note: x is not in S_list yet, so we just iterate S_list.
        # But u,v must be distinct from y.
        # Also u != v.
        
        # Create list of elements in S excluding y
        others = [v for v in S_list if v != y]
        # Note: x is not in S, so others is just S \ {y}.
        # We need to check pairs from 'others'.
        
        seen_residues = set()
        for u in others:
            ru = u % y
            # We need (v % y) such that (ru + rv) % y == target
            # rv = (target - ru) % y
            needed_rv = (target - ru) % y
            
            # If needed_rv is in seen_residues, we found a pair (prev_v, u)
            # BUT we must ensure prev_v != u.
            # Since we iterate u sequentially and check against previous residues,
            # the previous residue came from a different element v_prev.
            # So v_prev != u is guaranteed.
            
            if needed_rv in seen_residues:
                return False
            
            seen_residues.add(ru)
            
    return True

def _is_valid(S):
    """Full validation."""
    S_list = list(S)
    m = len(S_list)
    for a in S_list:
        # Check if a divides sum of any 3 distinct others
        others = [v for v in S_list if v != a]
        if len(others) < 3:
            continue
        # Triple loop
        found = False
        for i in range(len(others) - 2):
            for j in range(i + 1, len(others) - 1):
                for k in range(j + 1, len(others)):
                    if (others[i] + others[j] + others[k]) % a == 0:
                        found = True
                        break
                if found: break
            if found: break
        if found:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 192
    best_set = set()
    
    # Strategy 1: Descending Greedy
    # Large numbers are generally safer.
    order = list(range(n, 0, -1))
    current_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(current_set, x):
            current_set.add(x)
    if len(current_set) > len(best_set):
        best_set = current_set.copy()
        
    # Strategy 2: Randomized Greedy (Multiple Restarts)
    # Try different orderings.
    # Heuristic: Numbers with fewer divisors might be better?
    # Or just random shuffle.
    
    rng = random.Random(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Mix of large and random
        # Bias towards larger numbers
        order = list(range(n, 0, -1))
        rng.shuffle(order)
        
        current_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if _can_add(current_set, x):
                current_set.add(x)
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
    # Local Search / Improvement
    # Try to add more elements to best_set
    # Or swap elements to allow more additions
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Try adding any unselected element
        for x in range(1, n + 1):
            if time.time() > deadline - 5.0:
                break
            if x not in best_set:
                if _can_add(best_set, x):
                    best_set.add(x)
                    improved = True
                    # Restart the loop to try adding more
                    break
        
        # If no direct add, try 1-out 2-in swap?
        # This is expensive. Maybe just 1-out 1-in to change composition, then try add again.
        if not improved:
            # Try swapping one element out to see if it enables an add
            # Pick an element a from best_set
            # Remove it, try to add 2 elements?
            # Simpler: Try to remove a and add x, then try to add y.
            # This is O(N^3) per swap.
            # Let's just try a few random swaps.
            
            for _ in range(10):
                if time.time() > deadline - 5.0:
                    break
                if len(best_set) < 2:
                    break
                a = random.choice(list(best_set))
                trial_set = best_set - {a}
                
                # Try to add a new element x to trial_set
                added = False
                for x in range(1, n + 1):
                    if x in trial_set:
                        continue
                    if _can_add(trial_set, x):
                        trial_set.add(x)
                        added = True
                        # Now try to add another y to trial_set
                        for y in range(1, n + 1):
                            if y in trial_set:
                                continue
                            if _can_add(trial_set, y):
                                trial_set.add(y)
                                # We replaced 1 with 2, net gain +1
                                best_set = trial_set
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break

    # Final Validation
    if _is_valid(best_set):
        return list(best_set)
    
    # Fallback if something went wrong (shouldn't happen)
    return [192]
