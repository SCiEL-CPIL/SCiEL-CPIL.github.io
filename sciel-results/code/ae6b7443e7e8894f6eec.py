import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin for 60s limit
    
    # 1. Generate 2,3,5-smooth numbers up to 128
    smooth = set()
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.add(x)
    
    smooth_list = sorted(smooth)
    smooth_set = smooth
    
    def is_antichain(terms):
        # terms is a tuple of distinct values
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                a, b = terms[i], terms[j]
                if a > b:
                    a, b = b, a
                if b % a == 0:
                    return False
        return True

    answers = []
    
    # Precompute valid antichains by size for quick lookup?
    # Or just search per target.
    # Given N=128, per-target search is fine.
    
    for n in range(1, 129):
        if time.time() > deadline:
            # Fallback: return whatever we have or a valid non-optimal solution
            # Since we process in order, we need a valid solution for all.
            # If we run out of time, we should have finished long ago.
            # This is just a safety net.
            break
            
        found = None
        
        # Try size 1
        if n in smooth_set:
            found = [n]
        else:
            # Try size 2
            found = None
            for a in smooth_list:
                if a >= n:
                    break
                b = n - a
                if b in smooth_set and b > a: # b > a ensures distinct and avoids double checking
                    if b % a != 0 and a % b != 0: # a%b!=0 is always true if b>a and b!=0, but b%a!=0 is the check
                        if b % a != 0:
                            found = [a, b]
                            break
            
            if found is None:
                # Try size 3
                # Iterate a < b < c
                for a in smooth_list:
                    if a >= n:
                        break
                    for b in smooth_list:
                        if b <= a:
                            continue
                        if a + b >= n:
                            break
                        c = n - a - b
                        if c in smooth_set and c > b:
                            # Check antichain condition for (a, b, c)
                            if (b % a != 0) and (c % a != 0) and (c % b != 0):
                                found = [a, b, c]
                                break
                    if found:
                        break
            
            if found is None:
                # Try size 4
                # This might be needed for some numbers?
                # Let's add a generic search for size 4 if needed
                # But likely size 3 covers all?
                # Let's check if any number requires 4 terms.
                # If not found in 1, 2, 3, try 4.
                # To be safe, implement a simple recursive search or itertools for size 4.
                # Given the density, size 3 is very likely sufficient for most, but let's ensure validity.
                
                # We can use combinations from smooth_list
                # Filter smooth_list to those < n
                candidates = [x for x in smooth_list if x < n]
                
                # Try size 4
                found = None
                # Optimization: only consider combinations that sum to n
                # itertools.combinations is slow for large sets, but candidates <= 38
                # 38C4 is ~70k, which is fine.
                for combo in itertools.combinations(candidates, 4):
                    if sum(combo) == n:
                        if is_antichain(combo):
                            found = list(combo)
                            break
                            
                if found is None:
                    # Try size 5 (unlikely, but for completeness)
                    for combo in itertools.combinations(candidates, 5):
                        if sum(combo) == n:
                            if is_antichain(combo):
                                found = list(combo)
                                break
                    
                if found is None:
                    # Fallback: should not happen, but if it does, find ANY valid antichain
                    # This would be expensive, but we assume 1-4 is enough.
                    # If we get here, we have a problem.
                    # Let's just pick a valid one from a broader search if possible.
                    # For now, raise an error or return empty? No, must return valid.
                    # Let's assume size <= 4 is always possible.
                    pass
        
        if found is None:
             # Emergency fallback: if no small antichain found, find any.
             # This is very unlikely to be triggered.
             # We can do a BFS or just return a known valid one if we had one.
             # But we must construct it.
             # Let's try size up to 10 just in case.
             candidates = [x for x in smooth_list if x < n]
             for k in range(4, 11):
                 for combo in itertools.combinations(candidates, k):
                     if sum(combo) == n and is_antichain(combo):
                         found = list(combo)
                         break
                 if found:
                     break
             
             if found is None:
                 # If still None, something is wrong with the logic or constraints.
                 # Return a dummy valid list? No, it must sum to n.
                 # This should not happen for n<=128.
                 found = [] # Will cause error in evaluation if empty, but better than crash?
                 # Actually, the task says "All targets require valid representations".
                 # So this path is a failure.
        
        answers.append(found)
        
    return answers
