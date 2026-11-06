import time
import random

def _greedy_with_order(n, order):
    """
    Runs the greedy algorithm using the specified order of candidates.
    Returns the set of selected numbers.
    """
    S = set()
    S_list = []
    
    for v in order:
        valid = True
        # Check all pairs b, c in S_list
        # b, c are > v? No, order is arbitrary.
        # Condition: a < b <= c.
        # If we add v, v can be 'a', 'b', or 'c'.
        # The condition is: For all a, b, c in S_new with a < b <= c, a does not divide b+c.
        
        # If we just check if v violates the condition with existing pairs, we must consider all roles.
        # However, the original program assumed we add in decreasing order, so v is always the smallest 'a'.
        # If we use a random order, v can be larger than some elements in S.
        # So v can be 'b' or 'c' as well.
        
        # Let's check all triples involving v.
        # Case 1: v is 'a'. Then we need b, c in S such that v < b <= c and v | b+c.
        # Case 2: v is 'b'. Then we need a, c in S such that a < v <= c and a | v+c.
        #           Note: if v=c, then a < v and a | 2v.
        # Case 3: v is 'c'. Then we need a, b in S such that a < b < v and a | b+v.
        
        # This is O(K^2) per insertion if we do it naively.
        # K is size of S.
        
        L = len(S_list)
        # We can optimize by just checking pairs in S.
        # For each pair (x, y) in S:
        # 1. If v < x <= y: check v | x+y
        # 2. If x < v <= y: check x | v+y
        # 3. If x < y < v: check x | y+v
        # 4. If x < v and y < v: check x | v+y (same as 3 if we assume x<=y)
        
        # Actually, simpler:
        # Iterate all pairs (b, c) in S with b <= c.
        # Check if v can be 'a': if v < b, check v | b+c.
        # Check if v can be 'b': if b < v <= c, check b | v+c.
        # Check if v can be 'c': if b < c < v, check b | c+v.
        # Wait, the roles are symmetric for b and c in the sum, but the inequality a < b <= c fixes them.
        
        # Let's just check all pairs (u, w) in S.
        for i in range(L):
            b = S_list[i]
            for j in range(i, L):
                c = S_list[j]
                # Pairs (b, c) with b <= c are in S.
                
                # If v is the smallest (a=v):
                if v < b:
                    if (b + c) % v == 0:
                        valid = False
                        break
                # If v is in the middle (b=v):
                elif b < v <= c:
                    if (v + c) % b == 0:
                        valid = False
                        break
                # If v is the largest (c=v):
                elif b < c < v:
                    if (b + v) % b == 0: # a=b, c=v. a divides b+v? No, a divides c+b.
                        # Condition: a | b+c. Here a=b, b=c_old, c=v.
                        # So b | c_old + v.
                        if (c + v) % b == 0:
                            valid = False
                            break
                # If v is smaller than both? Covered by v < b.
                # If v is between b and c? Covered by b < v <= c.
                # What if v == b or v == c?
                # If v == b: b < v is false. v < b is false.
                # The condition a < b <= c.
                # If v is added, and v is already in S? No, we iterate distinct v.
                
                # Let's be precise.
                # We have existing set S. We add v.
                # New triples involving v:
                # 1. (v, b, c) with v < b <= c. Check v | b+c.
                # 2. (b, v, c) with b < v <= c. Check b | v+c.
                # 3. (b, c, v) with b < c < v. Check b | c+v.
                
                # My loop above covers:
                # if v < b: Case 1.
                # elif b < v <= c: Case 2.
                # elif b < c < v: Case 3.
                
                # What if b == v? Impossible, v not in S.
                # What if c == v? Impossible.
                # What if b < v and c < v?
                # Then b < c < v is false if b < v and c < v but b > c? No b<=c.
                # If b < c < v, it's Case 3.
                # If b < v < c, it's Case 2 (since b < v <= c).
                # If v < b <= c, it's Case 1.
                
                # Are there any gaps?
                # b <= c.
                # v < b => Case 1.
                # b < v <= c => Case 2.
                # b < c < v => Case 3.
                # What if v is between b and c but v > c? No.
                # What if v < c but v > b? That is b < v < c. Covered by Case 2 (b < v <= c).
                # What if v < b? Covered by Case 1.
                # What if v > c? Then b < c < v. Covered by Case 3.
                
                # Seems complete.
            if not valid:
                break
        if not valid:
            continue
        
        S.add(v)
        S_list.append(v)
        
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 256
    numbers = list(range(1, n + 1))
    
    best_set = set()
    
    # 1. Deterministic greedy (descending)
    order_desc = sorted(numbers, reverse=True)
    S = _greedy_with_order(n, order_desc)
    if len(S) > len(best_set):
        best_set = S
        
    # 2. Random restarts
    # Estimate time per run.
    # K ~ 150. Pairs ~ 11k. 256 items * 11k = 2.8M ops.
    # Python ~ 10M ops/sec? Maybe 1-2M/sec for complex logic.
    # Let's say 0.1s per run.
    # 50s / 0.1s = 500 runs.
    
    attempts = 0
    while time.time() < deadline:
        random.shuffle(numbers)
        S = _greedy_with_order(n, numbers)
        if len(S) > len(best_set):
            best_set = S
            # Optional: local search on best_set?
            # For now, just restarts.
        attempts += 1
        if attempts % 100 == 0:
            # Check time more frequently?
            pass
            
    # Final local search: try to add any number
    improved = True
    while improved:
        improved = False
        for v in range(1, n + 1):
            if v in best_set:
                continue
            # Check if v can be added to best_set
            valid = True
            S_list = list(best_set)
            L = len(S_list)
            for i in range(L):
                b = S_list[i]
                for j in range(i, L):
                    c = S_list[j]
                    if v < b:
                        if (b + c) % v == 0:
                            valid = False
                            break
                    elif b < v <= c:
                        if (v + c) % b == 0:
                            valid = False
                            break
                    elif b < c < v:
                        if (c + v) % b == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_set.add(v)
                improved = True
                break # Restart loop as set changed
                
    return sorted(best_set)
