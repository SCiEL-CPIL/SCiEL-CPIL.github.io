import time
import random
import sys

def _check_add(S, x, pair_sums, triple_sums):
    """
    Check if x can be added to S while maintaining B_3[1] property.
    S is a list of sorted integers.
    pair_sums is a set of all a+b for a<=b in S.
    triple_sums is a set of all a+b+c for a<=b<=c in S.
    Returns (valid, new_pair_sums, new_triple_sums)
    """
    n = len(S)
    # 1. Check Sidon condition (unique pair sums)
    new_pair_sums = set()
    new_pair_sums.add(2 * x)
    for s in S:
        new_pair_sums.add(x + s)
    
    # Check collision with existing pair sums
    if new_pair_sums & pair_sums:
        return False, None, None
        
    # Check internal collision in new pair sums
    # 2x == x+s => x=s (distinct, so no)
    # x+s1 == x+s2 => s1=s2 (distinct, so no)
    # So new pair sums are unique among themselves.
    
    # 2. Check Triple Sum collisions
    new_triple_sums = set()
    new_triple_sums.add(3 * x)
    
    for s in S:
        new_triple_sums.add(2 * x + s)
        
    for i in range(n):
        for j in range(i, n):
            new_triple_sums.add(x + S[i] + S[j])
            
    # Check collision with existing triple sums
    if new_triple_sums & triple_sums:
        return False, None, None
        
    # Check internal collision in new triple sums?
    # We established that if S is Sidon, new triples involving x don't collide with each other.
    # Proof: x+a+b == x+c+d => a+b == c+d. Since S is Sidon, {a,b}={c,d}. Triples are identical.
    # So no internal collisions.
    
    return True, new_pair_sums, new_triple_sums

def _remove(S, x, pair_sums, triple_sums):
    """
    Remove x from S. Returns new pair_sums and triple_sums.
    """
    n = len(S)
    S_idx = S.index(x)
    S_new = S[:S_idx] + S[S_idx+1:]
    
    new_pair_sums = set()
    new_triple_sums = set()
    
    # Rebuilding from scratch is O(n^3) or O(n^2).
    # For n~40, O(n^3) is 64000 ops. Very fast.
    # Let's just rebuild to be safe and simple.
    for i in range(len(S_new)):
        for j in range(i, len(S_new)):
            new_pair_sums.add(S_new[i] + S_new[j])
            for k in range(j, len(S_new)):
                new_triple_sums.add(S_new[i] + S_new[j] + S_new[k])
                
    return S_new, new_pair_sums, new_triple_sums

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    best_set = []
    best_size = 0
    
    # Initial greedy construction to get a decent start
    # Try to build a set greedily
    S = []
    pair_sums = set()
    triple_sums = set()
    
    # Greedy add smallest possible
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        valid, np, nt = _check_add(S, x, pair_sums, triple_sums)
        if valid:
            S.append(x)
            pair_sums |= np
            triple_sums |= nt
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
        if len(S) >= 42: # Theoretical max
            break
            
    # Local Search
    # We will try to improve the set found by greedy
    # Strategy: Random swaps and additions
    current_S = S[:]
    current_pair_sums = pair_sums.copy()
    current_triple_sums = triple_sums.copy()
    current_size = len(current_S)
    
    if current_size > best_size:
        best_size = current_size
        best_set = current_S[:]
        
    # If greedy didn't find anything, start with small random set
    if not current_S:
        # Fallback
        current_S = [1, 2, 5] # Known valid small set?
        # Rebuild sums
        current_pair_sums = set()
        current_triple_sums = set()
        for i in range(len(current_S)):
            for j in range(i, len(current_S)):
                current_pair_sums.add(current_S[i] + current_S[j])
                for k in range(j, len(current_S)):
                    current_triple_sums.add(current_S[i] + current_S[j] + current_S[k])
        current_size = len(current_S)
        
    # Main search loop
    while time.time() < deadline:
        # Try to add a random element
        added = False
        for _ in range(100): # Try 100 random additions
            x = random.randint(1, 4096)
            if x in current_S:
                continue
            valid, np, nt = _check_add(current_S, x, current_pair_sums, current_triple_sums)
            if valid:
                current_S.append(x)
                current_S.sort()
                current_pair_sums |= np
                current_triple_sums |= nt
                current_size += 1
                added = True
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_S[:]
                break # Restart addition attempts with new set state? Or continue?
                # Continuing might allow multiple adds in one loop iter
                # But we break to re-evaluate state in next outer loop iteration
                # Actually, let's just continue trying to add if we have time
                # But we need to update best_set
                # Let's not break, just continue the inner loop?
                # No, the set changed, so previous checks are invalid.
                # Break inner loop to go to next outer iteration (which might do a swap)
                # Or just continue checking other x?
                # If we add x, we can try to add y.
                # Let's just break and let the outer loop handle it.
                pass 
        
        if not added:
            # Try a random swap
            if len(current_S) > 0:
                idx_remove = random.randint(0, len(current_S) - 1)
                x_remove = current_S[idx_remove]
                
                # Try to find a replacement
                # Sample some candidates
                for _ in range(50):
                    x_add = random.randint(1, 4096)
                    if x_add in current_S:
                        continue
                    
                    # Perform swap: remove x_remove, add x_add
                    # Check if S \ {x_remove} U {x_add} is valid
                    # We need to check Sidon and Triple sums for the new set.
                    # It's easier to rebuild for the candidate set if n is small.
                    # Or use the _check_add logic on S \ {x_remove}
                    
                    S_temp = current_S[:idx_remove] + current_S[idx_remove+1:]
                    # Check if x_add can be added to S_temp
                    # We need pair_sums and triple_sums for S_temp.
                    # Rebuilding them is O(n^3) ~ 40^3 = 64000.
                    # 50 swaps * 64000 = 3.2M ops. Fast.
                    
                    p_temp = set()
                    t_temp = set()
                    for i in range(len(S_temp)):
                        for j in range(i, len(S_temp)):
                            p_temp.add(S_temp[i] + S_temp[j])
                            for k in range(j, len(S_temp)):
                                t_temp.add(S_temp[i] + S_temp[j] + S_temp[k])
                    
                    valid, np, nt = _check_add(S_temp, x_add, p_temp, t_temp)
                    if valid:
                        # Accept swap
                        current_S = S_temp + [x_add]
                        current_S.sort()
                        current_pair_sums = p_temp | np
                        current_triple_sums = t_temp | nt
                        # Size remains same
                        break
        
        # Occasionally, try to restart with a different greedy seed?
        # Or just keep going.
        
    return best_set
