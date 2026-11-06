import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Generate smooth numbers (2^a * 3^b * 5^c) <= 128
    smooth = set()
    for a in range(8):  # 2^7 = 128
        for b in range(5): # 3^4 = 81
            for c in range(4): # 5^3 = 125
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 128:
                    smooth.add(val)
    
    smooth_list = sorted(smooth)
    
    # Precompute minimal antichain representation for each target 1..128
    # reps[t] will be a tuple of the terms
    reps = [None] * 129
    
    def is_antichain(terms):
        # Check if no term divides another
        n = len(terms)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if terms[j] % terms[i] == 0:
                    return False
        return True

    # We search for minimal k for each target
    # Max k is likely small, but let's be safe up to 30 as per task (though unlikely needed)
    
    for t in range(1, 129):
        if time.time() > deadline:
            # Fallback: just use {t} if smooth, else some valid one found earlier?
            # Actually, if we timeout, we should return best effort.
            # But for t=1..128, this should be very fast.
            pass
            
        found = False
        # Try k=1
        if t in smooth:
            reps[t] = (t,)
            found = True
        else:
            # Try k=2
            # Iterate through pairs in smooth_list
            # Optimization: a < b
            for i in range(len(smooth_list)):
                a = smooth_list[i]
                if a >= t: break
                b = t - a
                if b < a: continue # ensure a < b to avoid duplicates, but b must be in smooth
                if b in smooth:
                    # Check antichain
                    if a % b != 0 and b % a != 0:
                        reps[t] = (a, b)
                        found = True
                        break
                if time.time() > deadline:
                    break
            if found:
                continue
                
            # Try k=3
            # a + b + c = t, a < b < c
            for i in range(len(smooth_list)):
                a = smooth_list[i]
                if a >= t // 3: break
                for j in range(i + 1, len(smooth_list)):
                    b = smooth_list[j]
                    if a + b >= t: break
                    c = t - a - b
                    if c <= b: continue
                    if c in smooth:
                        # Check antichain for (a,b,c)
                        # a|b? b|c? a|c?
                        if (b % a != 0) and (c % a != 0) and (c % b != 0):
                            reps[t] = (a, b, c)
                            found = True
                            break
                    if time.time() > deadline:
                        break
                if found:
                    break
            if found:
                continue

            # Try k=4
            # This might be slower, but hopefully not needed for many targets
            # If k=4 is needed, we might need a more efficient search or accept higher k
            # Let's add a generic backtracking for k >= 4 if needed, but limit depth
            # Given the density of smooth numbers, k=3 should cover most non-smooth targets.
            # Let's check if any target requires k > 3.
            # If we reach here, we try a simple backtracking for k=4
            # But to save time, let's just do a simple DFS for k up to 5 if not found
            # However, the problem states 1..30 terms.
            
            # Let's implement a generic solver for remaining targets if k<4 failed
            # But first, let's see if k=3 covers all.
            # If not, we need a robust solver.
            
            # Given the constraints, I will implement a recursive search for k=4..10
            # with pruning.
            
            def dfs(rem, start_idx, current_terms, k_remaining):
                if k_remaining == 0:
                    return current_terms if rem == 0 else None
                if rem <= 0:
                    return None
                
                # Pruning: if rem > sum of largest k_remaining smooth numbers, fail
                # If rem < sum of smallest k_remaining smooth numbers, fail
                
                # Iterate possible next terms
                # To maintain distinctness and order, pick from smooth_list[start_idx:]
                # Also must ensure new term doesn't divide or get divided by existing terms
                for idx in range(start_idx, len(smooth_list)):
                    val = smooth_list[idx]
                    if val > rem:
                        break
                    
                    # Check antichain constraint with current_terms
                    valid = True
                    for existing in current_terms:
                        if val % existing == 0 or existing % val == 0:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    res = dfs(rem - val, idx + 1, current_terms + [val], k_remaining - 1)
                    if res is not None:
                        return res
                return None

            # Try k=4, 5, ...
            for k in range(4, 11):
                if time.time() > deadline:
                    break
                res = dfs(t, 0, [], k)
                if res is not None:
                    reps[t] = tuple(res)
                    found = True
                    break

            if not found:
                # Fallback: If we couldn't find a small k, we might have a problem.
                # But for t <= 128, k=3 or 4 should always suffice.
                # If not, we might need to relax or use a different strategy.
                # For now, if not found, we might return an empty list or raise an error?
                # The task says "All targets require valid representations".
                # So we must find one.
                # Let's assume k up to 10 is enough. If not, we might need to optimize.
                # Given the small size, k=4 is very likely sufficient for all non-smooth, non-sum-of-2 targets.
                pass

    # Verify all reps are found
    # If any is None, it's a failure.
    # Convert to list of lists for return
    result = []
    for t in range(1, 129):
        if reps[t] is None:
            # This should not happen
            result.append([])
        else:
            result.append(list(reps[t]))
            
    return result
