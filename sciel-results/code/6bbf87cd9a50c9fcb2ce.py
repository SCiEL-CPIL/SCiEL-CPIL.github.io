import time
import random

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    # Helper: Check if adding v to S creates a 4-AP.
    # S is a set. v is not in S.
    # We check if there exists d > 0 such that v, v+d, v+2d, v+3d are in S (or v is part of such a tuple)
    # Actually, since S is 4-AP free, we only need to check if v completes a 4-AP with existing elements.
    # The 4-AP must contain v.
    # Cases:
    # 1. v is 1st: v, v+d, v+2d, v+3d in S
    # 2. v is 2nd: v-d, v, v+d, v+2d in S
    # 3. v is 3rd: v-2d, v-d, v, v+d in S
    # 4. v is 4th: v-3d, v-2d, v-d, v in S
    
    def creates_4ap(S, v):
        # S is a set of ints
        # v is an int not in S
        # Check if v is part of any 4-AP with elements from S
        # Max d is roughly n/3
        for d in range(1, n // 3 + 1):
            # Case 1: v is 1st term
            if v + 3 * d <= n:
                if (v + d in S) and (v + 2 * d in S) and (v + 3 * d in S):
                    return True
            # Case 2: v is 2nd term
            if v - d >= 1:
                if (v - d in S) and (v + d in S) and (v + 2 * d in S):
                    return True
            # Case 3: v is 3rd term
            if v - 2 * d >= 1:
                if (v - 2 * d in S) and (v - d in S) and (v + d in S):
                    return True
            # Case 4: v is 4th term
            if v - 3 * d >= 1:
                if (v - 3 * d in S) and (v - 2 * d in S) and (v - d in S):
                    return True
        return False

    # Helper: Count 3-APs in S
    def count_3aps(S):
        count = 0
        # To optimize, we can iterate over pairs (a, b) in S with a < b
        # and check if 2b - a is in S.
        # Or iterate a, d.
        # Iterating a, d is O(N^2).
        # Let's use the pair method which might be faster if S is sparse?
        # S size ~ 100. Pairs ~ 5000.
        # Let's stick to a, d for simplicity and correctness.
        for a in S:
            for d in range(1, (n - a) // 2 + 1):
                if a + d in S and a + 2 * d in S:
                    count += 1
        return count

    # Helper: Calculate change in 3-AP count when removing u and adding v from S
    # Returns (delta_count, valid)
    # valid is True if S \ {u} U {v} is 4-AP free.
    # We already check 4-AP for v in S \ {u}.
    # We need to count 3-APs lost by removing u and gained by adding v.
    
    def calc_delta_3aps(S, u, v):
        # Remove u: subtract 3-APs containing u
        # A 3-AP containing u can be:
        # u, u+d, u+2d (u is 1st)
        # u-d, u, u+d (u is 2nd)
        # u-2d, u-d, u (u is 3rd)
        
        # Note: We are removing u from S. The other elements must be in S \ {u}.
        # Since u is in S, we check if the other two are in S.
        
        lost = 0
        for d in range(1, n // 2 + 1):
            # u is 1st
            if u + 2 * d <= n:
                if (u + d in S) and (u + 2 * d in S):
                    lost += 1
            # u is 2nd
            if u - d >= 1:
                if (u - d in S) and (u + d in S):
                    lost += 1
            # u is 3rd
            if u - 2 * d >= 1:
                if (u - 2 * d in S) and (u - d in S):
                    lost += 1
        
        # Add v: add 3-APs containing v
        # The other elements must be in S \ {u}.
        # Since v is not in S, and we are adding it to S \ {u},
        # we check if the other two are in S \ {u}.
        # Note: if the other two include u, they are not in S \ {u}.
        # So we check if (x in S) and (x != u).
        
        gained = 0
        for d in range(1, n // 2 + 1):
            # v is 1st
            if v + 2 * d <= n:
                if (v + d in S) and (v + d != u) and (v + 2 * d in S) and (v + 2 * d != u):
                    gained += 1
            # v is 2nd
            if v - d >= 1:
                if (v - d in S) and (v - d != u) and (v + d in S) and (v + d != u):
                    gained += 1
            # v is 3rd
            if v - 2 * d >= 1:
                if (v - 2 * d in S) and (v - 2 * d != u) and (v - d in S) and (v - d != u):
                    gained += 1
        
        return gained - lost

    # 1. Initial Greedy Construction
    current_set = set()
    for x in range(1, n + 1):
        # Check if adding x creates a 4-AP with existing elements
        if not creates_4ap(current_set, x):
            current_set.add(x)
    
    current_3ap_count = count_3aps(current_set)
    best_set = set(current_set)
    best_3ap_count = current_3ap_count
    
    # 2. Local Search (Hill Climbing with Restarts)
    # We will try to improve the set by swapping elements.
    
    while time.time() - start_time < time_limit:
        # Try a number of random swaps
        # If we get stuck (no improvement in many tries), do a restart or kick
        stuck_counter = 0
        max_stuck = 500
        
        while stuck_counter < max_stuck and time.time() - start_time < time_limit:
            if len(current_set) == 0:
                break
            
            # Pick random u to remove
            u = random.choice(list(current_set))
            
            # Pick random v to add
            # v must be in [1, n] and not in current_set
            # To avoid bias, pick from all available
            available = [x for x in range(1, n + 1) if x not in current_set]
            if not available:
                break
            v = random.choice(available)
            
            # Check validity: Does v create a 4-AP in current_set \ {u}?
            # We need to check if v creates a 4-AP with elements in current_set \ {u}.
            # The function creates_4ap(S, v) checks if v creates a 4-AP with elements in S.
            # Here S should be current_set \ {u}.
            # But creates_4ap checks if v, v+d, ... are in S.
            # If u is one of the other elements, it shouldn't be counted.
            # So we can just call creates_4ap(current_set, v) but we must ensure u is not one of the witnesses?
            # No, creates_4ap checks if the *other* 3 terms are in S.
            # If one of the other terms is u, and u is being removed, then that 4-AP is broken.
            # So we need a version that checks if v creates a 4-AP with elements in (S \ {u}).
            
            # Let's write a specific check for this case to be safe and fast.
            valid_move = True
            for d in range(1, n // 3 + 1):
                # Case 1: v is 1st
                if v + 3 * d <= n:
                    t1, t2, t3 = v + d, v + 2 * d, v + 3 * d
                    if (t1 in current_set) and (t2 in current_set) and (t3 in current_set):
                        # If any of these is u, the AP is broken by removing u.
                        # If none is u, the AP exists.
                        if u not in (t1, t2, t3):
                            valid_move = False
                            break
                # Case 2: v is 2nd
                if v - d >= 1:
                    t0, t2, t3 = v - d, v + d, v + 2 * d
                    if (t0 in current_set) and (t2 in current_set) and (t3 in current_set):
                        if u not in (t0, t2, t3):
                            valid_move = False
                            break
                # Case 3: v is 3rd
                if v - 2 * d >= 1:
                    t0, t1, t3 = v - 2 * d, v - d, v + d
                    if (t0 in current_set) and (t1 in current_set) and (t3 in current_set):
                        if u not in (t0, t1, t3):
                            valid_move = False
                            break
                # Case 4: v is 4th
                if v - 3 * d >= 1:
                    t0, t1, t2 = v - 3 * d, v - 2 * d, v - d
                    if (t0 in current_set) and (t1 in current_set) and (t2 in current_set):
                        if u not in (t0, t1, t2):
                            valid_move = False
                            break
            
            if not valid_move:
                stuck_counter += 1
                continue
            
            # Calculate delta
            delta = calc_delta_3aps(current_set, u, v)
            
            if delta > 0:
                # Accept move
                current_set.remove(u)
                current_set.add(v)
                current_3ap_count += delta
                stuck_counter = 0
                
                # Update best
                if current_3ap_count > best_3ap_count:
                    best_3ap_count = current_3ap_count
                    best_set = set(current_set)
            else:
                # Reject move (Hill Climbing)
                stuck_counter += 1
        
        # If we got stuck, do a "kick" to escape local optimum
        # Remove a random element to create space, or just restart from best?
        # Let's restart from best_set but with a small perturbation.
        if time.time() - start_time < time_limit:
            current_set = set(best_set)
            current_3ap_count = best_3ap_count
            # Perturb: remove 1 random element
            if current_set:
                u = random.choice(list(current_set))
                current_set.remove(u)
                current_3ap_count -= 0 # Recounting is safer after perturbation
                current_3ap_count = count_3aps(current_set)

    return sorted(best_set)
