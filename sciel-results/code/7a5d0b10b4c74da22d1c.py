import numpy as np
import time

def _get_divisors(N):
    """Get all divisors of N greater than 1, sorted."""
    divisors = set()
    for i in range(2, int(np.sqrt(N)) + 1):
        if N % i == 0:
            divisors.add(i)
            divisors.add(N // i)
    divisors.add(N)
    return sorted(divisors)

def _score(classes, N):
    """Calculate the number of uncovered residues."""
    if not classes:
        return N
    covered = np.zeros(N, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return int(np.sum(1 - covered))

def _greedy_build(N, divisors, time_limit=10.0, start_time=None):
    """Run greedy algorithm to build an initial covering system."""
    if start_time is None:
        start_time = time.time()
    
    covered = np.zeros(N, dtype=np.uint8)
    used_moduli = set()
    classes = []
    
    # Pre-compute divisors for faster access
    divs = sorted(divisors)
    
    while time.time() - start_time < time_limit:
        best_gain = 0
        best_m = None
        best_r = None
        
        for m in divs:
            if m in used_moduli:
                continue
            
            # Find best residue for this m
            # We want to maximize uncovered[r::m]
            # Since N is small, we can just loop
            best_r_for_m = 0
            max_uncovered = -1
            for r in range(m):
                # Count uncovered in this slice
                # covered is uint8, so 0 is uncovered
                unc = np.count_nonzero(covered[r::m] == 0)
                if unc > max_uncovered:
                    max_uncovered = unc
                    best_r_for_m = r
            
            if max_uncovered > best_gain:
                best_gain = max_uncovered
                best_m = m
                best_r = best_r_for_m
        
        if best_gain == 0:
            break
        
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        covered[best_r::best_m] = 1
        
    return classes, covered

def _local_search(classes, N, divisors, time_limit=100.0, start_time=None):
    """Perform local search by trying to swap classes."""
    if start_time is None:
        start_time = time.time()
    
    current_classes = [list(c) for c in classes]
    used_moduli = set(m for m, r in current_classes)
    divs = sorted(divisors)
    
    # Pre-compute "coverage by others" is tricky to update incrementally efficiently 
    # in Python without C extensions, so we will recalculate the delta 
    # by temporarily modifying the covered array.
    
    # However, a full recalculation of covered array for every swap is O(K*N).
    # Let's try to be smarter.
    
    # We will maintain the `covered` array.
    # To evaluate a swap of class i with [m_new, r_new]:
    # 1. Remove class i from covered (set covered[r_i::m_i] to 0, but careful if others cover it).
    #    Actually, we can just compute the delta directly.
    
    # Let's use a simpler but potentially slower approach first to ensure correctness, 
    # then optimize if needed.
    # Given N=51975, K<=48.
    # One pass: for each i (48), for each m_new (48), find best r_new.
    # Finding best r_new requires knowing the state of "covered without i".
    
    # Optimization:
    # For a fixed i, the state "covered without i" is fixed for all m_new.
    # So, for each i:
    #   1. Compute covered_without_i.
    #   2. For each m_new not used, find best r_new that maximizes coverage in covered_without_i.
    #   3. Calculate total score change.
    
    improved = True
    pass_count = 0
    
    while improved and time.time() - start_time < time_limit:
        improved = False
        pass_count += 1
        
        # We iterate over all classes. If we make a swap, we restart the pass 
        # to ensure we are always working on the latest state.
        for i in range(len(current_classes)):
            if time.time() - start_time > time_limit:
                return current_classes
            
            m_old, r_old = current_classes[i]
            
            # Step 1: Build covered state without class i
            # We can do this by taking the full covered array and "uncovering" class i?
            # No, because other classes might cover those spots.
            # We must rebuild from scratch or maintain counts.
            # Rebuilding from scratch is O(K*N) ~ 48 * 52000 ~ 2.5M ops.
            # Doing this 48 times per pass is 120M ops.
            # Doing this for ~100 passes is 12B ops. Too slow in Python.
            
            # Alternative: Maintain an array `cover_count` where cover_count[x] is 
            # the number of classes covering x.
            # Then covered[x] is 1 if cover_count[x] > 0.
            # Removing class i: decrement cover_count for x in r_old::m_old.
            # Adding class j: increment cover_count for x in r_new::m_new.
            
            # Let's switch to cover_count strategy.
            break # Break out to implement cover_count strategy in the loop below
            
    # Restart with cover_count strategy
    cover_count = np.zeros(N, dtype=np.int16)
    for m, r in current_classes:
        cover_count[r::m] += 1
        
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        for i in range(len(current_classes)):
            if time.time() - start_time > time_limit:
                return current_classes
                
            m_old, r_old = current_classes[i]
            
            # Temporarily remove class i
            cover_count[r_old::m_old] -= 1
            
            # Current uncovered count
            # uncovered is where cover_count == 0
            # We want to find a new class [m_new, r_new] that minimizes the final uncovered count.
            # Final uncovered = Current uncovered - (Gain from new class)
            # Gain from new class = number of x in r_new::m_new where cover_count[x] == 0
            
            best_gain = -1
            best_m_new = None
            best_r_new = None
            
            for m_new in divs:
                if m_new == m_old or m_new in used_moduli:
                    continue
                
                # Find best r_new for this m_new
                # We want to maximize count of x in r::m_new where cover_count[x] == 0
                best_r_for_m = 0
                max_gain_for_m = -1
                
                # Optimization: If m_new is large, the slice is small.
                # If m_new is small, the slice is large.
                # We can vectorize the counting for all r?
                # Not easily with standard numpy without reshaping tricks.
                # Given N=51975, looping r is okay if we make it fast.
                
                for r_new in range(m_new):
                    # Count zeros in cover_count[r_new::m_new]
                    # cover_count is int16.
                    gain = np.count_nonzero(cover_count[r_new::m_new] == 0)
                    if gain > max_gain_for_m:
                        max_gain_for_m = gain
                        best_r_for_m = r_new
                
                if max_gain_for_m > best_gain:
                    best_gain = max_gain_for_m
                    best_m_new = m_new
                    best_r_new = best_r_for_m
            
            # Check if this swap improves the solution
            # The old class was covering some spots.
            # How many spots were exclusively covered by the old class?
            # i.e., cover_count[x] was 1, and x is in r_old::m_old.
            # These spots will become uncovered if we remove the old class and the new class doesn't cover them.
            
            # Let's calculate the net change in uncovered count.
            # Uncovered_before = sum(cover_count == 0)
            # Uncovered_after = Uncovered_before - Gain + Loss
            # Where Loss = number of x in r_old::m_old where cover_count[x] (after removal) == 0 
            #              AND x is NOT covered by the new class.
            # Actually, simpler:
            # Let U_old be the set of uncovered residues in the state WITH class i.
            # Let U_new be the set of uncovered residues in the state WITH new class.
            
            # We have cover_count WITHOUT class i.
            # U_intermediate = {x | cover_count[x] == 0}
            
            # The new class covers C_new = {x | x % m_new == r_new}
            # U_new = U_intermediate \ C_new
            
            # The old state had U_old.
            # U_old = (U_intermediate \ C_old) where C_old is the set covered by old class?
            # No.
            # In the state WITH class i, cover_count was cover_count + 1 on C_old.
            # So U_old = {x | cover_count[x] == 0 AND x NOT in C_old}
            #           = U_intermediate \ C_old
            
            # So, U_new = U_intermediate \ C_new
            # U_old = U_intermediate \ C_old
            
            # We want |U_new| < |U_old|
            # |U_intermediate| - |U_intermediate \cap C_new| < |U_intermediate| - |U_intermediate \cap C_old|
            # - |U_intermediate \cap C_new| < - |U_intermediate \cap C_old|
            # |U_intermediate \cap C_new| > |U_intermediate \cap C_old|
            
            # So we just need the new class to cover MORE currently-uncovered (in intermediate state) residues 
            # than the old class did.
            
            # Wait, C_old is the set of residues covered by the old class.
            # In the intermediate state (without old class), the residues in C_old are NOT necessarily uncovered.
            # They are uncovered ONLY IF no other class covers them.
            
            # Let's re-evaluate:
            # Gain from new class = |U_intermediate \cap C_new|
            # "Loss" relative to old class = |U_intermediate \cap C_old|
            
            # So if Gain > Loss, we improve.
            
            # We already calculated Gain (max_gain_for_m) for the best new class.
            # Now we need Loss for the old class in the intermediate state.
            
            loss = np.count_nonzero(cover_count[r_old::m_old] == 0)
            
            if best_gain > loss:
                # Perform the swap
                current_classes[i] = [best_m_new, best_r_new]
                
                # Update cover_count
                # We already subtracted old class.
                cover_count[best_r_new::best_m_new] += 1
                
                # Update used_moduli
                used_moduli.remove(m_old)
                used_moduli.add(best_m_new)
                
                improved = True
                break # Restart the loop with the new state
        
    return current_classes

def construct():
    N = 51975
    divisors = _get_divisors(N)
    
    start_time = time.time()
    total_time_limit = 145.0  # Safety margin before 150s
    
    # Strategy 1: Greedy initialization
    classes, covered = _greedy_build(N, divisors, time_limit=5.0, start_time=start_time)
    
    # Strategy 2: Local search
    elapsed = time.time() - start_time
    if elapsed < total_time_limit - 5.0:
        local_time = total_time_limit - elapsed - 2.0
        classes_final = _local_search(classes, N, divisors, time_limit=local_time, start_time=start_time)
        classes = classes_final
    
    return classes
