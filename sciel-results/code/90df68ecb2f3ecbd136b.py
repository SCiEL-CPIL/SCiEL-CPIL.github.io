import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 88  # Safety margin
    
    N = 128
    TARGET_START = 128
    TARGET_END = 256
    TARGET_RANGE = range(TARGET_START, TARGET_END + 1)
    TARGETS = list(TARGET_RANGE)
    
    # Precompute valid pairs for targets? 
    # Or just use the fast mult array.
    
    def compute_max_mult(mults):
        return max(mults[TARGET_START:TARGET_END + 1])
    
    def is_covered(mults):
        # Check if any target has 0 representations
        # mults[t] == 0 for any t in TARGETS
        # Using numpy for speed
        return np.all(mults[TARGET_START:TARGET_END + 1] > 0)

    def add_element(A_list, A_set, mults, x):
        """Add x to A. Update mults."""
        A_list.append(x)
        A_set.add(x)
        # Self pair
        mults[2 * x] += 1
        # Pairs with existing
        for y in A_list[:-1]: # Exclude x itself from the loop as handled above
            mults[x + y] += 1
        return mults

    def remove_element(A_list, A_set, mults, x):
        """Remove x from A. Update mults."""
        A_set.remove(x)
        # Find index
        idx = A_list.index(x)
        A_list.pop(idx)
        
        # Self pair
        mults[2 * x] -= 1
        # Pairs with remaining
        for y in A_list:
            mults[x + y] -= 1
        return mults

    def get_max_mult_and_targets(mults):
        target_mults = mults[TARGET_START:TARGET_END + 1]
        max_m = np.max(target_mults)
        # Get indices where max occurs
        max_indices = np.where(target_mults == max_m)[0] + TARGET_START
        return int(max_m), max_indices

    def local_search(A_list, A_set, mults, max_time=10.0):
        """
        Local search to minimize max multiplicity.
        Returns A_list, A_set, mults, max_mult
        """
        current_max, _ = get_max_mult_and_targets(mults)
        
        # If not covered, this is a repair phase? 
        # The caller should ensure coverage or this function should handle it.
        # Let's assume caller provides covered set.
        if not is_covered(mults):
             # Fallback: return as is, caller handles repair
             return A_list, A_set, mults, float('inf')

        time_limit = time.time() + max_time
        no_improve_count = 0
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try Removals
            # Iterate in random order to avoid bias
            candidates = A_list[:]
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > time_limit: break
                
                # Simulate removal
                # We need to check if coverage is maintained.
                # Removing x reduces mults[t] by 1 for each t such that x + y = t, y in A\{x}.
                # And mults[2x] by 1.
                
                # Efficient check:
                # We can just do the removal, check, and revert if bad.
                # Since update is O(|A|), and |A| is small (~30), this is fast.
                
                # Save state? No, just do it and undo.
                # But we need to know if coverage breaks.
                
                # Let's do the removal
                remove_element(A_list, A_set, mults, x)
                
                if is_covered(mults):
                    new_max, _ = get_max_mult_and_targets(mults)
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
                        no_improve_count = 0
                        break
                    elif new_max == current_max:
                        # Accept sideways move?
                        # Only if we are stuck?
                        if no_improve_count > 50:
                            current_max = new_max
                            improved = True
                            no_improve_count = 0
                            break
                        else:
                            # Revert
                            add_element(A_list, A_set, mults, x)
                else:
                    # Revert
                    add_element(A_list, A_set, mults, x)
            
            if improved:
                continue
            
            # 2. Try Swaps
            # Pick a random x in A to remove, and a random z not in A to add.
            # Or structured: Pick x from high-multiplicity targets?
            
            # Let's try random swaps first
            if len(A_list) > 0:
                x = random.choice(A_list)
                # Sample z
                z = random.randint(0, N)
                if z in A_set:
                    continue
                
                # Simulate swap: Remove x, Add z
                remove_element(A_list, A_set, mults, x)
                add_element(A_list, A_set, mults, z)
                
                if is_covered(mults):
                    new_max, _ = get_max_mult_and_targets(mults)
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
                        no_improve_count = 0
                    elif new_max == current_max:
                        if no_improve_count > 20:
                            current_max = new_max
                            improved = True
                            no_improve_count = 0
                        else:
                            # Revert
                            remove_element(A_list, A_set, mults, z)
                            add_element(A_list, A_set, mults, x)
                else:
                    # Revert
                    remove_element(A_list, A_set, mults, z)
                    add_element(A_list, A_set, mults, x)
            
            if not improved:
                no_improve_count += 1
                # If stuck for a long time, try a more aggressive random restart of local search state?
                # Or just break to allow outer loop to restart.
                if no_improve_count > 100:
                    break
                    
        return A_list, A_set, mults, current_max

    best_A_list = None
    best_mult = float('inf')
    
    # Generate multiple seeds
    # Sizes: 20 to 35
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
            
        # Choose size
        # Smaller sets have higher average mult but might have lower max if well-distributed?
        # Larger sets have more pairs, potentially higher max.
        # Optimal size for K=3 is around 25-30.
        k = random.randint(22, 32)
        
        # Random initial set
        A_init = set(random.sample(range(0, N + 1), k))
        A_list = list(A_init)
        
        # Initialize mults
        mults = np.zeros(257, dtype=int)
        for x in A_list:
            # Add logic manually for init
            mults[2 * x] += 1
            for y in A_list:
                if y < x:
                    mults[x + y] += 1
        # Wait, the add_element function above assumes A_list does NOT contain x.
        # For init, let's just build it properly.
        mults = np.zeros(257, dtype=int)
        A_list_init = list(A_init)
        for i in range(len(A_list_init)):
            x = A_list_init[i]
            mults[2 * x] += 1
            for j in range(i):
                y = A_list_init[j]
                mults[x + y] += 1
        
        # Check coverage
        if not is_covered(mults):
            # Repair: Add elements to cover missing targets
            # Simple greedy repair
            changed = True
            while changed and time.time() < deadline - 2:
                changed = False
                # Find missing targets
                missing = [t for t in TARGETS if mults[t] == 0]
                if not missing:
                    break
                
                # Pick a missing target
                t = missing[0]
                
                # Find x, y such that x+y=t, x,y in [0,128]
                # We want to add elements that cover t and ideally other missing targets.
                # Try adding a single element x such that t-x is in A.
                best_x = None
                best_gain = 0
                for x in range(0, N + 1):
                    if x in A_init:
                        continue
                    y = t - x
                    if 0 <= y <= N:
                        # If y is in A, adding x covers t.
                        if y in A_init:
                            # Calculate gain: how many other missing targets does x cover?
                            gain = 1
                            for t2 in missing[1:]:
                                y2 = t2 - x
                                if 0 <= y2 <= N and y2 in A_init:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                                best_y = y
                            
                if best_x is not None:
                    # Add best_x
                    A_init.add(best_x)
                    # Update mults
                    mults[2 * best_x] += 1
                    for y in A_list_init: # Note: A_list_init is stale, use A_init? 
                        # It's easier to just rebuild mults if we do this often, 
                        # or update carefully.
                        # Let's just rebuild for simplicity in repair phase, it's short.
                        pass
                    # Actually, let's just use the add_element function logic but manually.
                    # Rebuild mults from scratch is safer for repair.
                    mults = np.zeros(257, dtype=int)
                    A_list_rebuild = list(A_init)
                    for i in range(len(A_list_rebuild)):
                        x = A_list_rebuild[i]
                        mults[2 * x] += 1
                        for j in range(i):
                            y = A_list_rebuild[j]
                            mults[x + y] += 1
                    A_list = A_list_rebuild
                    changed = True
                else:
                    # No single element covers t. Must add a pair?
                    # Or add x and y?
                    # If we add x and y, we add 2 elements.
                    # Let's just pick x = t//2, y = t-x.
                    x = t // 2
                    y = t - x
                    A_init.add(x)
                    A_init.add(y)
                    # Rebuild mults
                    mults = np.zeros(257, dtype=int)
                    A_list_rebuild = list(A_init)
                    for i in range(len(A_list_rebuild)):
                        x = A_list_rebuild[i]
                        mults[2 * x] += 1
                        for j in range(i):
                            y = A_list_rebuild[j]
                            mults[x + y] += 1
                    A_list = A_list_rebuild
                    changed = True
            
            if not is_covered(mults):
                continue
        
        # Now we have a covered set. Run local search.
        # Allocate time based on remaining budget
        time_left = deadline - time.time()
        search_time = min(5.0, time_left * 0.5) # Don't spend all time on one
        
        A_list, A_set, mults, current_max = local_search(A_list, A_init, mults, max_time=search_time)
        
        if current_max < best_mult:
            best_mult = current_max
            best_A_list = A_list[:]
            
        # If we reached 3, we can stop or keep trying for 2? 
        # K=2 is likely impossible for 129 targets with small A.
        # 129 targets, max mult 2 => total pairs >= 129.
        # k(k+1)/2 >= 129 => k >= 16.
        # But uniformity is hard.
        # If best_mult == 3, we are very good.
        if best_mult == 3:
             # Try to find 2? Unlikely.
             # Keep searching for a while?
             pass

    # Fallback
    if best_A_list is None:
        # Use a known good structure?
        # Or just return a random covered set?
        # Let's construct a simple one: A = {0, 1, ..., 15} U {113, ..., 128}?
        # Just use the first successful one from the loop if we stored it?
        # The loop always updates best_A_list if it finds a covered set.
        # If no covered set found, we need a fallback.
        # Force a covered set: A = range(0, 129) is covered.
        best_A_list = list(range(0, 129))
        # This has high mult, but it's valid.
        
    return best_A_list
