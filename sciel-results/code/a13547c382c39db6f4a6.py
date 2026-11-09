"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 135
    prescribed = [1, 2, 5, 11]
    N = 284
    prescribed_set = set(prescribed)
    
    def is_valid_sidon(arr):
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    # Greedy construction: build a Sidon set greedily, then try to extend
    def greedy_construct(start_values, N, time_limit):
        """Greedy: add values one by one, choosing the smallest compatible value."""
        start_time = time.time()
        arr = list(start_values)
        used_values = set(arr)
        used_sums = set()
        for i in range(len(arr)):
            for j in range(i, len(arr)):
                used_sums.add(arr[i] + arr[j])
        
        # Precompute: for each candidate x, check compatibility with current set
        # x is compatible if:
        # 1. x not in used_values
        # 2. 2*x not in used_sums
        # 3. For all y in arr: x + y not in used_sums
        
        # We'll try to add as many as possible, up to target size
        target_k = 20
        while len(arr) < target_k:
            if time.time() - start_time > time_limit:
                break
            
            # Find smallest compatible x
            found = False
            for x in range(1, N + 1):
                if x in used_values:
                    continue
                if 2 * x in used_sums:
                    continue
                compatible = True
                for y in arr:
                    if x + y in used_sums:
                        compatible = False
                        break
                if compatible:
                    arr.append(x)
                    used_values.add(x)
                    for y in arr[:-1]:
                        used_sums.add(x + y)
                    used_sums.add(2 * x)
                    found = True
                    break
            
            if not found:
                break
        
        return arr
    
    # Simulated Annealing with difference-count tracking
    def sa_optimize(initial, target_k, time_budget, seed):
        """SA to find a valid Sidon set of exactly target_k elements."""
        start_time = time.time()
        rng = random.Random(seed)
        
        n = target_k
        arr = list(initial[:n])
        
        # Ensure prescribed elements are in the set
        # If any prescribed element is missing, force it in
        for p in prescribed:
            if p not in arr:
                # Replace a random non-prescribed element
                for i in range(n):
                    if arr[i] not in prescribed_set:
                        arr[i] = p
                        break
                else:
                    arr[-1] = p
        
        # Remove duplicates
        seen = set()
        unique_arr = []
        for x in arr:
            if x not in seen:
                seen.add(x)
                unique_arr.append(x)
        while len(unique_arr) < n:
            # Add random unused value
            while True:
                x = rng.randint(1, N)
                if x not in seen:
                    seen.add(x)
                    unique_arr.append(x)
                    break
        arr = unique_arr[:n]
        
        # Build difference counts
        max_diff = N
        diff_counts = [0] * (max_diff + 1)
        num_conflicts = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                diff_counts[d] += 1
                if diff_counts[d] > 1:
                    num_conflicts += 1
        
        # SA parameters
        T = 10.0
        T_min = 0.001
        cooling = 0.99995
        max_iters = 500000
        
        best_conflicts = num_conflicts
        best_arr = list(arr)
        
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
            if num_conflicts == 0:
                return list(arr)
            
            if num_conflicts < best_conflicts:
                best_conflicts = num_conflicts
                best_arr = list(arr)
            
            # Choose move type
            move_type = rng.random()
            if move_type < 0.6:
                # Reposition: change one non-prescribed element
                idx = rng.randint(4, n - 1)
                old_val = arr[idx]
                new_val = rng.randint(1, N)
                
                # Check uniqueness
                valid = True
                for i in range(n):
                    if i != idx and arr[i] == new_val:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Compute delta conflicts
                # Remove old_val's contributions
                delta = 0
                for j in range(n):
                    if j == idx:
                        continue
                    d_old = abs(arr[j] - old_val)
                    d_new = abs(arr[j] - new_val)
                    # Removing old_val's diff
                    diff_counts[d_old] -= 1
                    if diff_counts[d_old] == 1:
                        delta -= 1  # was a conflict, now resolved
                    elif diff_counts[d_old] == 0:
                        # was not a conflict (count was 1, now 0)
                        pass
                    # Adding new_val's diff
                    diff_counts[d_new] += 1
                    if diff_counts[d_new] == 2:
                        delta += 1  # new conflict
                    elif diff_counts[d_new] > 2:
                        delta += 1  # still a conflict, count increases
                    
                    # Wait, this logic is wrong. Let me redo it.
                    # The conflict count is the number of diffs with count > 1.
                    # When we change a value, we remove (n-1) old diffs and add (n-1) new diffs.
                    # For each old diff d_old:
                    #   Before: count = c_old
                    #   After removing: count = c_old - 1
                    #   If c_old == 2: was conflict (count>1), after: count=1, not conflict -> delta -= 1
                    #   If c_old > 2: was conflict, after: count=c_old-1 >= 2, still conflict -> delta unchanged
                    #   If c_old == 1: was not conflict, after: count=0, not conflict -> delta unchanged
                    # For each new diff d_new:
                    #   Before: count = c_new
                    #   After adding: count = c_new + 1
                    #   If c_new == 1: was not conflict, after: count=2, conflict -> delta += 1
                    #   If c_new >= 2: was conflict, after: count>=3, still conflict -> delta unchanged
                    #   If c_new == 0: was not conflict, after: count=1, not conflict -> delta unchanged
                    pass
                
                # Let me redo this properly
                # First, revert the changes I made above
                for j in range(n):
                    if j == idx:
                        continue
                    d_old = abs(arr[j] - old_val)
                    d_new = abs(arr[j] - new_val)
                    # Undo the above changes
                    diff_counts[d_new] -= 1
                    if diff_counts[d_new] == 1:
                        delta -= 1
                    elif diff_counts[d_new] > 1:
                        delta -= 1
                    diff_counts[d_old] += 1
                    if diff_counts[d_old] == 2:
                        delta += 1
                    elif diff_counts[d_old] > 2:
                        delta += 1
                
                # Now do it properly
                delta = 0
                for j in range(n):
                    if j == idx:
                        continue
                    d_old = abs(arr[j] - old_val)
                    # Remove old diff
                    c = diff_counts[d_old]
                    diff_counts[d_old] = c - 1
                    if c == 2:
                        delta -= 1
                    # Add new diff
                    d_new = abs(arr[j] - new_val)
                    c2 = diff_counts[d_new]
                    diff_counts[d_new] = c2 + 1
                    if c2 == 1:
                        delta += 1
                
                arr[idx] = new_val
                num_conflicts += delta
                
                if delta > 0 and rng.random() >= math.exp(-delta / T):
                    # Reject
                    arr[idx] = old_val
                    num_conflicts -= delta
                    # Revert diff_counts
                    for j in range(n):
                        if j == idx:
                            continue
                        d_old = abs(arr[j] - old_val)
                        d_new = abs(arr[j] - new_val)
                        diff_counts[d_new] -= 1
                        if diff_counts[d_new] == 1:
                            pass  # was 2, now 1, no conflict change
                        diff_counts[d_old] += 1
                        if diff_counts[d_old] == 2:
                            pass  # was 1, now 2, conflict created
                    # Actually, let me just rebuild from scratch for safety
                    # No, let me be more careful.
                    # When we rejected, we need to undo the diff_counts changes.
                    # We did: for each j, diff_counts[d_old] -= 1, diff_counts[d_new] += 1
                    # To undo: for each j, diff_counts[d_old] += 1, diff_counts[d_new] -= 1
                    for j in range(n):
                        if j == idx:
                            continue
                        d_old = abs(arr[j] - old_val)
                        d_new = abs(arr[j] - new_val)
                        diff_counts[d_old] += 1
                        diff_counts[d_new] -= 1
                    num_conflicts -= delta
                
                T *= cooling
                if T < T_min:
                    T = T_min
            
            elif move_type < 0.8:
                # Swap: swap two non-prescribed elements
                idx1 = rng.randint(4, n - 1)
                idx2 = rng.randint(4, n - 1)
                if idx1 == idx2:
                    continue
                v1 = arr[idx1]
                v2 = arr[idx2]
                if v1 == v2:
                    continue
                
                # Compute delta
                delta = 0
                for j in range(n):
                    if j == idx1 or j == idx2:
                        continue
                    # Remove v1-j and v2-j diffs, add v2-j and v1-j diffs
                    # Since |v1 - j| and |v2 - j| are symmetric, the multiset of diffs
                    # for other elements doesn't change! Only the v1-v2 diff changes.
                    # Actually no: the diff between idx1 and j changes from |v1 - arr[j]| to |v2 - arr[j]|
                    # and the diff between idx2 and j changes from |v2 - arr[j]| to |v1 - arr[j]|
                    # So the multiset of diffs involving other elements is the same.
                    # Only the diff between idx1 and idx2 stays the same (|v1 - v2|).
                    # So delta = 0 for swap of two non-prescribed elements!
                    pass
                
                # Wait, that's only true if we're swapping values, not positions.
                # If we swap arr[idx1] and arr[idx2], the set of values is the same,
                # so all pairwise differences are the same. Delta = 0.
                # So swap doesn't help with conflicts. It only helps with uniqueness.
                # Let's skip swap and use other moves.
                continue
            
            else:
                # Insert: replace a non-prescribed element with a random one
                idx = rng.randint(4, n - 1)
                old_val = arr[idx]
                new_val = rng.randint(1, N)
                
                valid = True
                for i in range(n):
                    if i != idx and arr[i] == new_val:
                        valid = False
                        break
                if not valid:
                    continue
                
                delta = 0
                for j in range(n):
                    if j == idx:
                        continue
                    d_old = abs(arr[j] - old_val)
                    d_new = abs(arr[j] - new_val)
                    c = diff_counts[d_old]
                    diff_counts[d_old] = c - 1
                    if c == 2:
                        delta -= 1
                    c2 = diff_counts[d_new]
                    diff_counts[d_new] = c2 + 1
                    if c2 == 1:
                        delta += 1
                
                arr[idx] = new_val
                num_conflicts += delta
                
                if delta > 0 and rng.random() >= math.exp(-delta / T):
                    arr[idx] = old_val
                    num_conflicts -= delta
                    for j in range(n):
                        if j == idx:
                            continue
                        d_old = abs(arr[j] - old_val)
                        d_new = abs(arr[j] - new_val)
                        diff_counts[d_old] += 1
                        diff_counts[d_new] -= 1
                
                T *= cooling
                if T < T_min:
                    T = T_min
        
        if best_conflicts == 0:
            return best_arr
        return None
    
    # Main search
    best_a = None
    best_size = len(prescribed)
    
    # Strategy 1: Greedy construction
    if time.time() < deadline - 60:
        greedy_result = greedy_construct(prescribed, N, 5)
        if greedy_result and is_valid_sidon(greedy_result):
            best_a = greedy_result
            best_size = len(greedy_result)
    
    # Strategy 2: SA with multiple restarts, trying to reach size 20
    target_sizes = [20, 19, 18]
    
    for target_k in target_sizes:
        if time.time() > deadline - 30:
            break
        if target_k <= best_size:
            continue
        
        remaining_time = deadline - time.time() - 20
        if remaining_time < 5:
            break
        
        # Try multiple SA restarts
        num_restarts = min(10, int(remaining_time / 5))
        found = False
        
        for restart in range(num_restarts):
            if time.time() > deadline - 25:
                break
            
            # Create initial solution: prescribed + random values
            init = list(prescribed)
            used = set(prescribed)
            for _ in range(target_k - len(prescribed)):
                # Try to add a compatible value
                added = False
                for x in range(1, N + 1):
                    if x in used:
                        continue
                    # Check compatibility
                    compatible = True
                    for y in init:
                        if x + y in {a + b for a in init for b in init if a <= b} or 2*x in {a + b for a in init for b in init if a <= b}:
                            # This is O(k^2), let me use a set
                            pass
                    # Let me just use a simpler check
                    sums = set()
                    for i in range(len(init)):
                        for j in range(i, len(init)):
                            sums.add(init[i] + init[j])
                    if 2 * x not in sums and all(x + y not in sums for y in init):
                        init.append(x)
                        used.add(x)
                        added = True
                        break
                if not added:
                    while True:
                        x = rng.randint(1, N) if 'rng' in dir() else random.randint(1, N)
                        if x not in used:
                            init.append(x)
                            used.add(x)
                            break
            
            budget_per = (deadline - time.time() - 25) / max(1, num_restarts - restart)
            result = sa_optimize(init, target_k, budget_per, seed=1000 + restart)
            if result is not None and len(result) == target_k:
                if is_valid_sidon(result):
                    best_a = result
                    best_size = target_k
                    found = True
                    break
        
        if found:
            break
    
    # If we didn't reach 20, try to improve what we have
    if best_a is None or best_size < 20:
        # Use the greedy result as a starting point for SA
        if greedy_result and time.time() < deadline - 15:
            target_k = max(len(greedy_result) + 1, 15)
            init = list(greedy_result)
            while len(init) < target_k:
                x = random.randint(1, N)
                if x not in init:
                    init.append(x)
            result = sa_optimize(init, target_k, 10, seed=9999)
            if result is not None and len(result) == target_k and is_valid_sidon(result):
                best_a = result
                best_size = target_k
    
    # Final verification
    if best_a is not None:
        ba = sorted(best_a)
        # Ensure prescribed elements are present
        for p in prescribed:
            if p not in ba:
                ba = sorted(list(set(ba + [p])))
        if is_valid_sidon(ba):
            return ba
        elif is_valid_sidon(list(prescribed)):
            return sorted(prescribed)
    
    return sorted(prescribed)
# EVOLVE-BLOCK-END
