"""Search-based solver for Sidon set extension containing {1, 2, 5, 11} in [1, 284]."""
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s budget, 15s safety margin before 150s kill

    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    # Pre-compute initial sums
    initial_sums = set()
    for i, x in enumerate(prescribed):
        for y in prescribed[i:]:
            initial_sums.add(x + y)
    
    # Best found so far
    best_set = set(prescribed)
    best_size = len(best_set)
    best_sums = initial_sums
    
    def build_sums(s):
        """Build the set of all pairwise sums a+b for a<=b in s."""
        sl = sorted(s)
        sums = set()
        for i, x in enumerate(sl):
            for y in sl[i:]:
                sums.add(x + y)
        return sums

    # Helper to check if adding x to a set with given sums is valid
    def can_add(x, current_set, current_sums):
        new_sums = {x + y for y in current_set} | {2 * x}
        if len(new_sums) == len(current_set) + 1 and not (new_sums & current_sums):
            return True
        return False

    def get_new_sums(x, current_set, current_sums):
        return {x + y for y in current_set} | {2 * x}

    # Strategy 1: Greedy forward construction with randomization
    remaining = [x for x in range(12, 285) if x not in prescribed]
    
    def greedy_with_random_order(order, time_limit=5.0):
        nonlocal best_set, best_size, best_sums
        current_set = set(prescribed)
        current_sums = set(initial_sums)
        start = time.time()
        
        for x in order:
            if time.time() - start > time_limit:
                break
            if x in current_set:
                continue
            if can_add(x, current_set, current_sums):
                new_sums = get_new_sums(x, current_set, current_sums)
                current_set.add(x)
                current_sums.update(new_sums)
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            best_sums = set(current_sums)
        
        return len(current_set)

    # Try quick greedy passes with different random seeds
    rng = random.Random(42)
    for trial in range(200):
        if time.time() > deadline - 60:
            break
        order = remaining[:]
        rng.shuffle(order)
        greedy_with_random_order(order, time_limit=0.3)
    
    # Strategy 2: Efficient local search with incremental sum tracking
    def efficient_local_search(start_set, time_limit=80.0):
        nonlocal best_set, best_size, best_sums
        current_set = set(start_set)
        current_sums = build_sums(current_set)
        start = time.time()
        all_candidates = list(range(1, 285))
        
        # Track which sums are produced by which pairs for efficient removal
        # sum_to_pairs: maps each sum to the set of pairs (a,b) that produce it
        # For efficiency, we track: for each element x, which sums would be lost if x removed
        # sums_by_element[x] = set of sums involving x
        
        def compute_sums_by_element(s, sums):
            """For each element x in s, compute the set of sums that involve x."""
            sle = {}
            sl = sorted(s)
            for x in s:
                sle[x] = set()
            for i, x in enumerate(sl):
                for y in sl[i:]:
                    s_val = x + y
                    sle[x].add(s_val)
                    if y != x:
                        sle[y].add(s_val)
            return sle
        
        sums_by_elem = compute_sums_by_element(current_set, current_sums)
        
        for iteration in range(1000000):
            if time.time() - start > time_limit or time.time() > deadline - 5:
                break
            
            # Try to add a random element
            x = rng.choice(all_candidates)
            if x not in current_set:
                if can_add(x, current_set, current_sums):
                    new_sums = get_new_sums(x, current_set, current_sums)
                    current_set.add(x)
                    current_sums.update(new_sums)
                    sums_by_elem[x] = set(new_sums)
                    for y in current_set:
                        if y != x:
                            sums_by_elem[y].add(x + y)
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(current_set)
                        best_sums = set(current_sums)
                    continue
            
            # Try swap: remove one non-prescribed, add one new
            removable = [x for x in current_set if x not in prescribed_set]
            if not removable:
                continue
            x_remove = rng.choice(removable)
            
            # Efficiently compute reduced sums
            # Sums that disappear when removing x_remove:
            # sums that are ONLY produced by pairs involving x_remove
            # A sum s disappears iff it was only produced by (x_remove, y) or (x_remove, x_remove)
            # We need to check: for each sum in sums_by_elem[x_remove], is it also produced by another pair?
            
            # Build a count of how many times each sum is produced
            # But that's expensive. Instead, let's just rebuild for the reduced set
            # but only for the sums we need to check
            
            remaining_set = current_set - {x_remove}
            
            # Quick check: can we find a valid x_add without full rebuild?
            # For x_add to be valid on remaining_set:
            # 1. x_add + y not in reduced_sums for all y in remaining_set
            # 2. 2*x_add not in reduced_sums
            # 3. x_add + y != x_add + z for y != z (always true)
            
            # reduced_sums = current_sums minus sums that are ONLY from x_remove pairs
            # A sum s is in reduced_sums iff there exists a pair (a,b) in remaining_set with a+b=s
            
            # Efficient approach: compute which sums are "exclusive" to x_remove
            # sum is exclusive to x_remove if the only pair producing it involves x_remove
            
            # For speed, let's just do a targeted rebuild
            reduced_sums = set()
            sl_rem = sorted(remaining_set)
            for i, a in enumerate(sl_rem):
                for b in sl_rem[i:]:
                    reduced_sums.add(a + b)
            
            # Find valid additions
            valid_adds = []
            for x_add in all_candidates:
                if x_add in current_set:
                    continue
                # Check if x_add can be added to remaining_set
                # New sums: x_add + y for y in remaining_set, plus 2*x_add
                # These must not conflict with reduced_sums and must be distinct among themselves
                # x_add + y1 == x_add + y2 iff y1 == y2, so distinctness is automatic
                # 2*x_add == x_add + y iff x_add == y, so need x_add not in remaining_set (already checked)
                
                conflict = False
                for y in remaining_set:
                    if (x_add + y) in reduced_sums:
                        conflict = True
                        break
                if not conflict and (2 * x_add) in reduced_sums:
                    conflict = True
                if not conflict:
                    valid_adds.append(x_add)
            
            if valid_adds:
                x_add = rng.choice(valid_adds)
                new_sums_test = {x_add + y for y in remaining_set} | {2 * x_add}
                current_set = remaining_set | {x_add}
                current_sums = reduced_sums | new_sums_test
                sums_by_elem = compute_sums_by_element(current_set, current_sums)
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
                    best_sums = set(current_sums)
        
        return len(current_set)

    # Run efficient local search from best found
    if best_size > len(prescribed):
        efficient_local_search(best_set, time_limit=70.0)
    
    # Strategy 3: Systematic greedy strategies
    if time.time() < deadline - 20:
        current_set = set(prescribed)
        current_sums = set(initial_sums)
        for x in range(12, 285):
            if x in current_set:
                continue
            if can_add(x, current_set, current_sums):
                new_sums = get_new_sums(x, current_set, current_sums)
                current_set.add(x)
                current_sums.update(new_sums)
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            best_sums = set(current_sums)
    
    # Greedy that picks the largest valid element each time
    if time.time() < deadline - 20:
        current_set = set(prescribed)
        current_sums = set(initial_sums)
        for x in range(284, 11, -1):
            if x in current_set:
                continue
            if can_add(x, current_set, current_sums):
                new_sums = get_new_sums(x, current_set, current_sums)
                current_set.add(x)
                current_sums.update(new_sums)
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            best_sums = set(current_sums)
    
    # Strategy 4: Greedy with lookahead
    if time.time() < deadline - 20:
        def greedy_with_lookahead(time_limit=5.0):
            nonlocal best_set, best_size, best_sums
            current_set = set(prescribed)
            current_sums = set(initial_sums)
            start = time.time()
            
            while time.time() - start < time_limit:
                valid_adds = []
                for x in range(1, 285):
                    if x not in current_set and can_add(x, current_set, current_sums):
                        valid_adds.append(x)
                
                if not valid_adds:
                    break
                
                x = rng.choice(valid_adds)
                new_sums = get_new_sums(x, current_set, current_sums)
                current_set.add(x)
                current_sums.update(new_sums)
            
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
                best_sums = set(current_sums)
            return len(current_set)
        
        for trial in range(80):
            if time.time() > deadline - 20:
                break
            greedy_with_lookahead(time_limit=1.5)
    
    # More randomized greedy passes
    if time.time() < deadline - 15:
        for trial in range(300):
            if time.time() > deadline - 15:
                break
            order = remaining[:]
            rng.shuffle(order)
            greedy_with_random_order(order, time_limit=0.5)
    
    # Final local search
    if time.time() < deadline - 5 and best_size > len(prescribed):
        efficient_local_search(best_set, time_limit=40.0)
    
    # Final check: ensure we return a valid set
    result = sorted(best_set)
    
    # Verify validity
    sums = set()
    valid = True
    for i, x in enumerate(result):
        for y in result[i:]:
            s = x + y
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    
    if not valid:
        return prescribed
    
    return result

# For testing
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Result: {result}")
    sums = set()
    valid = True
    for i, x in enumerate(result):
        for y in result[i:]:
            s = x + y
            if s in sums:
                valid = False
                break
            sums.add(s)
    print(f"Valid: {valid}")
