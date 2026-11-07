import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    best_size = 0
    best_set = []
    
    # Max possible sum for 30 elements of max 256
    MAX_SUM = 256 * 30
    
    def is_valid_addition_fast(current_sums_bool, x, max_sum):
        """
        Check if adding x to a set with subset sums is valid.
        current_sums_bool is a boolean array where current_sums_bool[s] = True if s is a subset sum.
        Returns True if valid, False otherwise.
        """
        # Generate new sums
        new_sums = set()
        for s in range(1, max_sum + 1):
            if current_sums_bool[s]:
                new_sums.add(s + x)
        new_sums.add(x)
        
        # Check new sums against existing sums and each other
        for ns in new_sums:
            # Check if ns is already an existing sum (equal sums forbidden)
            if current_sums_bool[ns]:
                return False
            
            # Check divisibility: does any existing sum divide ns?
            # Or does ns divide any existing sum?
            # We can check: for each divisor d of ns, if current_sums_bool[d] and d != ns, invalid
            # For each multiple m of ns, if current_sums_bool[m] and m != ns, invalid
            
            # Check if any existing sum divides ns
            d = 1
            while d * d <= ns:
                if ns % d == 0:
                    if d != ns and current_sums_bool[d]:
                        return False
                    other = ns // d
                    if other != d and other != ns and current_sums_bool[other]:
                        return False
                d += 1
            
            # Check if ns divides any existing sum
            # ns divides m if m = k * ns for some integer k >= 2
            k = 2
            while k * ns <= max_sum:
                m = k * ns
                if m != ns and current_sums_bool[m]:
                    return False
                k += 1
        
        # Check new sums against each other
        sorted_new = sorted(new_sums)
        for i in range(len(sorted_new)):
            ns_i = sorted_new[i]
            for j in range(i + 1, len(sorted_new)):
                ns_j = sorted_new[j]
                if ns_j % ns_i == 0:
                    return False
        
        return True

    def greedy_construct(order, max_sum):
        current_set = []
        current_sums_bool = [False] * (max_sum + 1)
        
        for x in order:
            if is_valid_addition_fast(current_sums_bool, x, max_sum):
                current_set.append(x)
                # Update sums
                new_sums = set()
                for s in range(1, max_sum + 1):
                    if current_sums_bool[s]:
                        new_sums.add(s + x)
                new_sums.add(x)
                for ns in new_sums:
                    current_sums_bool[ns] = True
                
        return current_set

    def local_search(current_set, max_sum, time_limit):
        """Try to improve by swapping elements."""
        current_set = list(current_set)
        current_sums_bool = [False] * (max_sum + 1)
        
        # Rebuild sums from scratch
        sums = set()
        for x in current_set:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            new_sums.add(x)
            sums.update(new_sums)
        for s in sums:
            current_sums_bool[s] = True
        
        candidates = list(range(1, 257))
        in_set = set(current_set)
        not_in_set = [c for c in candidates if c not in in_set]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to remove an element and add a better one
            for i, x in enumerate(current_set):
                # Try removing x
                test_set = current_set[:i] + current_set[i+1:]
                
                # Check if removing x is valid (it always is, since we're removing)
                # Now try adding each element not in the set
                for y in not_in_set:
                    if y in test_set:
                        continue
                    test_set_with_y = test_set + [y]
                    
                    # Check validity
                    sums_check = set()
                    valid = True
                    for z in test_set_with_y:
                        new_sums = set()
                        for s in sums_check:
                            new_sums.add(s + z)
                        new_sums.add(z)
                        
                        # Check for conflicts
                        for ns in new_sums:
                            if ns in sums_check:
                                valid = False
                                break
                            # Check divisibility with existing
                            d = 1
                            while d * d <= ns and valid:
                                if ns % d == 0:
                                    if d != ns and d in sums_check:
                                        valid = False
                                        break
                                    other = ns // d
                                    if other != d and other != ns and other in sums_check:
                                        valid = False
                                        break
                                d += 1
                            if not valid:
                                break
                            # Check if ns divides existing
                            k = 2
                            while k * ns <= max_sum and valid:
                                m = k * ns
                                if m != ns and m in sums_check:
                                    valid = False
                                    break
                                k += 1
                        
                        if valid:
                            sums_check.update(new_sums)
                    
                    if valid:
                        # Check new sums against each other
                        all_new = list(sums_check)
                        all_new.sort()
                        for a in range(len(all_new)):
                            for b in range(a + 1, len(all_new)):
                                if all_new[b] % all_new[a] == 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            current_set = test_set_with_y
                            improved = True
                            break
                if improved:
                    break
        
        return current_set

    # Run multiple restarts with different random orders
    candidates = list(range(1, 257))
    
    # Try descending order first
    best_set = greedy_construct(sorted(candidates, reverse=True), MAX_SUM)
    best_size = len(best_set)
    
    # Try ascending order
    s2 = greedy_construct(sorted(candidates), MAX_SUM)
    if len(s2) > best_size:
        best_size = len(s2)
        best_set = s2

    # Random restarts with local search
    while time.time() < deadline:
        random.shuffle(candidates)
        s = greedy_construct(candidates, MAX_SUM)
        
        # Apply local search if we have time
        if time.time() < deadline - 5:
            s = local_search(s, MAX_SUM, deadline - 2)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    return best_set
