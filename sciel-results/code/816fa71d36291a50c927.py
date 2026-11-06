import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def is_valid(arr):
        """
        Checks if arr is a valid Sidon set covering 1..30.
        Returns (is_valid, diffs_set, diameter).
        """
        if len(arr) > 32:
            return False, None, None
        if 0 not in arr:
            return False, None, None
        if max(arr) > 10**6:
            return False, None, None

        n = len(arr)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False, None, None
                diffs.add(d)

        # Check coverage 1..30
        for d in range(1, 31):
            if d not in diffs:
                return False, None, None
        return True, diffs, max(arr) - min(arr)

    def find_min_diameter_set(max_candidate=100):
        """
        Uses backtracking search with optimizations to find a Sidon set
        with minimal maximum element.
        """
        best_set = None
        
        # We need at least 9 elements (9*8/2 = 36 >= 30 differences)
        # Diameter must be at least 30
        for D in range(30, max_candidate + 1):
            if time.time() > deadline - 10:
                break
            
            found = None
            
            def backtrack(current_elements, all_diffs, covered, start_idx):
                nonlocal found
                if found is not None:
                    return
                
                if time.time() > deadline - 10:
                    return
                
                # Goal check
                if len(covered) == 30:
                    found = current_elements[:]
                    return
                
                # Pruning: Can we cover the remaining differences?
                m = len(current_elements)
                k = D - start_idx + 1  # remaining candidates
                missing = 30 - len(covered)
                
                if k > 0:
                    max_total_elems = m + k
                    max_total_diffs = max_total_elems * (max_total_elems - 1) // 2
                    current_diffs_count = m * (m - 1) // 2
                    max_new_diffs = max_total_diffs - current_diffs_count
                    if max_new_diffs < missing:
                        return
                else:
                    if missing > 0:
                        return
                
                # Generate candidates and order by coverage potential
                candidates = []
                for c in range(start_idx, D + 1):
                    # Check if adding c creates duplicate differences
                    ok = True
                    new_diffs = []
                    for x in current_elements:
                        d = c - x
                        if d in all_diffs:
                            ok = False
                            break
                        new_diffs.append(d)
                    
                    if not ok:
                        continue
                    
                    # Count how many new required diffs this covers
                    new_cover_count = 0
                    for d in new_diffs:
                        if 1 <= d <= 30 and d not in covered:
                            new_cover_count += 1
                    
                    candidates.append((c, new_cover_count, new_diffs))
                
                # Sort by coverage potential (descending), then by value (ascending)
                candidates.sort(key=lambda x: (-x[1], x[0]))
                
                for c, new_cover_count, new_diffs in candidates:
                    new_covered = covered.copy()
                    for d in new_diffs:
                        if 1 <= d <= 30:
                            new_covered.add(d)
                    
                    new_all_diffs = all_diffs.union(new_diffs)
                    new_elements = current_elements + [c]
                    
                    backtrack(new_elements, new_all_diffs, new_covered, c + 1)
                    
                    if found is not None:
                        return
            
            # Initial call
            backtrack([0], set(), set(), 1)
            
            if found is not None:
                return found
                
        return None

    def local_search_refine(initial_set, time_limit):
        """
        Try to reduce the maximum element through local search.
        """
        if initial_set is None:
            return None
        
        best_set = initial_set[:]
        best_max = max(best_set)
        
        def try_remove_and_replace(arr, time_limit):
            """Try removing an element and replacing with a smaller one."""
            for i in range(len(arr) - 1, 0, -1):  # Try removing largest first
                if time.time() > time_limit:
                    break
                
                removed = arr[i]
                arr_without = arr[:i] + arr[i+1:]
                
                # Try to find a smaller replacement
                for new_val in range(removed - 1, 0, -1):
                    if new_val in arr_without:
                        continue
                    
                    # Check if adding new_val keeps Sidon property
                    valid = True
                    for x in arr_without:
                        d = abs(new_val - x)
                        if d == 0:
                            continue
                        # Check if this difference already exists
                        for j in range(len(arr_without)):
                            for k in range(j + 1, len(arr_without)):
                                if abs(arr_without[j] - arr_without[k]) == d:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if not valid:
                        continue
                    
                    # Check coverage
                    new_arr = sorted(arr_without + [new_val])
                    diffs = set()
                    for j in range(len(new_arr)):
                        for k in range(j + 1, len(new_arr)):
                            diffs.add(new_arr[k] - new_arr[j])
                    
                    if all(d in diffs for d in range(1, 31)):
                        return new_arr
            
            return None
        
        # Simple local search: try to swap elements to reduce max
        for _ in range(100):  # Limited iterations
            if time.time() > time_limit:
                break
            
            improved = False
            for i in range(len(best_set)):
                if time.time() > time_limit:
                    break
                
                old_val = best_set[i]
                # Try smaller values
                for new_val in range(old_val - 1, max(0, old_val - 20), -1):
                    if new_val in best_set:
                        continue
                    
                    test_set = best_set[:]
                    test_set[i] = new_val
                    test_set.sort()
                    
                    # Validate
                    diffs = set()
                    valid = True
                    for j in range(len(test_set)):
                        for k in range(j + 1, len(test_set)):
                            d = test_set[k] - test_set[j]
                            if d in diffs:
                                valid = False
                                break
                            diffs.add(d)
                        if not valid:
                            break
                    
                    if not valid:
                        continue
                    
                    if all(d in diffs for d in range(1, 31)):
                        best_set = test_set
                        best_max = max(best_set)
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return best_set

    # Main Execution
    # 1. Try exact search for minimal diameter
    best_set = find_min_diameter_set(max_candidate=80)
    
    # 2. Try to refine with local search
    if best_set is not None and time.time() < deadline - 20:
        refined = local_search_refine(best_set, deadline - 5)
        if refined is not None and max(refined) <= max(best_set):
            best_set = refined
    
    # Final Validation
    if best_set is None:
        # Emergency: use a known valid set
        emergency = [0, 1, 4, 10, 12, 28, 32, 45, 63]
        v, _, _ = is_valid(emergency)
        if v:
            return emergency
        else:
            return [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]

    return best_set

# EVOLVE-BLOCK-END
