import random
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit, leaving 5s margin for safety
    
    max_val = 4096
    
    best_set = []
    best_size = 0
    
    def make_sum_set(arr):
        """Compute all unordered triple sums (with repetition) for sorted array arr."""
        n = len(arr)
        if n == 0:
            return set()
        sums = set()
        a = np.array(arr)
        # (i, i, i)
        for i in range(n):
            sums.add(3 * arr[i])
        # (i, i, j) with i < j
        for i in range(n):
            for j in range(i + 1, n):
                sums.add(2 * arr[i] + arr[j])
                sums.add(arr[i] + 2 * arr[j])
        # (i, j, k) with i < j < k
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    sums.add(arr[i] + arr[j] + arr[k])
        return sums
    
    def can_add_vectorized(x, current_set, sum_set):
        """Check if x can be added to current_set without creating duplicate triple sums."""
        n = len(current_set)
        if n == 0:
            return True, {3 * x}
        
        arr = np.array(current_set)
        
        # Check (x, x, x)
        s = 3 * x
        if s in sum_set:
            return False, None
        new_sums = {s}
        
        # Check (x, x, y) for all y in current_set
        new_sums_1 = 2 * x + arr
        for s_val in new_sums_1:
            s_val = int(s_val)
            if s_val in sum_set or s_val in new_sums:
                return False, None
            new_sums.add(s_val)
        
        # Check (x, y, y) for all y in current_set
        new_sums_2 = x + 2 * arr
        for s_val in new_sums_2:
            s_val = int(s_val)
            if s_val in sum_set or s_val in new_sums:
                return False, None
            new_sums.add(s_val)
        
        # Check (x, y, z) for all y < z in current_set
        for i in range(n):
            for j in range(i + 1, n):
                s = x + current_set[i] + current_set[j]
                if s in sum_set or s in new_sums:
                    return False, None
                new_sums.add(s)
        
        return True, new_sums
    
    def local_search(current_set, sum_set, time_limit):
        """Try to improve the set by removing one element and adding a better one, or swapping."""
        nonlocal best_set, best_size
        n = len(current_set)
        if n == 0:
            return current_set, sum_set
        
        current_set = sorted(current_set)
        
        # Try each element for removal, then try to add new elements
        for idx in range(n):
            if time.time() >= time_limit:
                break
            
            removed = current_set[idx]
            # Rebuild sum_set without removed element
            remaining = current_set[:idx] + current_set[idx+1:]
            sum_set_rem = make_sum_set(remaining)
            
            # Try to add new elements in descending order
            improved = False
            for x in range(max_val, 0, -1):
                if time.time() >= time_limit:
                    break
                if x == removed:
                    continue
                if x in remaining:
                    continue
                ok, new_sums = can_add_vectorized(x, remaining, sum_set_rem)
                if ok:
                    new_set = sorted(remaining + [x])
                    new_sum_set = sum_set_rem | new_sums
                    # Try to add more elements
                    for x2 in range(max_val, 0, -1):
                        if time.time() >= time_limit:
                            break
                        if x2 in new_set:
                            continue
                        ok2, new_sums2 = can_add_vectorized(x2, new_set, new_sum_set)
                        if ok2:
                            new_set = sorted(new_set + [x2])
                            new_sum_set = new_sum_set | new_sums2
                    if len(new_set) > len(current_set):
                        current_set = new_set
                        sum_set = new_sum_set
                        improved = True
                        break
            if improved:
                break
        
        return current_set, sum_set
    
    # Strategy 1: Base-3 inspired construction
    # Use powers of a base to ensure unique representations
    # For B_3[1], we need all triple sums unique
    # Start with a sparse geometric sequence, then compress
    
    # Initial construction: use a geometric sequence with base ~3
    # 3^0, 3^1, 3^2, ... up to 4096
    # 3^0=1, 3^1=3, 3^2=9, 3^3=27, 3^4=81, 3^5=243, 3^6=729, 3^7=2187, 3^8=6561>4096
    # So we get 8 elements from pure base-3. Let's use a slightly smaller base to get more.
    
    # Try different bases
    best_initial = []
    best_initial_size = 0
    
    for base in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        current_set = []
        current_sum_set = set()
        
        # Generate candidate sequence: base^i for i = 0, 1, 2, ...
        val = 1
        while val <= max_val:
            ok, new_sums = can_add_vectorized(val, current_set, current_sum_set)
            if ok:
                current_set.append(val)
                current_sum_set.update(new_sums)
            val *= base
        
        if len(current_set) > best_initial_size:
            best_initial_size = len(current_set)
            best_initial = current_set[:]
    
    # Use the best initial set as starting point
    if best_initial:
        current_set = best_initial[:]
        current_sum_set = make_sum_set(current_set)
    else:
        current_set = []
        current_sum_set = set()
    
    # Now try greedy from large to small to add more elements
    for x in range(max_val, 0, -1):
        if time.time() >= deadline - 10:
            break
        if x in current_set:
            continue
        ok, new_sums = can_add_vectorized(x, current_set, current_sum_set)
        if ok:
            current_set.append(x)
            current_set.sort()
            current_sum_set.update(new_sums)
    
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set[:]
    
    # Local search to improve
    if time.time() < deadline - 5:
        current_set, current_sum_set = local_search(current_set, current_sum_set, deadline - 5)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Strategy 2: Pure greedy from large to small (different starting point)
    if time.time() < deadline - 5:
        current_set = []
        current_sum_set = set()
        for x in range(max_val, 0, -1):
            if time.time() >= deadline - 5:
                break
            ok, new_sums = can_add_vectorized(x, current_set, current_sum_set)
            if ok:
                current_set.append(x)
                current_sum_set.update(new_sums)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        
        # Local search on this set
        if time.time() < deadline - 3:
            current_set, current_sum_set = local_search(current_set, current_sum_set, deadline - 3)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
    
    # Strategy 3: Randomized greedy with different seeds
    if time.time() < deadline - 3:
        random.seed(12345)
        for trial in range(50):
            if time.time() >= deadline - 3:
                break
            
            # Create a randomized ordering with bias
            candidates = list(range(1, max_val + 1))
            
            if trial % 4 == 0:
                # Descending with small random perturbation
                for i in range(len(candidates) - 1, 0, -1):
                    j = random.randint(max(0, i - 20), i)
                    candidates[i], candidates[j] = candidates[j], candidates[i]
            elif trial % 4 == 1:
                # Size-biased random
                weights = np.array([c ** 1.5 for c in candidates])
                weights /= weights.sum()
                idxs = np.random.choice(len(candidates), size=len(candidates), replace=False, p=weights)
                candidates = [candidates[i] for i in idxs]
            elif trial % 4 == 2:
                # Geometric spacing random
                base = random.uniform(2.5, 5)
                candidates = sorted([int(base ** i) for i in range(int(np.log(max_val) / np.log(base)) + 1) if int(base ** i) <= max_val], reverse=True)
                # Add some random extras
                extras = random.sample(range(1, max_val + 1), min(100, max_val))
                candidates = list(set(candidates + extras))
                random.shuffle(candidates)
            else:
                # Pure random
                random.shuffle(candidates)
            
            current_set = []
            current_sum_set = set()
            
            for x in candidates:
                if time.time() >= deadline - 3:
                    break
                ok, new_sums = can_add_vectorized(x, current_set, current_sum_set)
                if ok:
                    current_set.append(x)
                    current_sum_set.update(new_sums)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
                if best_size >= 40:
                    break
    
    # Final local search on the best set found
    if time.time() < deadline - 2 and best_set:
        best_sum_set = make_sum_set(best_set)
        best_set, best_sum_set = local_search(best_set, best_sum_set, deadline - 2)
        if len(best_set) > best_size:
            best_size = len(best_set)
    
    return best_set
