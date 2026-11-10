"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin from 90s limit
    n = 512
    
    # Use NumPy for fast vectorized 4-AP checks
    in_set_np = np.zeros(n + 1, dtype=np.bool_)
    
    def can_add_np(x, in_set_np):
        """Check if x can be added to in_set_np without creating a 4-AP using NumPy."""
        # x as 4th term: x-3d, x-2d, x-d in set
        max_d = x // 3
        if max_d >= 1:
            ds = np.arange(1, max_d + 1)
            a = x - 3*ds
            b = x - 2*ds
            c = x - ds
            if np.any(in_set_np[a] & in_set_np[b] & in_set_np[c]):
                return False
        # x as 1st term: x+d, x+2d, x+3d
        max_d = (n - x) // 3
        if max_d >= 1:
            ds = np.arange(1, max_d + 1)
            b = x + ds
            c = x + 2*ds
            e = x + 3*ds
            if np.any(in_set_np[b] & in_set_np[c] & in_set_np[e]):
                return False
        # x as 2nd term: x-d, x+d, x+2d
        max_d = min(x, (n - x) // 2)
        if max_d >= 1:
            ds = np.arange(1, max_d + 1)
            a = x - ds
            b = x + ds
            c = x + 2*ds
            if np.any(in_set_np[a] & in_set_np[b] & in_set_np[c]):
                return False
        # x as 3rd term: x-2d, x-d, x+d
        max_d = min(x // 2, n - x)
        if max_d >= 1:
            ds = np.arange(1, max_d + 1)
            a = x - 2*ds
            b = x - ds
            c = x + ds
            if np.any(in_set_np[a] & in_set_np[b] & in_set_np[c]):
                return False
        return True

    def is_valid_4ap_free_np(in_set_np):
        """Check if the current set is 4-AP free using NumPy."""
        for a in range(1, n + 1):
            if not in_set_np[a]:
                continue
            max_d = (n - a) // 3
            if max_d < 1:
                continue
            ds = np.arange(1, max_d + 1)
            b = a + ds
            c = a + 2*ds
            e = a + 3*ds
            if np.any(in_set_np[b] & in_set_np[c] & in_set_np[e]):
                return False
        return True

    # 1. Initialize with a greedy construction to get a good starting point
    # Use base-3 digit sum ordering which tends to avoid small 3-APs, hence 4-APs
    def base3_sum(x):
        s = 0
        while x > 0:
            s += x % 3
            x //= 3
        return s

    order = sorted(range(1, n + 1), key=lambda x: (base3_sum(x), x))
    in_set_np = np.zeros(n + 1, dtype=np.bool_)
    current_set = []
    
    for x in order:
        if time.time() > deadline - 10:
            break
        if can_add_np(x, in_set_np):
            in_set_np[x] = True
            current_set.append(x)
    
    best_size = len(current_set)
    best_set = list(current_set)
    
    # 2. Local Search with Random Restarts (kick and recover)
    random.seed(42)
    current_set = list(best_set)
    in_set_np = np.zeros(n + 1, dtype=np.bool_)
    for x in current_set:
        in_set_np[x] = True
        
    # Increased restart count since can_add is now much faster
    for restart in range(200):
        if time.time() > deadline - 5:
            break
            
        # Create a perturbed set
        perturbed_set = list(current_set)
        perturbed_in_set = in_set_np.copy()
        
        # Remove 5-15 random elements
        num_remove = random.randint(5, 15)
        for _ in range(num_remove):
            if perturbed_set:
                idx = random.randint(0, len(perturbed_set) - 1)
                x = perturbed_set.pop(idx)
                perturbed_in_set[x] = False
        
        # Now try to add elements back and new elements
        # Combine removed and all missing elements
        missing = [x for x in range(1, n + 1) if not perturbed_in_set[x]]
        random.shuffle(missing)
        
        # Greedily add
        for x in missing:
            if time.time() > deadline - 5:
                break
            if can_add_np(x, perturbed_in_set):
                perturbed_in_set[x] = True
                perturbed_set.append(x)
        
        if len(perturbed_set) > best_size:
            best_size = len(perturbed_set)
            best_set = list(perturbed_set)
            current_set = list(perturbed_set)
            in_set_np = perturbed_in_set.copy()
            
    # 3. Final polish: try multiple different orderings to add remaining elements
    if time.time() < deadline - 3:
        polish_orders = [
            list(range(1, n + 1)),  # natural order
            list(range(n, 0, -1)),  # reverse order
            sorted(range(1, n + 1), key=lambda x: (base3_sum(x), x)),  # base-3
            sorted(range(1, n + 1), key=lambda x: (x % 7, x)),  # mod 7
            sorted(range(1, n + 1), key=lambda x: (x % 11, x)),  # mod 11
        ]
        
        for pol_order in polish_orders:
            if time.time() > deadline - 2:
                break
            # Try adding elements in this order to the best set
            test_in_set = in_set_np.copy()
            test_set = list(current_set)
            for x in pol_order:
                if test_in_set[x]:
                    continue
                if can_add_np(x, test_in_set):
                    test_in_set[x] = True
                    test_set.append(x)
            if len(test_set) > best_size:
                best_size = len(test_set)
                best_set = list(test_set)
                current_set = list(test_set)
                in_set_np = test_in_set.copy()
    
    # Final cleanup: ensure best_set is sorted and valid
    # Re-verify best_set
    final_in_set = np.zeros(n + 1, dtype=np.bool_)
    for x in best_set:
        final_in_set[x] = True
        
    # Quick check
    valid = is_valid_4ap_free_np(final_in_set)
    
    if not valid:
        # Fallback to a simple valid set if something went wrong (should not happen)
        fallback_in_set = np.zeros(n + 1, dtype=np.bool_)
        fallback_set = []
        for x in order:
            if can_add_np(x, fallback_in_set):
                fallback_in_set[x] = True
                fallback_set.append(x)
        return sorted(fallback_set)
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
