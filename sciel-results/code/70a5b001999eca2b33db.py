import random
import math
import time

def count_ap_involving(set_s, x):
    """Count how many 4-term APs are formed in set_s U {x} that include x."""
    count = 0
    s = set_s  # We don't need to add x to s for checking others in s
    
    # Case 1: x is the 4th term (a, a+d, a+2d, x) => a+3d = x => a = x-3d
    # a, a+d, a+2d must be in s
    for d in range(1, (x // 3) + 1):
        a = x - 3 * d
        if a >= 1:
            if a in s and (a + d) in s and (a + 2 * d) in s:
                count += 1

    # Case 2: x is the 3rd term (a, a+d, x, a+3d) => a+2d = x => a = x-2d
    # a, a+d, a+3d must be in s
    for d in range(1, (x // 2) + 1):
        a = x - 2 * d
        if a >= 1:
            b3 = a + 3 * d
            if b3 <= 512:
                if a in s and (a + d) in s and b3 in s:
                    count += 1

    # Case 3: x is the 2nd term (a, x, a+2d, a+3d) => a+d = x => a = x-d
    # a, a+2d, a+3d must be in s
    for d in range(1, x):
        a = x - d
        if a >= 1:
            b2 = a + 2 * d
            b3 = a + 3 * d
            if b3 <= 512:
                if a in s and b2 in s and b3 in s:
                    count += 1

    # Case 4: x is the 1st term (x, x+d, x+2d, x+3d)
    # x+d, x+2d, x+3d must be in s
    for d in range(1, (512 - x) // 3 + 1):
        b1 = x + d
        b2 = x + 2 * d
        b3 = x + 3 * d
        if b3 <= 512:
            if b1 in s and b2 in s and b3 in s:
                count += 1

    return count

def construct():
    n = 512
    # Start with base-3 0/1 set
    current_set = set()
    for i in range(1, n + 1):
        num = i
        ok = True
        while num > 0:
            if num % 3 == 2:
                ok = False
                break
            num //= 3
        if ok:
            current_set.add(i)
            
    # Calculate initial AP count (should be 0)
    # We don't strictly need to count all APs, just track changes.
    # But for safety, let's assume it's 0.
    current_ap_count = 0
    current_size = len(current_set)
    
    best_set = set(current_set)
    best_size = current_size
    
    # Simulated Annealing parameters
    T = 5.0
    cooling_rate = 0.999
    min_T = 0.01
    penalty_M = 1000.0
    
    start_time = time.time()
    max_time = 55.0 # Leave 5s for cleanup
    
    while time.time() - start_time < max_time:
        if T < min_T:
            break
            
        # Choose a move: Add, Remove, or Swap
        move_type = random.random()
        
        if move_type < 0.4:
            # Add a random element not in set
            candidates = [x for x in range(1, n + 1) if x not in current_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            
            # Calculate delta
            new_aps = count_ap_involving(current_set, x)
            delta_ap = new_aps
            delta_size = 1
            delta_cost = -delta_size + penalty_M * delta_ap
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                current_set.add(x)
                current_ap_count += delta_ap
                current_size += delta_size
                
                if current_ap_count == 0 and current_size > best_size:
                    best_set = set(current_set)
                    best_size = current_size
                    
        elif move_type < 0.8:
            # Remove a random element from set
            if not current_set:
                continue
            x = random.choice(list(current_set))
            
            # Calculate delta
            # Removing x breaks all APs containing x
            broken_aps = count_ap_involving(current_set, x)
            delta_ap = -broken_aps
            delta_size = -1
            delta_cost = -delta_size + penalty_M * delta_ap
            # Note: -delta_size = 1. penalty_M * (-broken_aps) = -penalty_M * broken_aps
            # So delta_cost = 1 - penalty_M * broken_aps
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                current_set.remove(x)
                current_ap_count += delta_ap
                current_size += delta_size
                
                if current_ap_count == 0 and current_size > best_size:
                    best_set = set(current_set)
                    best_size = current_size
                    
        else:
            # Swap: Remove x, Add y
            if not current_set:
                continue
            x = random.choice(list(current_set))
            candidates = [y for y in range(1, n + 1) if y not in current_set]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            # This is complex to calculate delta exactly without intermediate state
            # Simulate:
            # 1. Remove x: breaks APs containing x
            # 2. Add y: creates APs containing y in (S - {x})
            
            # Approximation: Just try the move and evaluate if beneficial?
            # For efficiency, let's skip Swap in SA for now or implement carefully.
            # Let's implement carefully:
            
            # Count APs broken by removing x from S
            # These are APs in S that contain x.
            broken_by_x = count_ap_involving(current_set, x)
            
            # Count APs created by adding y to S' = S \ {x}
            # These are APs in S' that contain y.
            # We need to check if y forms an AP with elements in S \ {x}.
            # count_ap_involving(S', y)
            # We can compute this by calling count_ap_involving on a set without x.
            # To avoid creating a new set, we can modify count_ap_involving to accept an 'exclude' element?
            # Or just create a temporary set. n=512, set creation is fast.
            temp_set = current_set - {x}
            created_by_y = count_ap_involving(temp_set, y)
            
            # Net change in AP count:
            # Old APs: Some contain x (broken), some don't.
            # New APs: Some contain y (created), some don't.
            # APs not containing x or y are unchanged.
            # APs containing both x and y? 
            # If an AP contains both x and y, it is broken by removing x, and it is NOT created by adding y 
            # (because x is not there to form it). 
            # Wait, if AP = {x, y, z, w}, removing x breaks it. Adding y to S\x does NOT create it because x is missing.
            # So, Broken = count of APs in S containing x.
            # Created = count of APs in S\x containing y.
            # Are there APs that are broken AND created? No, because created ones are in S\x, so they don't contain x.
            # So delta_ap = created_by_y - broken_by_x.
            
            delta_ap = created_by_y - broken_by_x
            delta_size = 0
            delta_cost = penalty_M * delta_ap
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                current_set.remove(x)
                current_set.add(y)
                current_ap_count += delta_ap
                # current_size unchanged
                
                if current_ap_count == 0 and current_size > best_size:
                    best_set = set(current_set)
                    best_size = current_size

        T *= cooling_rate

    # Final cleanup: ensure best_set is valid (it is, by tracking)
    return sorted(best_set)
