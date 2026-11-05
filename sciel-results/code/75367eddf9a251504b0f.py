import time
import random

def construct():
    n = 256
    # Precompute 3-AP contributions for each x (1-indexed in logic, 0-indexed in bitmask)
    # mask bit i corresponds to element i+1
    three_aps = [[] for _ in range(n)]
    for x in range(1, n+1):
        idx = x - 1
        for d in range(1, n+1):
            # x is middle: x-d, x+d
            if x-d >= 1 and x+d <= n:
                m = (1 << (x-d-1)) | (1 << (x+d-1))
                three_aps[idx].append(m)
            # x is end: x-d, x-2d
            if x-d >= 1 and x-2*d >= 1:
                m = (1 << (x-d-1)) | (1 << (x-2*d-1))
                three_aps[idx].append(m)
            # x is start: x+d, x+2d
            if x+d <= n and x+2*d <= n:
                m = (1 << (x+d-1)) | (1 << (x+2*d-1))
                three_aps[idx].append(m)
        
    four_ap_conflicts = [[] for _ in range(n)]
    for x in range(1, n+1):
        idx = x - 1
        for d in range(1, n+1):
            # x is 1st: x+d, x+2d, x+3d
            if x+3*d <= n:
                m = (1 << (x+d-1)) | (1 << (x+2*d-1)) | (1 << (x+3*d-1))
                four_ap_conflicts[idx].append(m)
            # x is 2nd: x-d, x+d, x+2d
            if x-d >= 1 and x+2*d <= n:
                m = (1 << (x-d-1)) | (1 << (x+d-1)) | (1 << (x+2*d-1))
                four_ap_conflicts[idx].append(m)
            # x is 3rd: x-2d, x-d, x+d
            if x-2*d >= 1 and x+d <= n:
                m = (1 << (x-2*d-1)) | (1 << (x-d-1)) | (1 << (x+d-1))
                four_ap_conflicts[idx].append(m)
            # x is 4th: x-3d, x-2d, x-d
            if x-3*d >= 1:
                m = (1 << (x-3*d-1)) | (1 << (x-2*d-1)) | (1 << (x-d-1))
                four_ap_conflicts[idx].append(m)

    # Helper to calculate score of a mask
    def get_score(mask):
        score = 0
        for x in range(n):
            if mask & (1 << x):
                for m in three_aps[x]:
                    if (mask & m) == m:
                        score += 1
        return score // 2 # Each 3-AP is counted twice? 
        # Wait. 
        # If x is middle, we count it.
        # If x is start, we count it.
        # If x is end, we count it.
        # A 3-AP {a, b, c} with d>0.
        # When x=a (start), we check {a+d, a+2d} = {b, c}. Count 1.
        # When x=b (middle), we check {b-d, b+d} = {a, c}. Count 1.
        # When x=c (end), we check {c-d, c-2d} = {b, a}. Count 1.
        # So each 3-AP is counted 3 times?
        # No, my precomputation adds masks for each role.
        # Let's verify.
        # AP: 1, 2, 3. d=1.
        # x=1 (start): adds mask {2,3}.
        # x=2 (middle): adds mask {1,3}.
        # x=3 (end): adds mask {2,1}.
        # If mask has 1,2,3:
        # x=1: (mask & {2,3}) == {2,3} -> True.
        # x=2: (mask & {1,3}) == {1,3} -> True.
        # x=3: (mask & {2,1}) == {2,1} -> True.
        # Total count 3.
        # So score = sum / 3.
        # Let's use this.
        
    # Helper to check if adding x is valid
    def is_valid_addition(mask, x_idx):
        # x_idx is 0-indexed
        for cm in four_ap_conflicts[x_idx]:
            if (mask & cm) == cm:
                return False
        return True

    # Helper to get delta score for adding x
    def delta_score_add(mask, x_idx):
        delta = 0
        for m in three_aps[x_idx]:
            if (mask & m) == m:
                delta += 1
        return delta

    # Helper to get delta score for removing x
    def delta_score_remove(mask, x_idx):
        # mask currently contains x
        # We need to count how many 3-APs are destroyed.
        # This is the same as delta_score_add but on the mask WITHOUT x?
        # No. If we remove x, we lose all 3-APs that used x.
        # The count is the number of pairs in three_aps[x_idx] that are present in (mask without x).
        # Which is the same as the number of pairs in three_aps[x_idx] present in mask.
        # Because x is not in the pair masks.
        delta = 0
        for m in three_aps[x_idx]:
            if (mask & m) == m:
                delta += 1
        return delta

    # Initial Greedy Construction
    current_mask = 0
    for x in range(1, n+1):
        idx = x - 1
        if is_valid_addition(current_mask, idx):
            current_mask |= (1 << idx)
            
    current_score = get_score(current_mask)
    best_mask = current_mask
    best_score = current_score
    
    # Simulated Annealing
    start_time = time.time()
    deadline = 80.0 # 80 seconds
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick random element
        x_idx = random.randint(0, n-1)
        
        if current_mask & (1 << x_idx):
            # Remove
            new_score = current_score - delta_score_remove(current_mask, x_idx)
            if new_score > best_score:
                best_score = new_score
                best_mask = current_mask ^ (1 << x_idx)
            # Accept with prob exp(-delta/T) where delta = current - new = positive
            # So exp((new-current)/T)
            if new_score >= current_score or random.random() < pow(2.71828, (new_score - current_score) / T):
                current_mask ^= (1 << x_idx)
                current_score = new_score
        else:
            # Add
            if is_valid_addition(current_mask, x_idx):
                new_score = current_score + delta_score_add(current_mask, x_idx)
                if new_score > best_score:
                    best_score = new_score
                    best_mask = current_mask | (1 << x_idx)
                if new_score >= current_score or random.random() < pow(2.71828, (new_score - current_score) / T):
                    current_mask |= (1 << x_idx)
                    current_score = new_score
        
        T *= cooling
        if T < 0.01:
            T = 0.01 # Floor

    # Convert best_mask to list
    result = [i+1 for i in range(n) if best_mask & (1 << i)]
    return result
