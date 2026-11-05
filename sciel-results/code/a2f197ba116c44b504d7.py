import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 256
    
    # Precomputation
    # 1. Three-AP contributions for each element x
    # For each x, we store a list of masks.
    # If x is the middle term b: mask = (1 << (a-1)) | (1 << (c-1)) where a=x-d, c=x+d
    # If x is the first term a: mask = (1 << (b-1)) | (1 << (c-1)) where b=x+d, c=x+2d
    # If x is the last term c: mask = (1 << (a-1)) | (1 << (b-1)) where a=x-2d, b=x-d
    
    three_ap_masks = [[] for _ in range(N + 1)]
    
    # Middle term contributions
    for b in range(1, N + 1):
        max_d = min(b - 1, N - b)
        for d in range(1, max_d + 1):
            a = b - d
            c = b + d
            mask = (1 << (a - 1)) | (1 << (c - 1))
            three_ap_masks[b].append(mask)
            
    # First term contributions
    for a in range(1, N + 1):
        max_d = (N - a) // 2
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (b - 1)) | (1 << (c - 1))
            three_ap_masks[a].append(mask)
            
    # Last term contributions
    for c in range(1, N + 1):
        max_d = (c - 1) // 2
        for d in range(1, max_d + 1):
            a = c - 2 * d
            b = c - d
            mask = (1 << (a - 1)) | (1 << (b - 1))
            three_ap_masks[c].append(mask)

    # 2. Four-AP validity checks for each element x
    # If we add x, we create a 4-AP if the other 3 elements are present.
    # Store mask of the other 3 elements.
    four_ap_checks = [[] for _ in range(N + 1)]
    
    # Generate all 4-APs
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # For each element in the AP, store the mask of the others
            # a needs b,c,e
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            four_ap_checks[a].append(mask_a)
            
            # b needs a,c,e
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            four_ap_checks[b].append(mask_b)
            
            # c needs a,b,e
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))
            four_ap_checks[c].append(mask_c)
            
            # e needs a,b,c
            mask_e = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            four_ap_checks[e].append(mask_e)

    # Helper to count 3-APs involving x in current mask M
    def count_3ap_with_x(x, M):
        count = 0
        for mask in three_ap_masks[x]:
            if (M & mask) == mask:
                count += 1
        return count

    # Helper to check if adding x creates a 4-AP
    def is_valid_add(x, M):
        for mask in four_ap_checks[x]:
            if (M & mask) == mask:
                return False
        return True

    # Initialize with a valid set
    # Start with empty set and add elements greedily? Or random restarts?
    # Let's try to find a good initial set using random restarts and local search (greedy add)
    
    best_score = -1
    best_mask = 0
    
    def greedy_add_random():
        M = 0
        score = 0
        # Shuffle elements
        elems = list(range(1, N + 1))
        random.shuffle(elems)
        
        for x in elems:
            if is_valid_add(x, M):
                # Calculate delta score
                delta = count_3ap_with_x(x, M)
                # Check if adding x creates a 4-AP? is_valid_add already did that.
                M |= (1 << (x - 1))
                score += delta
        return M, score

    # Try a few random greedy starts
    for _ in range(20):
        M, score = greedy_add_random()
        if score > best_score:
            best_score = score
            best_mask = M
            
    # Simulated Annealing
    current_mask = best_mask
    current_score = best_score
    global_best_mask = best_mask
    global_best_score = best_score
    
    T = 1.0
    cooling_rate = 0.999
    
    # We will run SA until time limit
    # To speed up, we pre-compute the "neighbors" for 3-AP counting? 
    # count_3ap_with_x is O(deg(x)). deg(x) is small (~100).
    
    while time.time() < deadline:
        # Pick random element
        x = random.randint(1, N)
        bit = 1 << (x - 1)
        
        if current_mask & bit:
            # Try removing x
            # Delta score is negative of current contribution
            delta_score = -count_3ap_with_x(x, current_mask)
            new_mask = current_mask ^ bit
            
            # Validity is always maintained when removing
            if delta_score > 0 or random.random() < pow(2.718281828, delta_score / T):
                current_mask = new_mask
                current_score += delta_score
                
                if current_score > global_best_score:
                    global_best_score = current_score
                    global_best_mask = current_mask
        else:
            # Try adding x
            if is_valid_add(x, current_mask):
                delta_score = count_3ap_with_x(x, current_mask)
                new_mask = current_mask | bit
                
                if delta_score > 0 or random.random() < pow(2.718281828, delta_score / T):
                    current_mask = new_mask
                    current_score += delta_score
                    
                    if current_score > global_best_score:
                        global_best_score = current_score
                        global_best_mask = current_mask
        
        T *= cooling_rate
        if T < 0.001:
            T = 0.001
            
    # Convert best_mask to list
    result = []
    for i in range(N):
        if global_best_mask & (1 << i):
            result.append(i + 1)
            
    return sorted(result)

# Test locally? No, just return the function.
