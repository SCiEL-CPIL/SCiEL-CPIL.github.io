import time
import random
import numpy as np
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 128
    
    # Pre-compute all 4-term APs
    aps = []
    for d in range(1, N // 3 + 1):
        for i in range(N - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
            
    num_aps = len(aps)
    aps_np = np.array(aps, dtype=np.int32)
    
    def count_violations(colors):
        ap_colors = colors[aps_np]
        c0, c1, c2, c3 = ap_colors[:, 0], ap_colors[:, 1], ap_colors[:, 2], ap_colors[:, 3]
        
        d1 = (c0 != c1)
        d2 = (c0 != c2) & (c1 != c2)
        d3 = (c0 != c3) & (c1 != c3) & (c2 != c3)
        
        distinct_count = 1 + d1 + d2 + d3
        return np.sum(distinct_count < 3)

    # Initialize with unique colors (guaranteed valid)
    colors = np.arange(N, dtype=np.int32)
    active_colors = set(colors)
    
    # Function to check if merging color c2 into c1 is valid
    def is_merge_valid(c1, c2, colors):
        # Create a temporary copy of colors to test the merge
        # Instead of copying the whole array, we can check the condition directly
        # Merge c2 -> c1.
        # An AP becomes invalid if it contains both c1 and c2, and its current distinct color count is <= 3.
        # Why? If count is 4, removing one distinct color (c2 becomes c1) leaves 3. Valid.
        # If count is 3, removing one leaves 2. Invalid.
        # If count is 2, removing one leaves 1. Invalid.
        
        # We need to check all APs that contain at least one c1 and at least one c2.
        # If an AP contains c1 and c2, we check its distinct count.
        
        ap_colors = colors[aps_np]
        c0, c1_col, c2_col, c3_col = ap_colors[:, 0], ap_colors[:, 1], ap_colors[:, 2], ap_colors[:, 3]
        
        # Identify APs containing c1 and c2
        has_c1 = (c0 == c1) | (c1_col == c1) | (c2_col == c1) | (c3_col == c1)
        has_c2 = (c0 == c2) | (c1_col == c2) | (c2_col == c2) | (c3_col == c2)
        relevant_aps = has_c1 & has_c2
        
        if not np.any(relevant_aps):
            return True
            
        # Get the colors of the relevant APs
        rel_ap_colors = ap_colors[relevant_aps]
        
        # Calculate distinct counts for these APs
        r_c0, r_c1, r_c2, r_c3 = rel_ap_colors[:, 0], rel_ap_colors[:, 1], rel_ap_colors[:, 2], rel_ap_colors[:, 3]
        
        d1 = (r_c0 != r_c1)
        d2 = (r_c0 != r_c2) & (r_c1 != r_c2)
        d3 = (r_c0 != r_c3) & (r_c1 != r_c3) & (r_c2 != r_c3)
        
        distinct_count = 1 + d1 + d2 + d3
        
        # If any relevant AP has distinct count < 4, the merge will result in < 3 distinct colors
        # (since merging c2 into c1 reduces distinct count by 1 for these APs)
        # Wait, if distinct count is 4, after merge it is 3. Valid.
        # If distinct count is 3, after merge it is 2. Invalid.
        # So we require distinct_count >= 4 for all relevant APs.
        
        return np.all(distinct_count >= 4)

    # Merging Loop
    # We iterate through pairs of active colors.
    # To be efficient, we can shuffle the active colors and try to merge each into the first one,
    # or just pick random pairs.
    
    # Let's try to merge colors to reduce the count.
    # We'll maintain a list of active colors.
    active_list = list(active_colors)
    
    # To speed up, we can try to merge smaller color classes into larger ones?
    # Or just random.
    
    # We will loop until no more merges are possible or time limit.
    # To avoid getting stuck in a local minimum where no single merge works,
    # we can restart or use a different strategy if stuck for too long.
    # But for now, simple greedy merging.
    
    # To ensure we don't miss easy merges, we can sort active colors by frequency?
    # Not necessary.
    
    # We will try all pairs in a random order.
    # If a merge is successful, we remove c2 from active_list and update colors.
    # Then we restart the pair checking from the beginning (or continue).
    # Restarting is safer to find new merges enabled by previous merges.
    
    max_attempts = 100000
    attempts = 0
    last_improve = time.time()
    
    while time.time() < deadline:
        if len(active_list) <= 1:
            break
            
        # Shuffle active list to randomize merge order
        random.shuffle(active_list)
        
        found_merge = False
        # Try to merge each color into the first color in the list?
        # Or try all pairs?
        # Trying all pairs is O(K^2). K is small (starts 128, drops).
        # 128^2 = 16k checks. Each check is ~5000 ops. 80M ops.
        # This might be too slow if we do it every iteration.
        # Let's just pick random pairs.
        
        # Pick a random pair
        c1_idx = random.randint(0, len(active_list) - 1)
        c2_idx = random.randint(0, len(active_list) - 1)
        
        if c1_idx == c2_idx:
            continue
            
        c1 = active_list[c1_idx]
        c2 = active_list[c2_idx]
        
        if is_merge_valid(c1, c2, colors):
            # Perform merge
            colors[colors == c2] = c1
            active_list.remove(c2)
            found_merge = True
            last_improve = time.time()
            
            # If we find a merge, we can restart the search from the beginning of active_list
            # to ensure we don't miss other merges.
            # But restarting the shuffle is fine.
            # We break the inner loop to restart the outer while logic?
            # No, just continue the while loop.
            continue
            
        # If no merge found in this random pick, we continue.
        # To avoid infinite loops with no progress, we can track time since last improve.
        # If no improve for 5 seconds, we can try a "random restart" of the merging strategy?
        # Or just keep going. The randomization should eventually find a valid merge if one exists.
        
        # Optimization: If we haven't found a merge for a long time, 
        # it might be that we are stuck. 
        # But with unique start, there are always merges possible until K is small.
        
        # To speed up, we can check a batch of random pairs.
        # Let's try 100 random pairs per "step".
        for _ in range(100):
            if time.time() > deadline:
                break
                
            c1_idx = random.randint(0, len(active_list) - 1)
            c2_idx = random.randint(0, len(active_list) - 1)
            
            if c1_idx == c2_idx:
                continue
                
            c1 = active_list[c1_idx]
            c2 = active_list[c2_idx]
            
            if is_merge_valid(c1, c2, colors):
                colors[colors == c2] = c1
                active_list.remove(c2)
                found_merge = True
                last_improve = time.time()
                # Restart the batch to pick up new opportunities
                break
        
        if found_merge:
            # Restart the outer loop to re-shuffle
            continue
            
        # If we didn't find a merge in 100 random tries, 
        # it might be that no merge is possible (optimal reached) 
        # or we are just unlucky.
        # To distinguish, we can do a full scan if K is small.
        if len(active_list) < 20:
            # Full scan
            found_merge = False
            for i in range(len(active_list)):
                for j in range(i + 1, len(active_list)):
                    c1 = active_list[i]
                    c2 = active_list[j]
                    if is_merge_valid(c1, c2, colors):
                        colors[colors == c2] = c1
                        active_list.remove(c2)
                        found_merge = True
                        break
                if found_merge:
                    break
            if not found_merge:
                # No merges possible. We are at a local/global minimum.
                break
        else:
            # If K is large and we are stuck, maybe we need to be smarter.
            # But random should work.
            # To prevent infinite loop if stuck, we can force a restart?
            # No, just keep trying.
            pass

    # Final verification
    if count_violations(colors) != 0:
        # This should not happen, but if it does, fall back to unique colors
        colors = np.arange(N, dtype=np.int32)
        
    return colors.tolist()

# EVOLVE-BLOCK-END
