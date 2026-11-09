import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 256
    
    # Precompute 4-APs as bitmask arrays
    # Each 4-AP is a 256-bit mask
    all_4aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            mask = 0
            for x in (a, a + d, a + 2 * d, a + 3 * d):
                mask |= (1 << x)
            all_4aps.append(mask)
    
    # Map element to list of 4-AP indices it belongs to
    elem_to_4aps = [[] for _ in range(N + 1)]
    for idx, mask in enumerate(all_4aps):
        for x in range(1, N + 1):
            if mask & (1 << x):
                elem_to_4aps[x].append(idx)
    
    # Precompute 3-APs as bitmask arrays
    all_3aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            mask = 0
            for x in (a, a + d, a + 2 * d):
                mask |= (1 << x)
            all_3aps.append(mask)
    
    # Map element to list of 3-AP indices it belongs to
    elem_to_3aps = [[] for _ in range(N + 1)]
    for idx, mask in enumerate(all_3aps):
        for x in range(1, N + 1):
            if mask & (1 << x):
                elem_to_3aps[x].append(idx)
    
    # State: use Python int as bitmask (arbitrary precision)
    A_mask = 0
    current_3ap_count = 0
    
    # Helper: Check if adding x keeps set 4-AP free
    # Assumes x is NOT in A
    def is_valid_add(x):
        x_bit = 1 << x
        for idx in elem_to_4aps[x]:
            # Check if the other 3 elements are in A
            # mask has 4 bits set, one is x_bit
            # We need to check if (mask & A_mask) has 3 bits (the other 3)
            if (all_4aps[idx] & A_mask & ~x_bit).bit_count() >= 3:
                return False
        return True
        
    # Helper: Check if swapping x (out) and y (in) keeps set 4-AP free
    # Assumes x IS in A, y is NOT in A
    def is_valid_swap(x, y):
        x_bit = 1 << x
        y_bit = 1 << y
        # New mask would be A_mask & ~x_bit | y_bit
        # Check 4-APs containing y
        for idx in elem_to_4aps[y]:
            # The other 3 elements must be in A \ {x}
            other_mask = all_4aps[idx] & ~y_bit
            # Check if all 3 other elements are in A and not x
            if (other_mask & A_mask & ~x_bit).bit_count() >= 3:
                return False
        return True

    # Helper: Calculate gain in 3-AP count if we add x to A
    # Assumes x NOT in A
    def gain_add(x):
        x_bit = 1 << x
        gain = 0
        for idx in elem_to_3aps[x]:
            # Check if other 2 elements are in A
            other_mask = all_3aps[idx] & ~x_bit
            if (other_mask & A_mask).bit_count() == 2:
                gain += 1
        return gain

    # Helper: Calculate loss in 3-AP count if we remove x from A
    # Assumes x IS in A
    def loss_remove(x):
        x_bit = 1 << x
        loss = 0
        for idx in elem_to_3aps[x]:
            # Check if other 2 elements are in A
            other_mask = all_3aps[idx] & ~x_bit
            if (other_mask & A_mask).bit_count() == 2:
                loss += 1
        return loss

    # Helper: Calculate change in 3-AP count if we swap x (out) and y (in)
    # Assumes x IS in A, y is NOT in A
    def delta_swap(x, y):
        x_bit = 1 << x
        y_bit = 1 << y
        
        # Remove x: lose 3-APs containing x and 2 others in A
        loss = 0
        for idx in elem_to_3aps[x]:
            other_mask = all_3aps[idx] & ~x_bit
            if (other_mask & A_mask).bit_count() == 2:
                loss += 1
                
        # Add y: gain 3-APs containing y and 2 others in A \ {x}
        gain = 0
        for idx in elem_to_3aps[y]:
            other_mask = all_3aps[idx] & ~y_bit
            if (other_mask & A_mask & ~x_bit).bit_count() == 2:
                gain += 1
                
        return gain - loss

    # Greedy Initialization
    remaining = list(range(1, N + 1))
    
    # We want to maximize 3-APs.
    # Greedy: pick element with max gain_add, subject to is_valid_add
    while remaining:
        best_x = None
        best_gain = -1
        for x in remaining:
            if is_valid_add(x):
                g = gain_add(x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
        
        if best_x is None:
            break
            
        A_mask |= (1 << best_x)
        current_3ap_count += best_gain
        remaining.remove(best_x)
        
    best_mask = A_mask
    best_score = current_3ap_count
    
    # Simulated Annealing with faster bitmask operations
    T = 5.0
    cooling = 0.9997  # Faster cooling to reach low T sooner
    
    # Precompute lists for random selection
    in_A = [x for x in range(1, N + 1) if A_mask & (1 << x)]
    not_in_A = [x for x in range(1, N + 1) if not (A_mask & (1 << x))]
    
    step = 0
    while time.time() < deadline:
        step += 1
        # Choose move type
        r = random.random()
        
        if r < 0.3:
            # Try Add
            if not_in_A:
                x = random.choice(not_in_A)
                if is_valid_add(x):
                    delta = gain_add(x)
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        A_mask |= (1 << x)
                        in_A.append(x)
                        not_in_A.remove(x)
                        current_3ap_count += delta
                        if current_3ap_count > best_score:
                            best_score = current_3ap_count
                            best_mask = A_mask
        elif r < 0.6:
            # Try Remove
            if in_A:
                x = random.choice(in_A)
                delta = -loss_remove(x)
                # Removing always valid 4-AP wise
                if delta >= 0 or random.random() < np.exp(delta / T):
                    A_mask &= ~(1 << x)
                    in_A.remove(x)
                    not_in_A.append(x)
                    current_3ap_count += delta
                    if current_3ap_count > best_score:
                        best_score = current_3ap_count
                        best_mask = A_mask
        else:
            # Try Swap
            if in_A and not_in_A:
                x = random.choice(in_A)
                y = random.choice(not_in_A)
                if is_valid_swap(x, y):
                    delta = delta_swap(x, y)
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        A_mask = (A_mask & ~(1 << x)) | (1 << y)
                        in_A.remove(x)
                        in_A.append(y)
                        not_in_A.remove(y)
                        not_in_A.append(x)
                        current_3ap_count += delta
                        if current_3ap_count > best_score:
                            best_score = current_3ap_count
                            best_mask = A_mask
        
        # Cool down
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Final polish: local search on best_mask
    # Try all possible swaps to see if we can improve
    polish_deadline = deadline - 2.0
    while time.time() < polish_deadline:
        improved = False
        in_best = [x for x in range(1, N + 1) if best_mask & (1 << x)]
        not_in_best = [x for x in range(1, N + 1) if not (best_mask & (1 << x))]
        
        # Try swaps
        for x in in_best:
            if time.time() >= polish_deadline:
                break
            for y in not_in_best:
                # Temporarily set best_mask for evaluation
                # We need to check validity and compute delta on best_mask
                # This is expensive, so only do a few passes
                pass
        
        if not improved:
            break

    # Return best set found
    # best_mask is a Python int
    # Convert to sorted list of ints
    res = [i for i in range(1, N + 1) if best_mask & (1 << i)]
    
    if not res:
        # Fallback: return a small valid set
        return [1]
        
    return res
