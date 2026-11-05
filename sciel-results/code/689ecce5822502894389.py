import time
import random
import sys

N = 256
DEADLINE_MARGIN = 10  # seconds before the 90s limit

# Precompute 4-term AP masks
FAP_MASKS = []
for a in range(1, N + 1):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = b + d
        e = c + d
        if e <= N:
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            FAP_MASKS.append(mask)

def count_3aps_fast(mask):
    """Count 3-APs using bitmask operations."""
    total = 0
    for d in range(1, 128):
        total += (mask & (mask >> d) & (mask << d)).bit_count()
    return total

def has_4ap_fast(mask):
    """Check if mask contains any 4-AP."""
    for ap_mask in FAP_MASKS:
        if (mask & ap_mask) == ap_mask:
            return True
    return False

def greedy_construct():
    """Greedy construction: add elements that maximize 3-APs."""
    mask = 0
    # Try adding elements in order of potential contribution
    # Or just iterate and add if valid and improves score
    for x in range(1, N + 1):
        bit = 1 << (x - 1)
        new_mask = mask | bit
        if not has_4ap_fast(new_mask):
            # Check if it adds any 3-APs
            # If it does, or even if it doesn't, we might add it to allow future additions
            # For now, just add if valid
            mask = new_mask
    return mask

def local_search(mask, time_limit=5.0):
    """Local search with simulated annealing."""
    start = time.time()
    current_mask = mask
    current_score = count_3aps_fast(current_mask)
    
    best_mask = current_mask
    best_score = current_score
    
    T = 10.0
    cooling = 0.99
    
    while time.time() - start < time_limit and T > 0.01:
        improved = False
        
        # Try adding elements
        unselected_bits = [x for x in range(1, N + 1) if not (current_mask & (1 << (x - 1)))]
        random.shuffle(unselected_bits)
        
        for x in unselected_bits:
            bit = 1 << (x - 1)
            new_mask = current_mask | bit
            if has_4ap_fast(new_mask):
                continue
            new_score = count_3aps_fast(new_mask)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.71828, delta / T):
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                improved = True
                break
        
        if not improved:
            # Try removing elements
            selected_bits = [x for x in range(1, N + 1) if (current_mask & (1 << (x - 1)))]
            random.shuffle(selected_bits)
            
            for x in selected_bits:
                bit = 1 << (x - 1)
                new_mask = current_mask & ~bit
                new_score = count_3aps_fast(new_mask)
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.71828, delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                    improved = True
                    break
        
        if not improved:
            # Try swaps
            if unselected_bits and selected_bits:
                x = random.choice(unselected_bits)
                y = random.choice(selected_bits)
                bit_x = 1 << (x - 1)
                bit_y = 1 << (y - 1)
                new_mask = (current_mask & ~bit_y) | bit_x
                if not has_4ap_fast(new_mask):
                    new_score = count_3aps_fast(new_mask)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < pow(2.71828, delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
                        improved = True
        
        T *= cooling
    
    return best_mask, best_score

def construct():
    start_time = time.time()
    
    best_mask = 0
    best_score = -1
    
    # Strategy 1: Greedy
    if time.time() - start_time < DEADLINE_MARGIN:
        mask1 = greedy_construct()
        score1 = count_3aps_fast(mask1)
        if score1 > best_score:
            best_score = score1
            best_mask = mask1
        
        # Local search on greedy
        if time.time() - start_time < DEADLINE_MARGIN:
            mask1_ls, score1_ls = local_search(mask1, time_limit=min(5.0, DEADLINE_MARGIN - (time.time() - start_time)))
            if score1_ls > best_score:
                best_score = score1_ls
                best_mask = mask1_ls
    
    # Strategy 2: Randomized Greedy with multiple restarts
    for trial in range(20):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        random.seed(trial)
        order = list(range(1, N + 1))
        random.shuffle(order)
        mask2 = 0
        for x in order:
            bit = 1 << (x - 1)
            new_mask = mask2 | bit
            if not has_4ap_fast(new_mask):
                mask2 = new_mask
        
        score2 = count_3aps_fast(mask2)
        if score2 > best_score:
            best_score = score2
            best_mask = mask2
        
        # Local search on randomized greedy
        if time.time() - start_time < DEADLINE_MARGIN:
            mask2_ls, score2_ls = local_search(mask2, time_limit=min(5.0, DEADLINE_MARGIN - (time.time() - start_time)))
            if score2_ls > best_score:
                best_score = score2_ls
                best_mask = mask2_ls
    
    # Strategy 3: Simulated Annealing from best
    if best_mask and time.time() - start_time < DEADLINE_MARGIN:
        remaining = DEADLINE_MARGIN - (time.time() - start_time)
        if remaining > 2.0:
            mask3, score3 = local_search(best_mask, time_limit=min(15.0, remaining))
            if score3 > best_score:
                best_score = score3
                best_mask = mask3
    
    # Convert mask to list
    result = []
    for x in range(1, N + 1):
        if best_mask & (1 << (x - 1)):
            result.append(x)
    
    if not result:
        result = [1]
        
    return result

def validate(result):
    mask = 0
    for x in result:
        mask |= (1 << (x - 1))
    return not has_4ap_fast(mask)

if __name__ == "__main__":
    result = construct()
    score = count_3aps_fast(0) # Just for structure, actual score is in construct
    # Recalculate score for printing
    mask = 0
    for x in result:
        mask |= (1 << (x - 1))
    score = count_3aps_fast(mask)
    valid = validate(result)
    print(f"Result size: {len(result)}")
    print(f"3-AP count: {score}")
    print(f"Valid (no 4-AP): {valid}")
