import time
import random

def construct():
    N = 256
    deadline = time.time() + 55.0  # Safety margin before 60s limit

    # Precompute 4-AP conflicts using bitmasks for O(1) validity checks
    # conflicts[x] is a list of (mask, count) where mask has bits set for the other 3 elements of a 4-AP containing x
    # If all those bits are set in the current mask, adding x creates a 4-AP
    conflicts = [[] for _ in range(N)]
    
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # 4-AP: (a, b, c, e)
            # For a: conflicts are (b, c, e)
            mask_a = (1 << b) | (1 << c) | (1 << e)
            conflicts[a].append(mask_a)
            # For b: conflicts are (a, c, e)
            mask_b = (1 << a) | (1 << c) | (1 << e)
            conflicts[b].append(mask_b)
            # For c: conflicts are (a, b, e)
            mask_c = (1 << a) | (1 << b) | (1 << e)
            conflicts[c].append(mask_c)
            # For e: conflicts are (a, b, c)
            mask_e = (1 << a) | (1 << b) | (1 << c)
            conflicts[e].append(mask_e)

    def is_valid_add(mask, x):
        """Check if adding x to mask creates a 4-AP."""
        for cmask in conflicts[x]:
            if (mask & cmask) == cmask:
                return False
        return True

    def count_3aps(mask):
        """Count 3-term APs in the set represented by mask."""
        count = 0
        for i in range(N):
            if not ((mask >> i) & 1):
                continue
            max_d = (N - 1 - i) // 2
            for d in range(1, max_d + 1):
                j = i + d
                k = i + 2 * d
                if ((mask >> j) & 1) and ((mask >> k) & 1):
                    count += 1
        return count

    def delta_3aps_add(mask, x):
        """Calculate increase in 3-AP count if x is added to mask."""
        delta = 0
        # Case 1: x is the start: x, x+d, x+2d
        max_d = (N - 1 - x) // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x + d)) & 1) and ((mask >> (x + 2 * d)) & 1):
                delta += 1
        # Case 2: x is the middle: x-d, x, x+d
        max_d = min(x, N - 1 - x)
        for d in range(1, max_d + 1):
            if ((mask >> (x - d)) & 1) and ((mask >> (x + d)) & 1):
                delta += 1
        # Case 3: x is the end: x-2d, x-d, x
        max_d = x // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x - 2 * d)) & 1) and ((mask >> (x - d)) & 1):
                delta += 1
        return delta

    def delta_3aps_remove(mask, x):
        """Calculate decrease in 3-AP count if x is removed from mask."""
        return delta_3aps_add(mask, x)

    # Greedy Initialization: build a valid set quickly
    def greedy_init():
        mask = 0
        # Process in random order to break symmetry
        elems = list(range(N))
        random.shuffle(elems)
        
        for x in elems:
            if is_valid_add(mask, x):
                # Add if it increases 3-AP count or with some probability to ensure diversity
                if delta_3aps_add(mask, x) > 0 or random.random() < 0.3:
                    mask |= (1 << x)
        return mask

    # Simulated Annealing
    best_mask = 0
    best_score = -1
    current_mask = greedy_init()
    current_score = count_3aps(current_mask)
    
    if current_score > best_score:
        best_score = current_score
        best_mask = current_mask

    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        move_type = random.random()
        
        new_mask = current_mask
        new_score = current_score
        valid_move = False
        
        if move_type < 0.5:
            # Add
            x = random.randint(0, N - 1)
            if not ((current_mask >> x) & 1):
                if is_valid_add(current_mask, x):
                    new_mask = current_mask | (1 << x)
                    new_score = current_score + delta_3aps_add(current_mask, x)
                    valid_move = True
        elif move_type < 0.8:
            # Remove
            x = random.randint(0, N - 1)
            if (current_mask >> x) & 1:
                new_mask = current_mask & ~(1 << x)
                new_score = current_score - delta_3aps_remove(current_mask, x)
                valid_move = True
        else:
            # Swap
            x = random.randint(0, N - 1)
            if (current_mask >> x) & 1:
                y = random.randint(0, N - 1)
                if not ((current_mask >> y) & 1):
                    temp_mask = current_mask & ~(1 << x)
                    if is_valid_add(temp_mask, y):
                        d_rem = delta_3aps_remove(current_mask, x)
                        d_add = delta_3aps_add(temp_mask, y)
                        new_score = current_score - d_rem + d_add
                        new_mask = temp_mask | (1 << y)
                        valid_move = True

        if valid_move:
            if new_score >= current_score:
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
            else:
                # Accept with probability exp(-delta / T)
                delta = current_score - new_score
                if delta > 0:
                    if random.random() < pow(2.718281828, -delta / T):
                        current_mask = new_mask
                        current_score = new_score
            T *= cooling
            if T < T_min:
                T = T_min

    # Convert best_mask back to sorted list (1-indexed)
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    return sorted(result)
