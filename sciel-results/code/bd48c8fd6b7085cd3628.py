import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0
    
    n = 256
    
    # Precompute 4-APs containing each x
    # aps_with_x[x] = list of masks of the other 3 elements
    # x is 1-indexed. Mask bits are 0-indexed.
    aps_with_x = [[] for _ in range(n + 1)]
    
    for d in range(1, 86):
        for a in range(1, n - 3*d + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            # AP: a, b, c, e
            # For each element, add the mask of the others
            # a: others are b, c, e
            mask_a = (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            aps_with_x[a].append(mask_a)
            
            # b: others are a, c, e
            mask_b = (1 << (a-1)) | (1 << (c-1)) | (1 << (e-1))
            aps_with_x[b].append(mask_b)
            
            # c: others are a, b, e
            mask_c = (1 << (a-1)) | (1 << (b-1)) | (1 << (e-1))
            aps_with_x[c].append(mask_c)
            
            # e: others are a, b, c
            mask_e = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            aps_with_x[e].append(mask_e)

    def creates_4ap(mask, x):
        # Check if any of the other 3 sets for x is subset of mask
        for other_mask in aps_with_x[x]:
            if (mask & other_mask) == other_mask:
                return True
        return False

    def count_3aps(mask):
        count = 0
        for d in range(1, 128):
            inter = mask & (mask >> d) & (mask >> (2*d))
            count += inter.bit_count()
        return count

    def gain_3aps(mask, x):
        gain = 0
        for d in range(1, 128):
            if x - d >= 1 and x + d <= 256:
                if (mask & (1 << (x - d - 1))) and (mask & (1 << (x + d - 1))):
                    gain += 1
            if x + 2*d <= 256:
                if (mask & (1 << (x + d - 1))) and (mask & (1 << (x + 2*d - 1))):
                    gain += 1
            if x - 2*d >= 1:
                if (mask & (1 << (x - 2*d - 1))) and (mask & (1 << (x - d - 1))):
                    gain += 1
        return gain

    def loss_3aps(mask, y):
        return gain_3aps(mask ^ (1 << (y - 1)), y)

    # Initialize
    current_mask = 0
    candidates = list(range(1, 257))
    random.shuffle(candidates)
    
    for x in candidates:
        if not creates_4ap(current_mask, x):
            current_mask |= (1 << (x - 1))

    # Local Search
    while time.time() - start_time < time_limit:
        best_x = -1
        best_gain = 0
        
        for x in range(1, 257):
            if current_mask & (1 << (x - 1)):
                continue
            if creates_4ap(current_mask, x):
                continue
            g = gain_3aps(current_mask, x)
            if g > best_gain:
                best_gain = g
                best_x = x
        
        if best_x != -1:
            current_mask |= (1 << (best_x - 1))
        else:
            best_y = -1
            min_loss = float('inf')
            
            for y in range(1, 257):
                if current_mask & (1 << (y - 1)):
                    l = loss_3aps(current_mask, y)
                    if l < min_loss:
                        min_loss = l
                        best_y = y
            
            if best_y != -1:
                current_mask &= ~(1 << (best_y - 1))
            else:
                break

    result = []
    for i in range(256):
        if current_mask & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
