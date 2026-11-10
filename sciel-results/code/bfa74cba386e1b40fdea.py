import random

def construct():
    n = 512
    
    # Precompute conflict masks
    conflict_masks = [[] for _ in range(n + 1)]
    
    for x in range(1, n + 1):
        # x is the 4th term
        for d in range(1, (x - 1) // 3 + 1):
            a = x - 3 * d
            mask = (1 << a) | (1 << (a + d)) | (1 << (a + 2 * d))
            conflict_masks[x].append(mask)
            
        # x is the 2nd term
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            a = x - d
            mask = (1 << a) | (1 << (x + d)) | (1 << (x + 2 * d))
            conflict_masks[x].append(mask)
            
        # x is the 3rd term
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            a = x - 2 * d
            mask = (1 << a) | (1 << (a + d)) | (1 << (x + d))
            conflict_masks[x].append(mask)
            
        # x is the 1st term
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            mask = (1 << (x + d)) | (1 << (x + 2 * d)) | (1 << (x + 3 * d))
            conflict_masks[x].append(mask)

    def is_valid(x, set_mask):
        for mask in conflict_masks[x]:
            if (set_mask & mask) == mask:
                return False
        return True

    def greedy_random():
        set_mask = 0
        order = list(range(1, n + 1))
        random.shuffle(order)
        for x in order:
            if is_valid(x, set_mask):
                set_mask |= (1 << x)
        return set_mask

    best_mask = 0
    best_size = 0
    
    # Run 5000 iterations
    for _ in range(5000):
        m = greedy_random()
        size = bin(m).count('1')
        if size > best_size:
            best_size = size
            best_mask = m
            
    result = []
    for i in range(1, n + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    return sorted(result)
