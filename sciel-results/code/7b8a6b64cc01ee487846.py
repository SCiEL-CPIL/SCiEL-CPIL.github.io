"""Model-and-optimize: precomputed 3-AP/4-AP structures + fast hill climbing with bitmasks."""
import time
import numpy as np

N = 256
DEADLINE = 0.85  # fraction of 90s

def construct():
    start = time.time()
    deadline = start + 90 * DEADLINE
    
    # Precompute all 3-APs
    three_aps = []  # list of (a, b, c) where a<b<c and b-a==c-b
    for a in range(1, N+1):
        for d in range(1, (N - a + 1) // 2 + 1):
            b = a + d
            c = a + 2*d
            if c <= N:
                three_aps.append((a, b, c))
    
    # For each element x, precompute which 3-APs it participates in
    x_to_3ap_partners = {x: [] for x in range(1, N+1)}
    for (a, b, c) in three_aps:
        x_to_3ap_partners[a].append((b, c))  # a is first
        x_to_3ap_partners[b].append((a, c))  # b is middle
        x_to_3ap_partners[c].append((a, b))  # c is last
    
    # Precompute 4-AP constraints
    x_to_4ap_constraints = {x: [] for x in range(1, N+1)}
    for x in range(1, N+1):
        # x as first
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            x_to_4ap_constraints[x].append((x+d, x+2*d, x+3*d))
        # x as second
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            x_to_4ap_constraints[x].append((x-d, x+d, x+2*d))
        # x as third
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            x_to_4ap_constraints[x].append((x-2*d, x-d, x+d))
        # x as fourth
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            x_to_4ap_constraints[x].append((x-3*d, x-2*d, x-d))
    
    # Bitmask helpers: bit i corresponds to element i+1
    def set_to_mask(S):
        m = 0
        for x in S:
            m |= (1 << (x - 1))
        return m
    
    def mask_to_set(m):
        return [i+1 for i in range(N) if (m >> i) & 1]
    
    def count_3aps_from_mask(mask):
        """Count 3-APs using bitmask."""
        count = 0
        for (a, b, c) in three_aps:
            if (mask >> (a-1)) & 1 and (mask >> (b-1)) & 1 and (mask >> (c-1)) & 1:
                count += 1
        return count
    
    def has_4ap_from_mask(mask, x):
        """Check if adding x to mask creates a 4-AP."""
        for (y, z, w) in x_to_4ap_constraints[x]:
            if ((mask >> (y-1)) & 1) and ((mask >> (z-1)) & 1) and ((mask >> (w-1)) & 1):
                return True
        return False
    
    def new_3aps_with_x_from_mask(mask, x):
        """Count new 3-APs created by adding x to mask."""
        count = 0
        for (y, z) in x_to_3ap_partners[x]:
            if ((mask >> (y-1)) & 1) and ((mask >> (z-1)) & 1):
                count += 1
        return count
    
    # Initial construction: greedy with scoring
    def greedy_construct(seed):
        rng = np.random.default_rng(seed)
        mask = 0
        order = list(range(1, N+1))
        rng.shuffle(order)
        
        for x in order:
            if has_4ap_from_mask(mask, x):
                continue
            gain = new_3aps_with_x_from_mask(mask, x)
            if gain > 0 or mask.bit_count() < 20:
                mask |= (1 << (x - 1))
        return mask
    
    # Hill climbing with bitmask
    def hill_climb(mask, deadline):
        current_score = count_3aps_from_mask(mask)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding elements
            for x in range(1, N+1):
                if (mask >> (x-1)) & 1:
                    continue
                if has_4ap_from_mask(mask, x):
                    continue
                gain = new_3aps_with_x_from_mask(mask, x)
                if gain > 0:
                    mask |= (1 << (x-1))
                    current_score += gain
                    improved = True
            
            # Try swapping: remove x, add y
            mask_list = mask_to_set(mask)
            for x in mask_list:
                if time.time() > deadline:
                    return mask, current_score
                
                # Remove x temporarily
                temp_mask = mask & ~(1 << (x-1))
                lost = new_3aps_with_x_from_mask(temp_mask, x)  # 3-APs involving x that are currently present
                
                best_y = None
                best_net = -1
                for y in range(1, N+1):
                    if (temp_mask >> (y-1)) & 1:
                        continue
                    if has_4ap_from_mask(temp_mask, y):
                        continue
                    gain_y = new_3aps_with_x_from_mask(temp_mask, y)
                    net = gain_y - lost
                    if net > best_net:
                        best_net = net
                        best_y = y
                
                if best_y is not None and best_net > 0:
                    mask = temp_mask | (1 << (best_y-1))
                    current_score = count_3aps_from_mask(mask)
                    improved = True
                # else: keep original mask (x stays)
        
        return mask, current_score
    
    # Try multiple seeds
    best_mask = None
    best_score = -1
    
    for seed in range(50):
        if time.time() > deadline:
            break
        m = greedy_construct(seed)
        score = count_3aps_from_mask(m)
        if score > best_score:
            best_score = score
            best_mask = m
    
    # Hill climb the best
    if best_mask is not None:
        best_mask, best_score = hill_climb(best_mask, deadline)
    
    if best_mask is None or best_mask == 0:
        return [1]
    
    return mask_to_set(best_mask)
