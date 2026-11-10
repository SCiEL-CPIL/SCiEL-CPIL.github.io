"""Model-and-optimize: precomputed 3-AP/4-AP structures + fast hill climbing with bitmasks."""
import time
import numpy as np

N = 256
DEADLINE = 0.85  # fraction of 90s

def construct():
    start = time.time()
    deadline = start + 90 * DEADLINE
    
    # Precompute 3-AP and 4-AP constraints as bitmasks
    # Bit i corresponds to element i+1
    
    # all_3ap_masks: list of bitmasks for all 3-APs
    # three_ap_pairs[x]: list of bitmasks for pairs (y,z) such that {x,y,z} is 3-AP
    # four_ap_triples[x]: list of bitmasks for triples (y,z,w) such that {x,y,z,w} is 4-AP
    
    all_3ap_masks = []
    three_ap_pairs = [[] for _ in range(N + 1)]
    four_ap_triples = [[] for _ in range(N + 1)]
    
    # Generate 3-APs
    for a in range(1, N + 1):
        for d in range(1, (N - a + 1) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
                all_3ap_masks.append(mask)
                
                # x=a is first: partners b,c
                three_ap_pairs[a].append((1 << (b - 1)) | (1 << (c - 1)))
                # x=b is middle: partners a,c
                three_ap_pairs[b].append((1 << (a - 1)) | (1 << (c - 1)))
                # x=c is last: partners a,b
                three_ap_pairs[c].append((1 << (a - 1)) | (1 << (b - 1)))
                
    # Generate 4-APs
    for a in range(1, N + 1):
        for d in range(1, (N - a + 1) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                # a is first: others b,c,e
                four_ap_triples[a].append((1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1)))
                # b is second: others a,c,e
                four_ap_triples[b].append((1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1)))
                # c is third: others a,b,e
                four_ap_triples[c].append((1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1)))
                # e is fourth: others a,b,c
                four_ap_triples[e].append((1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)))
                
    def count_3aps(mask):
        """Count 3-APs in mask."""
        return sum(1 for m in all_3ap_masks if m & mask == m)
        
    def can_add(mask, x):
        """Check if x can be added to mask without creating 4-AP."""
        # x is 1-indexed
        for triple_mask in four_ap_triples[x]:
            if triple_mask & mask == triple_mask:
                return False
        return True
        
    def gain_add(mask, x):
        """Count new 3-APs created by adding x to mask."""
        count = 0
        for pair_mask in three_ap_pairs[x]:
            if pair_mask & mask == pair_mask:
                count += 1
        return count
        
    def lost_remove(mask, x):
        """Count 3-APs lost by removing x from mask."""
        count = 0
        for pair_mask in three_ap_pairs[x]:
            if pair_mask & mask == pair_mask:
                count += 1
        return count

    def greedy_construct(seed):
        rng = np.random.default_rng(seed)
        mask = 0
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        for x in order:
            if can_add(mask, x):
                gain = gain_add(mask, x)
                # Heuristic: add if gain > 0, or if we are very sparse
                if gain > 0 or bin(mask).count('1') < 10:
                    mask |= (1 << (x - 1))
        return mask

    def hill_climb(mask, deadline):
        current_score = count_3aps(mask)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding elements
            for x in range(1, N + 1):
                if time.time() > deadline:
                    return mask, current_score
                if (mask >> (x - 1)) & 1:
                    continue
                if can_add(mask, x):
                    gain = gain_add(mask, x)
                    if gain > 0:
                        mask |= (1 << (x - 1))
                        current_score += gain
                        improved = True
                        
            # Try swapping elements
            # Iterate over x in S
            s_list = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
            for x in s_list:
                if time.time() > deadline:
                    return mask, current_score
                    
                # Remove x
                mask_without_x = mask & ~(1 << (x - 1))
                lost = lost_remove(mask, x)
                
                best_y = None
                best_net = -1
                
                # Try adding y not in S
                for y in range(1, N + 1):
                    if (mask >> (y - 1)) & 1:
                        continue
                    if can_add(mask_without_x, y):
                        gain_y = gain_add(mask_without_x, y)
                        net = gain_y - lost
                        if net > best_net:
                            best_net = net
                            best_y = y
                            
                if best_y is not None and best_net > 0:
                    mask = mask_without_x | (1 << (best_y - 1))
                    current_score += best_net
                    improved = True
                    # Note: current_score update is approximate if we don't recalc, 
                    # but gain/lost are exact for the specific move.
                    # However, interactions might change future gains/losses.
                    # For safety, we could recalc current_score = count_3aps(mask)
                    # But that's O(32k). Let's recalc only if we accept a move?
                    # Actually, gain/lost logic is exact for the delta.
                    # But `lost` was calculated on `mask` (with x).
                    # `gain_y` was calculated on `mask_without_x`.
                    # The net change in total 3-APs is exactly `gain_y - lost`.
                    # So `current_score += best_net` is correct.
        
        return mask, current_score

    best_mask = 0
    best_score = -1
    
    # Run multiple restarts
    for seed in range(100):
        if time.time() > deadline:
            break
        m = greedy_construct(seed)
        s = count_3aps(m)
        if s > best_score:
            best_score = s
            best_mask = m
            
    # Hill climb the best
    if best_mask != 0:
        best_mask, best_score = hill_climb(best_mask, deadline)
        
    # Convert mask to list
    result = [x + 1 for x in range(N) if (best_mask >> x) & 1]
    
    if not result:
        return [1]
        
    return sorted(result)
