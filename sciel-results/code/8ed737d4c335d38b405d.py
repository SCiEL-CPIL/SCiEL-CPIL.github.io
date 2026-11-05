import time
import random

N = 256

# ---- Precompute per-element 3-AP pair lists ----
# f3[x]: list of (y, z) such that x, y, z is a 3-AP (x first)
# m3[x]: list of (y, z) such that y, x, z is a 3-AP (x middle)
# l3[x]: list of (y, z) such that y, z, x is a 3-AP (x last)
f3 = [[] for _ in range(N + 1)]
m3 = [[] for _ in range(N + 1)]
l3 = [[] for _ in range(N + 1)]

for x in range(1, N + 1):
    # x first: x+d, x+2d
    for d in range(1, (N - x) // 2 + 1):
        f3[x].append((x + d, x + 2 * d))
    # x middle: x-d, x+d
    for d in range(1, min(x - 1, N - x) + 1):
        m3[x].append((x - d, x + d))
    # x last: x-2d, x-d
    for d in range(1, (x - 1) // 2 + 1):
        l3[x].append((x - 2 * d, x - d))

# ---- Precompute 4-AP conflict masks ----
# f4_masks[x]: list of bitmasks. Each mask has bits set for the other 3 elements in a 4-AP containing x.
# If (A & mask) == mask, then adding x to A creates a 4-AP.
f4_masks = [[] for _ in range(N + 1)]

def to_mask(vals):
    m = 0
    for v in vals:
        m |= (1 << (v - 1))
    return m

for x in range(1, N + 1):
    # pos 1: x, x+d, x+2d, x+3d
    for d in range(1, (N - x) // 3 + 1):
        f4_masks[x].append(to_mask([x + d, x + 2 * d, x + 3 * d]))
    # pos 2: x-d, x, x+d, x+2d
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        f4_masks[x].append(to_mask([x - d, x + d, x + 2 * d]))
    # pos 3: x-2d, x-d, x, x+d
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        f4_masks[x].append(to_mask([x - 2 * d, x - d, x + d]))
    # pos 4: x-3d, x-2d, x-d, x
    for d in range(1, (x - 1) // 3 + 1):
        f4_masks[x].append(to_mask([x - 3 * d, x - 2 * d, x - d]))

def is_in_set(A, x):
    return bool((A >> (x - 1)) & 1)

def creates_4AP(A, x):
    for mask in f4_masks[x]:
        if (A & mask) == mask:
            return True
    return False

def gain_add(A, x):
    """Calculate number of 3-APs formed by adding x to A using bitmask operations."""
    g = 0
    # x is first: need (x+d) and (x+2d) both in A
    for a, b in f3[x]:
        if (A >> (a-1)) & 1 and (A >> (b-1)) & 1:
            g += 1
    # x is middle: need (x-d) and (x+d) both in A
    for a, b in m3[x]:
        if (A >> (a-1)) & 1 and (A >> (b-1)) & 1:
            g += 1
    # x is last: need (x-2d) and (x-d) both in A
    for a, b in l3[x]:
        if (A >> (a-1)) & 1 and (A >> (b-1)) & 1:
            g += 1
    return g

def count_3AP_total(A):
    """Count total 3-APs in set A (bitmask) using optimized bitmask operations."""
    count = 0
    temp = A
    while temp:
        lsb = temp & -temp
        b = lsb.bit_length()
        temp ^= lsb
        for a, c in m3[b]:
            if (A >> (a-1)) & 1 and (A >> (c-1)) & 1:
                count += 1
    return count

def greedy_seed(rng):
    """Greedy seed that prioritizes elements with high potential 3-AP contributions."""
    A = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    
    # First pass: add elements in random order
    for x in order:
        if not creates_4AP(A, x):
            A |= (1 << (x - 1))
    
    # Second pass: try to add more elements, prioritizing those with high gain
    remaining = [x for x in range(1, N + 1) if not is_in_set(A, x)]
    rng.shuffle(remaining)
    for x in remaining:
        if not creates_4AP(A, x):
            A |= (1 << (x - 1))
    
    return A

def loss_remove(A, r):
    """Calculate number of 3-APs lost when removing r from A."""
    loss = 0
    for a, b in f3[r]:
        if (A >> (a-1)) & 1 and (A >> (b-1)) & 1:
            loss += 1
    for a, b in m3[r]:
        if (A >> (a-1)) & 1 and (A >> (b-1)) & 1:
            loss += 1
    for a, b in l3[r]:
        if (A >> (a-1)) & 1 and (A >> (b-1)) & 1:
            loss += 1
    return loss

def local_search(A, deadline):
    best_score = count_3AP_total(A)
    best_A = A
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try single additions
        best_gain = 0
        best_x = -1
        for x in range(1, N + 1):
            if is_in_set(A, x):
                continue
            if creates_4AP(A, x):
                continue
            g = gain_add(A, x)
            if g > best_gain:
                best_gain = g
                best_x = x
        
        if best_x != -1 and best_gain > 0:
            A |= (1 << (best_x - 1))
            best_score += best_gain
            best_A = A
            no_improve_count = 0
            continue
        
        # 2. Try single swaps (remove r, add x)
        best_net = 0
        best_r = -1
        best_x = -1
        
        in_A = [x for x in range(1, N + 1) if is_in_set(A, x)]
        not_in_A = [x for x in range(1, N + 1) if not is_in_set(A, x)]
        
        for r in in_A:
            loss = loss_remove(A, r)
            A_temp = A & ~(1 << (r - 1))
            
            for x in not_in_A:
                if creates_4AP(A_temp, x):
                    continue
                
                g = gain_add(A_temp, x)
                net = g - loss
                
                if net > best_net:
                    best_net = net
                    best_r = r
                    best_x = x
        
        if best_r != -1 and best_x != -1 and best_net > 0:
            A = A & ~(1 << (best_r - 1))
            A |= (1 << (best_x - 1))
            best_score += best_net
            best_A = A
            no_improve_count = 0
            continue
        
        # 3. Try 2-swap moves (remove r1,r2, add x1,x2) - only if time allows
        if time.time() < deadline - 5.0 and no_improve_count >= 2:
            best_net2 = 0
            best_r1 = best_r2 = -1
            best_x1 = best_x2 = -1
            
            in_A = [x for x in range(1, N + 1) if is_in_set(A, x)]
            not_in_A = [x for x in range(1, N + 1) if not is_in_set(A, x)]
            
            # Limit the search space for 2-swaps
            if len(in_A) <= 30 and len(not_in_A) <= 30:
                for i, r1 in enumerate(in_A):
                    for r2 in in_A[i+1:]:
                        A_temp = A & ~(1 << (r1 - 1)) & ~(1 << (r2 - 1))
                        loss = loss_remove(A, r1) + loss_remove(A & ~(1 << (r1 - 1)), r2)
                        
                        for x1 in not_in_A:
                            if creates_4AP(A_temp, x1):
                                continue
                            A_temp1 = A_temp | (1 << (x1 - 1))
                            g1 = gain_add(A_temp, x1)
                            
                            for x2 in not_in_A:
                                if x2 == x1:
                                    continue
                                if creates_4AP(A_temp1, x2):
                                    continue
                                g2 = gain_add(A_temp1, x2)
                                net2 = g1 + g2 - loss
                                
                                if net2 > best_net2:
                                    best_net2 = net2
                                    best_r1, best_r2 = r1, r2
                                    best_x1, best_x2 = x1, x2
            
            if best_r1 != -1 and best_net2 > 0:
                A = A & ~(1 << (best_r1 - 1)) & ~(1 << (best_r2 - 1))
                A |= (1 << (best_x1 - 1)) | (1 << (best_x2 - 1))
                best_score += best_net2
                best_A = A
                no_improve_count = 0
                continue
        
        no_improve_count += 1
        if no_improve_count > 5:
            break
             
    return best_score, best_A

def construct():
    start = time.time()
    # Safety margin: evaluator kills at 90s wall-clock
    deadline = start + 85.0
    rng = random.Random(12345)

    best_A = 0
    best_score = -1

    # Multi-restart with different random seeds for diversity
    seeds = [12345, 54321, 99999, 11111, 22222, 33333, 44444, 55555]
    
    for seed in seeds:
        if time.time() > deadline - 10.0:
            break
        rng = random.Random(seed)
        A = greedy_seed(rng)
        score, A = local_search(A, deadline)
        if score > best_score:
            best_score = score
            best_A = A

    # Final polish with remaining time on the incumbent
    if time.time() < deadline - 2.0 and best_A != 0:
        score, best_A = local_search(best_A, deadline)
        best_score = score

    if best_A == 0:
        # Fallback
        best_A = greedy_seed(random.Random(0))

    # Convert bitmask to list
    A_list = []
    temp = best_A
    while temp:
        lsb = temp & -temp
        x = lsb.bit_length()
        A_list.append(x)
        temp ^= lsb
    
    if not A_list:
        A_list = [1]
        
    # Hard validity guarantee: drop any element that sits in a 4-AP
    final_mask = 0
    for x in A_list:
        final_mask |= (1 << (x - 1))
        
    valid_list = []
    for x in A_list:
        ok = True
        for mask in f4_masks[x]:
            if (final_mask & mask) == mask:
                ok = False
                break
        if ok:
            valid_list.append(x)
        else:
            final_mask &= ~(1 << (x - 1))
            
    if not valid_list:
        valid_list = [1]

    return valid_list
