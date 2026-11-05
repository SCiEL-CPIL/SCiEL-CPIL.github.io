import time
import random
from collections import Counter

N = 256
DEADLINE_MARGIN = 5.0  # seconds before the 90s kill

# ---- Precompute 3-AP pair lists ----
f3 = [[] for _ in range(N + 1)]
m3 = [[] for _ in range(N + 1)]
l3 = [[] for _ in range(N + 1)]

for x in range(1, N + 1):
    for d in range(1, (N - x) // 2 + 1):
        f3[x].append((x + d, x + 2 * d))
    for d in range(1, min(x - 1, N - x) + 1):
        m3[x].append((x - d, x + d))
    for d in range(1, (x - 1) // 2 + 1):
        l3[x].append((x - 2 * d, x - d))

# ---- Precompute 4-AP conflict masks ----
f4_masks = [[] for _ in range(N + 1)]

def to_mask(vals):
    m = 0
    for v in vals:
        m |= (1 << (v - 1))
    return m

for x in range(1, N + 1):
    for d in range(1, (N - x) // 3 + 1):
        f4_masks[x].append(to_mask([x + d, x + 2 * d, x + 3 * d]))
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        f4_masks[x].append(to_mask([x - d, x + d, x + 2 * d]))
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        f4_masks[x].append(to_mask([x - 2 * d, x - d, x + d]))
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
    g = 0
    for a, b in f3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            g += 1
    for a, b in m3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            g += 1
    for a, b in l3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            g += 1
    return g

def loss_remove(A, r):
    loss = 0
    for a, b in f3[r]:
        if is_in_set(A, a) and is_in_set(A, b):
            loss += 1
    for a, b in m3[r]:
        if is_in_set(A, a) and is_in_set(A, b):
            loss += 1
    for a, b in l3[r]:
        if is_in_set(A, a) and is_in_set(A, b):
            loss += 1
    return loss

def count_3AP_total(A):
    count = 0
    temp = A
    while temp:
        lsb = temp & -temp
        b = lsb.bit_length()
        temp ^= lsb
        for a, c in m3[b]:
            if is_in_set(A, a) and is_in_set(A, c):
                count += 1
    return count

def greedy_forward(rng):
    A = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        if not creates_4AP(A, x):
            A |= (1 << (x - 1))
    return A

def greedy_reverse(rng):
    """Start with all elements, remove those in 4-APs iteratively."""
    A = (1 << N) - 2  # bits 1..N set (value 1 is bit 0, value 256 is bit 255)
    
    # Find all 4-APs in the current set
    while True:
        # Count how many 4-APs each element is in
        elem_4ap_count = [0] * (N + 1)
        
        # For each element x in A, check if it's part of a 4-AP
        # We need to find all 4-APs where all 4 elements are in A
        # Iterate through all possible 4-APs
        for a in range(1, N + 1):
            if not is_in_set(A, a):
                continue
            for d in range(1, (N - a) // 3 + 1):
                b = a + d
                c = b + d
                e = c + d
                if e > N:
                    break
                if is_in_set(A, b) and is_in_set(A, c) and is_in_set(A, e):
                    elem_4ap_count[a] += 1
                    elem_4ap_count[b] += 1
                    elem_4ap_count[c] += 1
                    elem_4ap_count[e] += 1
        
        # Find the element with the highest 4-AP count
        max_count = 0
        worst_elem = -1
        for x in range(1, N + 1):
            if is_in_set(A, x) and elem_4ap_count[x] > max_count:
                max_count = elem_4ap_count[x]
                worst_elem = x
        
        if worst_elem == -1 or max_count == 0:
            break  # No more 4-APs
        
        # Remove the worst element
        A &= ~(1 << (worst_elem - 1))
    
    return A

def local_search(A, deadline):
    best_score = count_3AP_total(A)
    
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
        
        if best_x != -1:
            A |= (1 << (best_x - 1))
            best_score += best_gain
            improved = True
            continue
        
        # 2. Try single swaps (remove r, add x)
        best_net = 0
        best_r = -1
        best_x = -1
        
        for r in range(1, N + 1):
            if not is_in_set(A, r):
                continue
            
            loss = loss_remove(A, r)
            A_temp = A & ~(1 << (r - 1))
            
            for x in range(1, N + 1):
                if is_in_set(A, x) and x != r:
                    continue
                if x == r:
                    continue
                
                if creates_4AP(A_temp, x):
                    continue
                
                g = gain_add(A_temp, x)
                net = g - loss
                
                if net > best_net:
                    best_net = net
                    best_r = r
                    best_x = x
        
        if best_r != -1 and best_x != -1:
            A = A & ~(1 << (best_r - 1))
            A |= (1 << (best_x - 1))
            best_score += best_net
            improved = True
            continue
        
        if not improved:
            break
            
    return best_score, A

def construct():
    start = time.time()
    deadline = start + 85.0 - DEADLINE_MARGIN  # 80s effective
    
    rng = random.Random(42)
    best_A = 0
    best_score = -1
    
    # Strategy 1: Multiple forward greedy restarts
    num_forward = 15
    for i in range(num_forward):
        if time.time() > deadline - 20.0:
            break
        A = greedy_forward(rng)
        score, A = local_search(A, deadline - 15.0)
        if score > best_score:
            best_score = score
            best_A = A
    
    # Strategy 2: Multiple reverse greedy restarts
    num_reverse = 8
    for i in range(num_reverse):
        if time.time() > deadline - 10.0:
            break
        A = greedy_reverse(rng)
        score, A = local_search(A, deadline - 5.0)
        if score > best_score:
            best_score = score
            best_A = A
    
    # Strategy 3: Final polish on the best incumbent
    if best_A != 0 and time.time() < deadline - 1.0:
        score, best_A = local_search(best_A, deadline - 0.5)
        best_score = score
    
    if best_A == 0:
        best_A = greedy_forward(random.Random(0))
    
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
    
    # Validity check: ensure no 4-APs
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
