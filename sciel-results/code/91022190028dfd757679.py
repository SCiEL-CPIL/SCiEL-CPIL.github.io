import time
import random
from collections import defaultdict

N = 256
DEADLINE_MARGIN = 5.0  # seconds to spare before 90s limit

# Precompute 3-AP information
# For each element x, we need to know which pairs (a,b) with a<b form a 3-AP with x
# A 3-AP is (x, x+d, x+2d) or (x-d, x, x+d) or (x-2d, x-d, x)
# We'll store for each x the list of masks of the other two elements

pair_masks = [[] for _ in range(N + 1)]

for x in range(1, N + 1):
    # x as first term: x, x+d, x+2d
    for d in range(1, (N - x) // 2 + 1):
        a, b = x + d, x + 2 * d
        if a <= N and b <= N:
            mask = (1 << (a - 1)) | (1 << (b - 1))
            pair_masks[x].append(mask)
    # x as middle term: x-d, x, x+d
    for d in range(1, min(x - 1, N - x) + 1):
        a, b = x - d, x + d
        if a >= 1 and b <= N:
            mask = (1 << (a - 1)) | (1 << (b - 1))
            pair_masks[x].append(mask)
    # x as last term: x-2d, x-d, x
    for d in range(1, (x - 1) // 2 + 1):
        a, b = x - 2 * d, x - d
        if a >= 1 and b <= N:
            mask = (1 << (a - 1)) | (1 << (b - 1))
            pair_masks[x].append(mask)

# Precompute 4-AP conflict masks
# For each element x, list of masks of the other 3 elements that would form a 4-AP with x
f4_masks = [[] for _ in range(N + 1)]

def to_mask(vals):
    m = 0
    for v in vals:
        m |= (1 << (v - 1))
    return m

for x in range(1, N + 1):
    # x as 1st: x, x+d, x+2d, x+3d
    for d in range(1, (N - x) // 3 + 1):
        if x + 3 * d <= N:
            f4_masks[x].append(to_mask([x + d, x + 2 * d, x + 3 * d]))
    # x as 2nd: x-d, x, x+d, x+2d
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        if x - d >= 1 and x + 2 * d <= N:
            f4_masks[x].append(to_mask([x - d, x + d, x + 2 * d]))
    # x as 3rd: x-2d, x-d, x, x+d
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        if x - 2 * d >= 1 and x + d <= N:
            f4_masks[x].append(to_mask([x - 2 * d, x - d, x + d]))
    # x as 4th: x-3d, x-2d, x-d, x
    for d in range(1, (x - 1) // 3 + 1):
        if x - 3 * d >= 1:
            f4_masks[x].append(to_mask([x - 3 * d, x - 2 * d, x - d]))

def is_in_set(A, x):
    return bool((A >> (x - 1)) & 1)

def creates_4AP(A, x):
    """Check if adding x to A creates a 4-AP."""
    for mask in f4_masks[x]:
        if (A & mask) == mask:
            return True
    return False

def gain_add(A, x):
    """Number of 3-APs formed by adding x to A."""
    g = 0
    for mask in pair_masks[x]:
        if (A & mask) == mask:
            g += 1
    return g

def loss_remove(A, x):
    """Number of 3-APs lost by removing x from A."""
    l = 0
    for mask in pair_masks[x]:
        if (A & mask) == mask:
            l += 1
    return l

def count_3AP_total(A):
    """Count total 3-APs in set A."""
    count = 0
    temp = A
    while temp:
        lsb = temp & -temp
        b = lsb.bit_length()
        temp ^= lsb
        for mask in pair_masks[b]:
            if (A & mask) == mask:
                count += 1
    return count

def greedy_seed(rng):
    """Create a random greedy seed."""
    A = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        if not creates_4AP(A, x):
            A |= (1 << (x - 1))
    return A

def local_search(A, deadline):
    """Perform local search to improve the solution."""
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
        
        # 2. Try single swaps
        best_net = 0
        best_r = -1
        best_x = -1
        
        for r in range(1, N + 1):
            if not is_in_set(A, r):
                continue
            
            loss = loss_remove(A, r)
            A_temp = A & ~(1 << (r - 1))
            
            for x in range(1, N + 1):
                if x == r or is_in_set(A, x):
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

def simulated_annealing(A, deadline):
    """Simulated annealing to escape local optima."""
    current_score = count_3AP_total(A)
    best_score = current_score
    best_A = A
    T = 100.0
    cooling = 0.999
    
    while time.time() < deadline and T > 0.01:
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.3:
            # Add an element
            candidates = [x for x in range(1, N + 1) if not is_in_set(A, x) and not creates_4AP(A, x)]
            if not candidates:
                T *= cooling
                continue
            x = random.choice(candidates)
            new_A = A | (1 << (x - 1))
            delta = gain_add(A, x)
            
        elif move_type < 0.6:
            # Remove an element
            in_set = [x for x in range(1, N + 1) if is_in_set(A, x)]
            if not in_set:
                T *= cooling
                continue
            x = random.choice(in_set)
            new_A = A & ~(1 << (x - 1))
            delta = -loss_remove(A, x)
            
        else:
            # Swap
            in_set = [x for x in range(1, N + 1) if is_in_set(A, x)]
            not_in_set = [x for x in range(1, N + 1) if not is_in_set(A, x)]
            if not in_set or not not_in_set:
                T *= cooling
                continue
            r = random.choice(in_set)
            x = random.choice(not_in_set)
            A_temp = A & ~(1 << (r - 1))
            if creates_4AP(A_temp, x):
                T *= cooling
                continue
            new_A = A_temp | (1 << (x - 1))
            delta = gain_add(A_temp, x) - loss_remove(A, r)
        
        new_score = current_score + delta
        
        # Accept if better or with probability exp(-delta/T)
        if delta > 0 or random.random() < pow(2.718281828, -(-delta) / T):
            A = new_A
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_A = A
        
        T *= cooling
    
    return best_score, best_A

def construct():
    start = time.time()
    deadline = start + 90.0 - DEADLINE_MARGIN
    rng = random.Random(42)

    best_A = 0
    best_score = -1

    # Run multiple rounds of greedy + local search + simulated annealing
    num_rounds = 0
    while time.time() < deadline - 10.0:
        num_rounds += 1
        A = greedy_seed(rng)
        score, A = local_search(A, deadline - 5.0)
        if score > best_score:
            best_score = score
            best_A = A
        
        # If we have time, try simulated annealing on the best solution
        if time.time() < deadline - 10.0 and best_A != 0:
            sa_score, sa_A = simulated_annealing(best_A, deadline - 5.0)
            if sa_score > best_score:
                best_score = sa_score
                best_A = sa_A
    
    # Final local search on the best solution
    if time.time() < deadline - 1.0 and best_A != 0:
        score, best_A = local_search(best_A, deadline)
        best_score = score

    if best_A == 0:
        best_A = 1

    A_list = [x for x in range(1, N + 1) if is_in_set(best_A, x)]
    
    if not A_list:
        A_list = [1]
        
    return A_list

if __name__ == "__main__":
    res = construct()
    print(len(res))
