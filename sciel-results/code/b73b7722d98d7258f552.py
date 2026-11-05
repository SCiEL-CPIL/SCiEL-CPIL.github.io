import time
import random
import math

N = 256

# Precompute 3-AP neighbors
# f3[x]: (y, z) such that x, y, z is a 3-AP (x first)
# m3[x]: (y, z) such that y, x, z is a 3-AP (x middle)
# l3[x]: (y, z) such that y, z, x is a 3-AP (x last)
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

# Precompute 4-AP conflict masks
# f4_masks[x]: list of bitmasks. If (A & mask) == mask, adding x creates a 4-AP.
f4_masks = [[] for _ in range(N + 1)]

def to_mask(vals):
    m = 0
    for v in vals:
        m |= (1 << (v - 1))
    return m

for x in range(1, N + 1):
    # x is 1st term
    for d in range(1, (N - x) // 3 + 1):
        f4_masks[x].append(to_mask([x + d, x + 2 * d, x + 3 * d]))
    # x is 2nd term
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        f4_masks[x].append(to_mask([x - d, x + d, x + 2 * d]))
    # x is 3rd term
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        f4_masks[x].append(to_mask([x - 2 * d, x - d, x + d]))
    # x is 4th term
    for d in range(1, (x - 1) // 3 + 1):
        f4_masks[x].append(to_mask([x - 3 * d, x - 2 * d, x - d]))

def is_in_set(A, x):
    return bool((A >> (x - 1)) & 1)

def creates_4AP(A, x):
    for mask in f4_masks[x]:
        if (A & mask) == mask:
            return True
    return False

def count_3APs(A):
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

def loss_remove(A, x):
    l = 0
    for a, b in f3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            l += 1
    for a, b in m3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            l += 1
    for a, b in l3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            l += 1
    return l

def greedy_seed(rng):
    A = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    # We can try to be smarter: sort by "potential" but shuffle is safer for diversity
    for x in order:
        if not creates_4AP(A, x):
            A |= (1 << (x - 1))
    return A

def local_search(A, deadline):
    best_score = count_3APs(A)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding
        best_gain = 0
        best_x = -1
        for x in range(1, N + 1):
            if not is_in_set(A, x) and not creates_4AP(A, x):
                g = gain_add(A, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
        if best_x != -1:
            A |= (1 << (best_x - 1))
            best_score += best_gain
            improved = True
            continue
        
        # Try swapping
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
            
    return best_score, A

def simulated_annealing(A, start_temp, end_temp, deadline):
    current_score = count_3APs(A)
    best_A = A
    best_score = current_score
    
    T = start_temp
    steps = 0
    total_steps = 1000000 # Rough estimate, will be cut by time
    
    while time.time() < deadline:
        # Cooling
        T *= 0.9999
        
        # Choose move type: 0=Add, 1=Remove, 2=Swap
        move_type = rng.randint(0, 2)
        
        new_A = A
        delta_score = 0
        valid_move = True
        
        if move_type == 0: # Add
            x = rng.randint(1, N)
            if not is_in_set(A, x):
                if creates_4AP(A, x):
                    valid_move = False
                else:
                    new_A = A | (1 << (x - 1))
                    delta_score = gain_add(A, x)
        elif move_type == 1: # Remove
            # Pick a random element in A
            # Efficient way: pick random bit position? No, just pick random x and check if in A.
            # To be more uniform, we could maintain a list of elements, but bitmask is faster.
            # Let's just pick random x and retry if not in A, or just pick from a precomputed list of current elements.
            # For speed, let's just pick random x. If not in A, skip.
            x = rng.randint(1, N)
            if is_in_set(A, x):
                new_A = A & ~(1 << (x - 1))
                delta_score = -loss_remove(A, x)
            else:
                valid_move = False
        else: # Swap
            r = rng.randint(1, N)
            if is_in_set(A, r):
                x = rng.randint(1, N)
                if x != r and not is_in_set(A, x):
                    A_temp = A & ~(1 << (r - 1))
                    if creates_4AP(A_temp, x):
                        valid_move = False
                    else:
                        new_A = A_temp | (1 << (x - 1))
                        loss = loss_remove(A, r)
                        gain = gain_add(A_temp, x)
                        delta_score = gain - loss
                else:
                    valid_move = False
            else:
                valid_move = False
        
        if not valid_move:
            continue
            
        new_score = current_score + delta_score
        
        if new_score > current_score:
            A = new_A
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_A = A
        else:
            # Accept with probability exp(delta_score / T)
            if delta_score < 0:
                # Avoid overflow/underflow issues, though T is positive
                prob = math.exp(delta_score / T)
                if rng.random() < prob:
                    A = new_A
                    current_score = new_score
            else:
                # delta_score >= 0 but not > current_score (i.e., equal)
                A = new_A
                current_score = new_score

    return best_score, best_A

def construct():
    start = time.time()
    deadline = start + 55.0
    
    best_score = -1
    best_A = 0
    
    # Multiple restarts with SA
    # We allocate time to each restart
    # Let's do a few restarts
    rng = random.Random(42)
    
    # First, a quick greedy to get a baseline
    A0 = greedy_seed(rng)
    score0, A0 = local_search(A0, start + 5.0)
    if score0 > best_score:
        best_score = score0
        best_A = A0
        
    # Main SA loop
    # We will run SA for the remaining time
    # Split time into chunks for different "temperatures" or just one long run
    # One long run with cooling is usually good.
    
    # Let's try a few restarts of SA to escape deep local minima
    num_restarts = 3
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for i in range(num_restarts):
        restart_deadline = time.time() + time_per_restart
        if i == num_restarts - 1:
            restart_deadline = deadline - 1.0 # Leave 1s for final polish
            
        # Generate a diverse seed
        # Maybe perturb the best found so far? Or just a new greedy seed.
        # New greedy seed is safer for diversity.
        A_curr = greedy_seed(rng)
        
        # Run SA
        # Start temp high enough to allow exploration
        # End temp low enough to converge
        # N=256, max 3-APs is roughly N^2/12 ~ 5000.
        # Delta E is usually small (1-10).
        # T_start ~ 10, T_end ~ 0.1
        s, A_curr = simulated_annealing(A_curr, 10.0, 0.1, restart_deadline)
        
        if s > best_score:
            best_score = s
            best_A = A_curr

    # Final polish
    if time.time() < deadline - 0.5 and best_A != 0:
        final_score, best_A = local_search(best_A, deadline)
        best_score = final_score

    # Convert to list
    A_list = []
    temp = best_A
    while temp:
        lsb = temp & -temp
        x = lsb.bit_length()
        A_list.append(x)
        temp ^= lsb
        
    if not A_list:
        A_list = [1]
        
    return A_list

# Global rng for SA
rng = random.Random(12345)

if __name__ == "__main__":
    res = construct()
    print(len(res))
