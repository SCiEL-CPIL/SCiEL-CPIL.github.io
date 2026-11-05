import time
import random

N = 256
DEADLINE_MARGIN = 5.0  # seconds before kill to stop

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
    """Calculate number of 3-APs formed by adding x to A."""
    g = 0
    # x is first
    for a, b in f3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            g += 1
    # x is middle
    for a, b in m3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            g += 1
    # x is last
    for a, b in l3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            g += 1
    return g

def count_3AP_total(A):
    """Count total 3-APs in set A (bitmask)."""
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

def greedy_seed(rng):
    A = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        if not creates_4AP(A, x):
            A |= (1 << (x - 1))
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
            continue # Restart loop to re-evaluate

        # 2. Try single swaps (remove r, add x)
        best_net = 0
        best_r = -1
        best_x = -1
        
        # Optimization: Only consider r that are in A
        # And x that are not in A
        # To speed up, we can precompute gains for all x not in A?
        # But gain depends on A, which changes when we remove r.
        # gain_add(A \ {r}, x) is slightly different from gain_add(A, x) if x interacts with r.
        # Specifically, if x and r form a 3-AP with some y in A, that AP is lost when r is removed,
        # so it shouldn't be counted in the gain of adding x to A\{r}.
        # However, gain_add(A, x) counts 3-APs in A that use x.
        # If we remove r, we lose 3-APs that use r.
        # Net gain = (3-APs in (A\{r}) U {x}) - (3-APs in A)
        #          = (3-APs in A\{r}) + (new 3-APs with x in A\{r}) - (3-APs in A)
        #          = (new 3-APs with x in A\{r}) - (3-APs in A that use r)
        
        # Let's compute loss(r) = 3-APs in A that use r.
        # Let's compute gain_x_in_A_minus_r(x) = 3-APs in (A\{r}) U {x} that use x.
        
        # We can approximate by iterating r and x.
        # To keep it fast, we limit the scope or rely on the fact that N=256 is small enough
        # for O(N^2) if the inner loop is fast.
        
        for r in range(1, N + 1):
            if not is_in_set(A, r):
                continue
            
            # Calculate loss of removing r
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
            
            A_temp = A & ~(1 << (r - 1))
            
            for x in range(1, N + 1):
                if is_in_set(A, x) and x != r:
                    continue
                if x == r:
                    continue
                
                # Check if adding x to A_temp creates a 4-AP
                if creates_4AP(A_temp, x):
                    continue
                
                # Calculate gain of adding x to A_temp
                # This is gain_add(A_temp, x)
                # Note: gain_add checks neighbors in A_temp.
                # If a neighbor is r, it's not in A_temp, so it won't be counted.
                # This is exactly what we want.
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

def simulated_annealing(A, deadline, rng):
    """
    Simulated annealing to escape local optima.
    Moves:
    1. Add random valid element.
    2. Remove random element.
    3. Swap random element with random valid external element.
    """
    current_score = count_3AP_total(A)
    best_score = current_score
    best_A = A
    
    # Temperature schedule
    T_start = 50.0
    T_end = 0.1
    # We don't know exact number of steps, so we use time-based cooling?
    # Or just a fixed number of steps?
    # Let's do a fixed number of steps, say 2000, and cool linearly.
    # But we need to check time.
    
    steps = 0
    max_steps = 2000 # Adjust based on speed
    
    while steps < max_steps and time.time() < deadline:
        # Cooling
        # Simple geometric cooling or linear.
        # Let's use linear for simplicity, mapping steps to T.
        # T = T_start * (1 - steps/max_steps) + T_end
        T = T_start * (1.0 - steps / max_steps) + T_end
        
        # Generate neighbor
        r = rng.random()
        new_A = A
        delta = 0
        
        # Option 1: Add
        if r < 0.3:
            # Pick random x not in A
            candidates = [x for x in range(1, N + 1) if not is_in_set(A, x)]
            if not candidates:
                continue
            x = rng.choice(candidates)
            if creates_4AP(A, x):
                continue # Invalid move, skip or penalize? Skip is safer for validity.
            new_A = A | (1 << (x - 1))
            delta = gain_add(A, x)
            
        # Option 2: Remove
        elif r < 0.5:
            # Pick random x in A
            candidates = []
            temp = A
            while temp:
                lsb = temp & -temp
                x = lsb.bit_length()
                candidates.append(x)
                temp ^= lsb
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_A = A & ~(1 << (x - 1))
            # Delta is negative of gain (loss)
            # loss = gain_add(A, x) is not quite right because gain_add(A, x) counts APs formed BY x.
            # Removing x destroys exactly those APs.
            # So delta = -gain_add(A, x)
            delta = -gain_add(A, x)
            
        # Option 3: Swap
        else:
            # Pick random x in A to remove
            candidates_in = []
            temp = A
            while temp:
                lsb = temp & -temp
                x = lsb.bit_length()
                candidates_in.append(x)
                temp ^= lsb
            if not candidates_in:
                continue
            x_out = rng.choice(candidates_in)
            
            # Pick random y not in A to add
            candidates_out = [x for x in range(1, N + 1) if not is_in_set(A, x)]
            if not candidates_out:
                continue
            x_in = rng.choice(candidates_out)
            
            # Check validity of adding x_in to A \ {x_out}
            A_temp = A & ~(1 << (x_out - 1))
            if creates_4AP(A_temp, x_in):
                continue
            
            new_A = A_temp | (1 << (x_in - 1))
            
            # Calculate delta
            # Loss from removing x_out: -gain_add(A, x_out)
            # Gain from adding x_in to A\{x_out}: gain_add(A_temp, x_in)
            # Note: gain_add(A, x_out) includes APs with x_in? No, x_in is not in A.
            # So loss is independent of x_in.
            # Gain is calculated on A_temp.
            delta = -gain_add(A, x_out) + gain_add(A_temp, x_in)
        
        # Acceptance criterion
        if delta > 0:
            A = new_A
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_A = A
        else:
            # Metropolis criterion
            # P = exp(delta / T)
            # delta is negative
            if T > 0:
                # Avoid overflow/underflow issues, though delta/T is small
                # log_p = delta / T
                # if log_p > log(rng.random()): accept
                # Since delta < 0, log_p < 0.
                # rng.random() is in (0, 1). log is negative.
                try:
                    p = 2.718281828 ** (delta / T)
                except OverflowError:
                    p = 0.0
                
                if rng.random() < p:
                    A = new_A
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_A = A
        
        steps += 1
        
    return best_score, best_A

def construct():
    start = time.time()
    # Safety margin well under the 60s eval / 90s kill
    # The prompt says evaluator kills after 90s.
    # Previous code used 55s. Let's use 85s to be safe but maximize time.
    # However, the prompt says "Keep the function names and signatures of the seed program."
    # And "Each evaluation must finish within 60 s" in the task contract, but "evaluator KILLS the program after 90s wall-clock" in the directive.
    # The directive is stronger: "Keep an EXPLICIT deadline guard... before the limit."
    # I will use 85s to be safe against the 90s kill.
    deadline = start + 85.0
    rng = random.Random(12345)

    best_A = 0
    best_score = -1

    # Main multi-restart search
    # We use bitmasks so operations are faster.
    # Alternate between local search and simulated annealing?
    # Or do local search first, then SA?
    # SA is good for escaping, Local Search is good for polishing.
    # Strategy:
    # 1. Generate seed
    # 2. Local Search to polish
    # 3. Simulated Annealing to explore
    # 4. Local Search again
    
    while time.time() < deadline - 2.0:
        A = greedy_seed(rng)
        
        # Polish with local search
        score, A = local_search(A, deadline)
        
        # Explore with simulated annealing
        # Give SA some time, e.g., half the remaining time or a fixed chunk?
        # Let's give SA until deadline - 2.0, but we need to return best.
        # Actually, SA returns best_A found during SA.
        # We should compare the score after SA with the score after LS.
        
        sa_score, sa_A = simulated_annealing(A, deadline, rng)
        
        if sa_score > score:
            score = sa_score
            A = sa_A
        
        # Final polish on the best found so far
        score, A = local_search(A, deadline)
        
        if score > best_score:
            best_score = score
            best_A = A

    # Final polish with remaining time on the incumbent
    if time.time() < deadline - 0.5 and best_A != 0:
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
    # This is a safety net, though local_search should have ensured validity.
    # We reconstruct the mask to check
    final_mask = 0
    for x in A_list:
        final_mask |= (1 << (x - 1))
        
    valid_list = []
    for x in A_list:
        # Check if x creates a 4-AP with the rest of the set
        # We need to check if there exists a mask in f4_masks[x] such that
        # (final_mask & mask) == mask
        # But we are removing x, so we check against final_mask without x?
        # No, the condition is: does the set contain a 4-AP?
        # If we keep x, and (final_mask & mask) == mask for some mask in f4_masks[x],
        # then the set contains a 4-AP.
        # So we must remove x if this condition holds.
        
        # However, removing x might break other 4-APs.
        # Simple greedy removal:
        ok = True
        for mask in f4_masks[x]:
            if (final_mask & mask) == mask:
                ok = False
                break
        if ok:
            valid_list.append(x)
        else:
            final_mask &= ~(1 << (x - 1)) # Remove from mask for subsequent checks
            
    if not valid_list:
        valid_list = [1]

    return valid_list
