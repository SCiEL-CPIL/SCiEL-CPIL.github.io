import time
import random
import math

N = 256
DEADLINE_MARGIN = 5.0

# Precomputation for 3-APs and 4-APs
# 3-AP: (a, b, c) such that a+c=2b, a<b<c.
# For each x, we store the pairs (y, z) such that x is part of a 3-AP with y and z.
# If x is the middle, y=x-d, z=x+d.
# If x is the first, y=x+d, z=x+2d.
# If x is the last, y=x-2d, z=x-d.

# We will use a bitmask representation for the set A.
# A is an integer where bit (i-1) is 1 if i is in A.

# Precompute 3-AP contributions
# gain_3ap[x]: list of (y, z) such that {x, y, z} is a 3-AP.
# To count 3-APs in A, we can sum for each x in A the number of pairs (y, z) in gain_3ap[x]
# such that both y and z are in A. However, this counts each 3-AP 3 times (once for each element).
# So total_3aps = (sum_{x in A} count_pairs(x)) / 3.

# Actually, it's easier to just define the pairs for each position.
# Let's precompute for each x, the list of other two elements it forms a 3-AP with.
# We'll just store a list of tuples (y, z).

ap3_pairs = [[] for _ in range(N + 1)]
for x in range(1, N + 1):
    # x is middle: x-d, x+d
    for d in range(1, min(x - 1, N - x) + 1):
        ap3_pairs[x].append((x - d, x + d))
    # x is first: x+d, x+2d
    for d in range(1, (N - x) // 2 + 1):
        ap3_pairs[x].append((x + d, x + 2 * d))
    # x is last: x-2d, x-d
    for d in range(1, (x - 1) // 2 + 1):
        ap3_pairs[x].append((x - 2 * d, x - d))

# Precompute 4-AP conflicts
# For each x, list of masks. If (A & mask) == mask, adding x creates a 4-AP.
# Actually, we want to know if x *completes* a 4-AP given current A.
# So for each x, we look for all 4-APs containing x.
# If the other 3 elements are in A, then adding x is forbidden.
# Let's precompute for each x, a list of masks representing the other 3 elements of 4-APs containing x.

ap4_masks = [[] for _ in range(N + 1)]

def to_mask(vals):
    m = 0
    for v in vals:
        m |= (1 << (v - 1))
    return m

for x in range(1, N + 1):
    # x is 1st: x, x+d, x+2d, x+3d
    for d in range(1, (N - x) // 3 + 1):
        ap4_masks[x].append(to_mask([x + d, x + 2 * d, x + 3 * d]))
    # x is 2nd: x-d, x, x+d, x+2d
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        ap4_masks[x].append(to_mask([x - d, x + d, x + 2 * d]))
    # x is 3rd: x-2d, x-d, x, x+d
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        ap4_masks[x].append(to_mask([x - 2 * d, x - d, x + d]))
    # x is 4th: x-3d, x-2d, x-d, x
    for d in range(1, (x - 1) // 3 + 1):
        ap4_masks[x].append(to_mask([x - 3 * d, x - 2 * d, x - d]))

def is_in_set(A, x):
    return bool((A >> (x - 1)) & 1)

def creates_4ap(A, x):
    # Check if adding x to A creates a 4-AP.
    # This happens if there exists a mask in ap4_masks[x] such that all bits in mask are set in A.
    for mask in ap4_masks[x]:
        if (A & mask) == mask:
            return True
    return False

def gain_3ap(A, x):
    """Calculate the change in the number of 3-APs if we add x to A."""
    # We count the number of pairs (y, z) in ap3_pairs[x] such that y and z are in A.
    # Each such pair forms a 3-AP with x.
    # Since we are adding x, all these 3-APs are new.
    count = 0
    for y, z in ap3_pairs[x]:
        if is_in_set(A, y) and is_in_set(A, z):
            count += 1
    return count

def loss_3ap(A, x):
    """Calculate the change in the number of 3-APs if we remove x from A."""
    # We lose the 3-APs that involve x.
    # These are the pairs (y, z) in ap3_pairs[x] such that y and z are in A.
    count = 0
    for y, z in ap3_pairs[x]:
        if is_in_set(A, y) and is_in_set(A, z):
            count += 1
    return count

def count_total_3aps(A):
    """Count total 3-APs in A."""
    # Sum gains for all x in A, divide by 3.
    # Or just sum for each x in A, the number of pairs in A.
    # Each 3-AP (a,b,c) is counted 3 times: once for a, once for b, once for c.
    total = 0
    temp = A
    while temp:
        lsb = temp & -temp
        x = lsb.bit_length()
        total += gain_3ap(A, x)
        temp ^= lsb
    return total // 3

def random_valid_seed(rng):
    """Generate a random valid set using greedy method."""
    A = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        if not creates_4ap(A, x):
            A |= (1 << (x - 1))
    return A

def simulate_annealing(start_A, start_time, deadline):
    A = start_A
    current_score = count_total_3aps(A)
    best_A = A
    best_score = current_score
    
    # Initial temperature
    # The score range is roughly 0 to ~1000. Let's start high.
    T = 10.0
    T_min = 0.1
    # Cooling rate
    # We want to cool down over the available time.
    # Let's estimate the number of steps we can do.
    # Each step involves checking 4-AP validity and 3-AP gain/loss.
    # This is fast, maybe ~1000 steps/sec.
    # Let's just use a time-based cooling schedule.
    
    while time.time() < deadline:
        # Pick a random move: add, remove, or swap
        move_type = rng.random()
        
        if move_type < 0.4:
            # Add
            # Find candidates not in A
            candidates = [x for x in range(1, N + 1) if not is_in_set(A, x)]
            if not candidates:
                continue
            x = rng.choice(candidates)
            if creates_4ap(A, x):
                continue
            delta = gain_3ap(A, x)
            new_A = A | (1 << (x - 1))
        elif move_type < 0.7:
            # Remove
            # Find candidates in A
            # To avoid removing too many, we can bias towards keeping size.
            # But for SA, we should allow removal.
            # Let's pick a random element in A.
            temp = A
            if not temp:
                continue
            # Weighted choice? No, uniform is fine.
            # We can iterate to pick a random set bit.
            # Or just pick a random x in 1..N and check if it's in A.
            # For efficiency, let's maintain a list of elements in A? 
            # Or just try random x until we find one in A.
            x = None
            while x is None:
                x = rng.randint(1, N)
                if is_in_set(A, x):
                    break
            delta = -loss_3ap(A, x)
            new_A = A & ~(1 << (x - 1))
        else:
            # Swap
            # Pick r in A, x not in A.
            # To make it efficient, we can pick a random r in A and random x not in A.
            temp = A
            if not temp:
                continue
            r = None
            while r is None:
                r = rng.randint(1, N)
                if is_in_set(A, r):
                    break
            
            # Pick x not in A
            x = None
            tries = 0
            while x is None and tries < 100:
                x = rng.randint(1, N)
                if not is_in_set(A, x):
                    break
                tries += 1
            if x is None:
                continue
            
            # Check if swap is valid
            # Remove r, add x.
            # First, check if adding x to A\{r} creates 4-AP.
            A_temp = A & ~(1 << (r - 1))
            if creates_4ap(A_temp, x):
                continue
            
            # Calculate delta
            # Loss of r: -loss_3ap(A, r)
            # Gain of x: gain_3ap(A_temp, x)
            # Note: gain_3ap(A_temp, x) might be different from gain_3ap(A, x) if r is involved in 3-APs with x.
            # Specifically, if (r, x, y) is a 3-AP, then in A, x and r are both present.
            # When we remove r, that 3-AP is broken.
            # gain_3ap(A, x) counts pairs (y, z) in A. If r is one of them, it counts it.
            # gain_3ap(A_temp, x) counts pairs in A_temp. r is not in A_temp.
            # So we need to calculate gain in A_temp.
            
            loss_r = loss_3ap(A, r)
            gain_x = gain_3ap(A_temp, x)
            delta = gain_x - loss_r
            new_A = A_temp | (1 << (x - 1))

        # Acceptance criterion
        new_score = current_score + delta
        
        if new_score > current_score or rng.random() < math.exp(delta / T):
            A = new_A
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_A = A
        
        # Cool down
        # Simple exponential cooling based on time
        # Or just a fixed step.
        # Let's use a time-based approach to ensure we finish.
        # T = T_min + (T_initial - T_min) * exp(-k * t)
        # We don't know k. Let's just reduce T slightly each step.
        # Given we have ~50s and ~50k steps, we want T to drop to T_min.
        # T_new = T * 0.999
        T *= 0.999
        if T < T_min:
            T = T_min

    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 60.0 - DEADLINE_MARGIN
    rng = random.Random(42)
    
    best_A = 0
    best_score = -1
    
    # Run multiple restarts
    # Each restart gets a fraction of the time.
    # Let's do a long initial run, then shorter ones.
    # Or just loop until time is up.
    
    while time.time() < deadline:
        # Create a seed
        A_seed = random_valid_seed(rng)
        
        # Determine time for this SA run
        # If we have plenty of time, do a long run.
        # If little, do a short run.
        remaining = deadline - time.time()
        # Reserve 1s for final checks
        if remaining < 2.0:
            break
        
        # Let's allocate 80% of remaining time to SA, but cap it to avoid one huge run
        # that might not finish cooling properly?
        # Actually, SA is robust. Let's just use a chunk of time.
        # Let's say we run SA for min(remaining - 1, 10) seconds.
        # This allows multiple restarts.
        run_time = min(remaining - 1.0, 10.0)
        if run_time <= 0:
            break
            
        run_deadline = time.time() + run_time
        
        A_res, score_res = simulate_annealing(A_seed, time.time(), run_deadline)
        
        if score_res > best_score:
            best_score = score_res
            best_A = A_res

    # Final polish: Local search on the best found
    if time.time() < deadline - 0.5 and best_A != 0:
        # Do a quick local search (greedy improvement)
        A = best_A
        improved = True
        while improved and time.time() < deadline - 0.1:
            improved = False
            # Try adds
            for x in range(1, N + 1):
                if is_in_set(A, x):
                    continue
                if creates_4ap(A, x):
                    continue
                g = gain_3ap(A, x)
                if g > 0:
                    A |= (1 << (x - 1))
                    improved = True
                    break # Restart to recalculate gains? 
                    # Actually, adding one might enable others.
                    # But we just break and restart the loop.
            
            # Try swaps
            # This is expensive. Let's just do adds for now in the polish phase.
            # The SA phase should have handled swaps.

        # Update best if improved
        score_final = count_total_3aps(A)
        if score_final > best_score:
            best_score = score_final
            best_A = A

    # Convert bitmask to list
    if best_A == 0:
        return [1] # Fallback
        
    A_list = []
    temp = best_A
    while temp:
        lsb = temp & -temp
        x = lsb.bit_length()
        A_list.append(x)
        temp ^= lsb
        
    # Sort for consistency (optional, but good)
    A_list.sort()
    
    # Hard validity check
    # The algorithm ensures validity, but let's double check.
    # If invalid, we need to fix it.
    # Since we check creates_4ap before adding, it should be valid.
    # However, if there's a bug, we might return an invalid set.
    # Let's do a final check.
    valid_list = []
    mask = 0
    # We need to remove elements that cause 4-APs.
    # This is a repair step.
    for x in A_list:
        # Check if x creates a 4-AP with the *currently kept* set.
        # If yes, skip x.
        # This greedy removal ensures no 4-APs.
        if not creates_4ap(mask, x):
            valid_list.append(x)
            mask |= (1 << (x - 1))
            
    if not valid_list:
        return [1]
        
    return valid_list

