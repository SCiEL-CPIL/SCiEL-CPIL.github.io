import time
import random
import math

N = 256
DEADLINE = 85.0  # Safety margin for 90s limit

# Precomputation
three_aps = []
four_aps = []

# three_ap_info[x] = list of masks for the other 2 elements
three_masks = [[] for _ in range(N + 1)]
# four_ap_info[x] = list of masks for the other 3 elements
four_masks = [[] for _ in range(N + 1)]

def precompute():
    global three_aps, four_aps
    # Generate 3-APs
    for d in range(1, (N // 2) + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            three_aps.append((a, b, c))
            
            # Update masks for a, b, c
            # For a: others are b, c
            mask_bc = (1 << b) | (1 << c)
            three_masks[a].append(mask_bc)
            
            # For b: others are a, c
            mask_ac = (1 << a) | (1 << c)
            three_masks[b].append(mask_ac)
            
            # For c: others are a, b
            mask_ab = (1 << a) | (1 << b)
            three_masks[c].append(mask_ab)

    # Generate 4-APs
    for d in range(1, (N // 3) + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            four_aps.append((a, b, c, e))
            
            # Update masks for a, b, c, e
            # For a: others b, c, e
            mask_bce = (1 << b) | (1 << c) | (1 << e)
            four_masks[a].append(mask_bce)
            
            # For b: others a, c, e
            mask_ace = (1 << a) | (1 << c) | (1 << e)
            four_masks[b].append(mask_ace)
            
            # For c: others a, b, e
            mask_abe = (1 << a) | (1 << b) | (1 << e)
            four_masks[c].append(mask_abe)
            
            # For e: others a, b, c
            mask_abc = (1 << a) | (1 << b) | (1 << c)
            four_masks[e].append(mask_abc)

precompute()

def count_3aps(bits):
    """Count 3-APs in a bitset representation."""
    count = 0
    # Iterate through all 3-APs
    # This is O(num_3aps) ~ 16k. Fast enough for evaluation.
    for a, b, c in three_aps:
        if (bits >> a) & 1 and (bits >> b) & 1 and (bits >> c) & 1:
            count += 1
    return count

def is_valid_add(bits, x):
    """Check if adding x to set represented by bits creates a 4-AP."""
    for mask in four_masks[x]:
        if (bits & mask) == mask:
            return False
    return True

def get_3ap_gain(bits, x):
    """Calculate number of new 3-APs formed by adding x."""
    gain = 0
    for mask in three_masks[x]:
        if (bits & mask) == mask:
            gain += 1
    return gain

def get_3ap_loss(bits, x):
    """Calculate number of 3-APs lost by removing x."""
    # Note: bits contains x. The masks in three_masks[x] do NOT contain x.
    # So we just check if the mask is subset of bits.
    loss = 0
    for mask in three_masks[x]:
        if (bits & mask) == mask:
            loss += 1
    return loss

def greedy_init(seed):
    random.seed(seed)
    bits = 0
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    for x in candidates:
        if is_valid_add(bits, x):
            bits |= (1 << x)
    return bits

def simulated_annealing(bits, time_limit):
    start_time = time.time()
    deadline = start_time + time_limit
    
    current_bits = bits
    current_score = count_3aps(current_bits)
    best_bits = current_bits
    best_score = current_score
    
    # Temperature schedule
    T = 5.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        T *= cooling_rate
        
        # Generate a random move
        # Options: Add, Remove, Swap
        move_type = random.random()
        
        delta = 0
        new_bits = current_bits
        
        if move_type < 0.2:
            # Add a random element not in set
            # Find random x not in current_bits
            # We can pick a random index and check, or maintain a list of available
            # For simplicity, pick random 1..N, if in set, try swap or remove?
            # Let's stick to valid moves.
            # Efficient way: pick random x, if in set, convert to Remove or Swap.
            x = random.randint(1, N)
            if (current_bits >> x) & 1:
                # It's in the set. Let's try Remove
                new_bits = current_bits & ~(1 << x)
                delta = -get_3ap_loss(current_bits, x)
            else:
                # It's not in the set. Try Add
                if is_valid_add(current_bits, x):
                    new_bits = current_bits | (1 << x)
                    delta = get_3ap_gain(current_bits, x)
                else:
                    # Invalid add, skip this iteration? Or try swap?
                    # Let's just skip to save time
                    continue
                    
        elif move_type < 0.4:
            # Remove a random element in set
            # Find a random set bit
            # Trick: pick random x, if not in set, pick another?
            # Better: iterate? No, N=256 is small.
            # Just pick random x, if not in set, reject move (continue)
            x = random.randint(1, N)
            if not ((current_bits >> x) & 1):
                continue
            new_bits = current_bits & ~(1 << x)
            delta = -get_3ap_loss(current_bits, x)
            
        else:
            # Swap: Remove x (in set), Add y (not in set)
            x = random.randint(1, N)
            if not ((current_bits >> x) & 1):
                continue
            y = random.randint(1, N)
            if (current_bits >> y) & 1:
                continue
            
            # Check validity of adding y to set minus x
            bits_without_x = current_bits & ~(1 << x)
            
            valid = True
            for mask in four_masks[y]:
                # Check if mask is subset of bits_without_x
                if (bits_without_x & mask) == mask:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate delta
            # Gain from y in bits_without_x
            gain_y = 0
            for mask in three_masks[y]:
                if (bits_without_x & mask) == mask:
                    gain_y += 1
                    
            # Loss from x in current_bits (which is bits_without_x | (1<<x))
            # The 3-APs lost are those containing x and two others in current_bits.
            # Since y is not in current_bits, no 3-AP involving y is lost.
            # So we just count 3-APs containing x where other 2 are in current_bits.
            loss_x = 0
            for mask in three_masks[x]:
                if (current_bits & mask) == mask:
                    loss_x += 1
            
            delta = gain_y - loss_x
            new_bits = bits_without_x | (1 << y)
            
        # Acceptance
        if delta > 0:
            current_bits = new_bits
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_bits = current_bits
        else:
            # Metropolis criterion
            if T > 0.01:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_bits = new_bits
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_bits = current_bits

    return best_bits, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_result_bits = 0
    best_result_score = -1
    
    # Multiple restarts with different seeds
    # We have ~85 seconds.
    # Each restart: greedy init (fast) + SA (variable time)
    # Let's do a few restarts, allocating time based on remaining budget.
    
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        seed = 42 + i * 137
        
        # Greedy initialization
        bits = greedy_init(seed)
        score = count_3aps(bits)
        
        if score > best_result_score:
            best_result_score = score
            best_result_bits = bits
            
        # Allocate time for SA
        remaining = deadline - time.time()
        if remaining < 2.0:
            continue
            
        # Run SA for a chunk of time
        # If it's the first restart, give it more time?
        # Let's give each restart roughly equal time, but ensure we finish early.
        sa_time = (remaining - 1.0) / (num_restarts - i)
        if sa_time > 5.0:
            sa_time = 5.0 # Cap per restart to allow more restarts
            
        new_bits, new_score = simulated_annealing(bits, sa_time)
        
        if new_score > best_result_score:
            best_result_score = new_score
            best_result_bits = new_bits
            
    # Convert bits to sorted list
    result = [i for i in range(1, N + 1) if (best_result_bits >> i) & 1]
    
    return result
