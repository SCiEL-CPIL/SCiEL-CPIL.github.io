import time
import random
import math

N = 256
DEADLINE = 55.0  # Safety margin under 90s kill

# Precomputation
# For each x, store the other two elements for every 3-AP containing x
# ap3_partners[x] is a list of tuples (y, z) such that {x, y, z} is a 3-AP
# We store y, z as 0-indexed or 1-indexed? Let's use 0-indexed for bitmask shifts.
# Elements are 1..N. Bit i corresponds to element i+1.

ap3_partners = [[] for _ in range(N)]
ap4_partners = [[] for _ in range(N)]

def precompute():
    # 3-APs: a, a+d, a+2d
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            # Indices: a-1, b-1, c-1
            idx_a, idx_b, idx_c = a - 1, b - 1, c - 1
            
            # For a: partners are b, c
            ap3_partners[idx_a].append((idx_b, idx_c))
            # For b: partners are a, c
            ap3_partners[idx_b].append((idx_a, idx_c))
            # For c: partners are a, b
            ap3_partners[idx_c].append((idx_a, idx_b))

    # 4-APs: a, a+d, a+2d, a+3d
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # Indices: a-1, b-1, c-1, e-1
            idx_a, idx_b, idx_c, idx_e = a - 1, b - 1, c - 1, e - 1
            
            # For a: partners are b, c, e
            ap4_partners[idx_a].append((idx_b, idx_c, idx_e))
            # For b: partners are a, c, e
            ap4_partners[idx_b].append((idx_a, idx_c, idx_e))
            # For c: partners are a, b, e
            ap4_partners[idx_c].append((idx_a, idx_b, idx_e))
            # For e: partners are a, b, c
            ap4_partners[idx_e].append((idx_a, idx_b, idx_c))

precompute()

def has_4ap_with_mask(mask, x_idx):
    """Check if adding x_idx to mask creates a 4-AP."""
    for y, z, w in ap4_partners[x_idx]:
        if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
            return True
    return False

def delta_3ap_add(mask, x_idx):
    """Calculate gain in 3-APs if x_idx is added to mask."""
    gain = 0
    for y, z in ap3_partners[x_idx]:
        if (mask >> y) & 1 and (mask >> z) & 1:
            gain += 1
    return gain

def delta_3ap_remove(mask, x_idx):
    """Calculate loss in 3-APs if x_idx is removed from mask."""
    # The APs lost are those involving x_idx and two other elements currently in mask.
    # This is exactly the same calculation as delta_3ap_add, because if {x, y, z} is in mask,
    # removing x destroys it.
    loss = 0
    for y, z in ap3_partners[x_idx]:
        if (mask >> y) & 1 and (mask >> z) & 1:
            loss += 1
    return loss

def count_3aps(mask):
    """Count total 3-APs in mask."""
    count = 0
    for x in range(N):
        if (mask >> x) & 1:
            # Count APs where x is the 'added' element relative to the rest
            # To avoid double counting, we can just sum delta_3ap_add for all present x?
            # No, delta_3ap_add(mask_without_x, x) counts APs involving x.
            # Summing this over all x in mask counts each AP 3 times (once for each member).
            # So total_3aps = sum(delta_3ap_add(mask ^ (1<<x), x)) / 3
            # Or simpler: just count unique APs.
            # Given N=256, iterating over all x and checking partners is fast enough for init.
            pass
    
    # Efficient count:
    # Each 3-AP is counted 3 times if we sum delta_add for each member.
    total = 0
    for x in range(N):
        if (mask >> x) & 1:
            total += delta_3ap_add(mask ^ (1 << x), x)
    return total // 3

def greedy_seed():
    """Build a maximal 4-AP-free set."""
    mask = 0
    for x in range(N):
        if not has_4ap_with_mask(mask, x):
            mask |= (1 << x)
    return mask

def construct():
    start_time = time.time()
    random.seed(12345)
    
    # Initial solution: Greedy
    mask = greedy_seed()
    current_score = count_3aps(mask)
    
    best_mask = mask
    best_score = current_score
    
    # Simulated Annealing Parameters
    # T_start should be high enough to accept significant losses
    # Max loss is roughly N (if we remove a hub element)
    T_start = 50.0
    T_end = 0.01
    # Number of steps is dynamic based on time
    # We run until deadline
    
    # We can also do a quick local search pass first to polish
    # But SA with high T might do it.
    
    # To make SA effective, we need to pick moves carefully.
    # Random bit flip is okay, but maybe biased towards "interesting" bits?
    # For now, uniform random.
    
    # Track time
    while time.time() - start_time < DEADLINE:
        # Cooling schedule: Linear or Exponential?
        # Let's use a fixed number of iterations per "temperature step" or just continuous cooling.
        # Continuous cooling: T = T_start * (T_end/T_start)^(elapsed/total_time)
        # But we don't know total_time exactly, we have a deadline.
        # Let's assume we run for ~50 seconds.
        elapsed = time.time() - start_time
        # Simple exponential cooling
        # T = T_start * exp(-alpha * elapsed)
        # Let's pick alpha such that T drops to T_end in 50s
        # T_end = T_start * exp(-alpha * 50) => alpha = ln(T_start/T_end)/50
        alpha = math.log(T_start / T_end) / 50.0
        T = T_start * math.exp(-alpha * elapsed)
        
        # Pick a random element to flip
        x_idx = random.randint(0, N - 1)
        bit = 1 << x_idx
        is_present = (mask >> x_idx) & 1
        
        if is_present:
            # Try to remove
            new_mask = mask ^ bit
            delta = -delta_3ap_remove(mask, x_idx)
            # Constraint check: Removing always valid for 4-APs
            # Acceptance
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                mask = new_mask
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
        else:
            # Try to add
            # Check 4-AP constraint
            if has_4ap_with_mask(mask, x_idx):
                continue # Reject
            new_mask = mask | bit
            delta = delta_3ap_add(mask, x_idx)
            # Acceptance
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                mask = new_mask
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
    
    # Convert best_mask back to sorted list of integers (1-indexed)
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    return result

# Run once to verify it doesn't crash immediately
if __name__ == "__main__":
    # Just to ensure precomputation and basic logic works in isolation
    # construct() is the entry point
    pass
