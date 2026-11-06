import time
import random
import math

N = 256
DEADLINE = 85.0  # Safety margin under 90s kill

# Precomputation
# ap3_partners[x] is a list of tuples (y, z) such that {x, y, z} is a 3-AP
# ap4_partners[x] is a list of tuples (y, z, w) such that {x, y, z, w} is a 4-AP

ap3_partners = [[] for _ in range(N)]
ap4_partners = [[] for _ in range(N)]

def precompute():
    # 3-APs: a, a+d, a+2d
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            idx_a, idx_b, idx_c = a - 1, b - 1, c - 1
            
            ap3_partners[idx_a].append((idx_b, idx_c))
            ap3_partners[idx_b].append((idx_a, idx_c))
            ap3_partners[idx_c].append((idx_a, idx_b))

    # 4-APs: a, a+d, a+2d, a+3d
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            idx_a, idx_b, idx_c, idx_e = a - 1, b - 1, c - 1, e - 1
            
            ap4_partners[idx_a].append((idx_b, idx_c, idx_e))
            ap4_partners[idx_b].append((idx_a, idx_c, idx_e))
            ap4_partners[idx_c].append((idx_a, idx_b, idx_e))
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
    loss = 0
    for y, z in ap3_partners[x_idx]:
        if (mask >> y) & 1 and (mask >> z) & 1:
            loss += 1
    return loss

def count_3aps(mask):
    """Count total 3-APs in mask."""
    total = 0
    for x in range(N):
        if (mask >> x) & 1:
            total += delta_3ap_add(mask ^ (1 << x), x)
    return total // 3

def greedy_seed(order=None):
    """Build a maximal 4-AP-free set."""
    if order is None:
        order = list(range(N))
    mask = 0
    for x in order:
        if not has_4ap_with_mask(mask, x):
            mask |= (1 << x)
    return mask

def local_search_hill_climb(mask, current_score, timeout=5.0, start_time=None):
    """
    Strict hill climbing: Only accept moves that strictly improve the score.
    Moves: Add, Remove, Swap.
    """
    if start_time is None:
        start_time = time.time()
        
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        best_delta = 0
        best_move = None
        best_type = None
        best_idx1 = -1
        best_idx2 = -1

        # 1. Try single additions
        for x in range(N):
            if (mask >> x) & 1:
                continue
            if has_4ap_with_mask(mask, x):
                continue
            delta = delta_3ap_add(mask, x)
            if delta > best_delta:
                best_delta = delta
                best_move = 'add'
                best_idx1 = x

        # 2. Try single removals (usually negative, but might be necessary? No, hill climb only improves)
        # Actually, removals reduce score, so they won't be best_delta > 0 unless score is negative? 
        # Score is always positive. So removals are never best in pure hill climb.
        
        # 3. Try swaps
        # To avoid O(N^2) full scan every time, we can sample or do full if N is small enough.
        # N=256, N^2 = 65536. Each check is fast. Let's do full scan for best move.
        present = [i for i in range(N) if (mask >> i) & 1]
        absent = [i for i in range(N) if not (mask >> i) & 1]
        
        # Optimization: If best_delta from add is high, we might not need to check all swaps?
        # But a swap could yield a higher delta.
        
        for x in present:
            if not (mask >> x) & 1: continue
            loss = delta_3ap_remove(mask, x)
            mask_without_x = mask ^ (1 << x)
            
            for y in absent:
                # Check validity
                if has_4ap_with_mask(mask_without_x, y):
                    continue
                gain = delta_3ap_add(mask_without_x, y)
                delta = gain - loss
                if delta > best_delta:
                    best_delta = delta
                    best_move = 'swap'
                    best_idx1 = x
                    best_idx2 = y

        if best_delta > 0:
            if best_move == 'add':
                mask |= (1 << best_idx1)
                current_score += best_delta
            elif best_move == 'swap':
                mask = (mask ^ (1 << best_idx1)) | (1 << best_idx2)
                current_score += best_delta
            improved = True

    return mask, current_score

def construct():
    start_time = time.time()
    random.seed(42)
    
    best_mask = 0
    best_score = 0
    
    # 1. Generate multiple greedy seeds
    for i in range(20):
        if time.time() - start_time > DEADLINE:
            break
        if i == 0:
            order = list(range(N))
        else:
            order = list(range(N))
            random.shuffle(order)
        mask = greedy_seed(order)
        score = count_3aps(mask)
        if score > best_score:
            best_score = score
            best_mask = mask
            
    # 2. Hill climb the best seed
    mask, current_score = local_search_hill_climb(best_mask, best_score, timeout=10.0)
    if current_score > best_score:
        best_score = current_score
        best_mask = mask

    # 3. Simulated Annealing / WalkSAT Hybrid
    # We use a temperature schedule.
    T_start = 5.0
    T_end = 0.01
    # Total time for SA phase: remaining time
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time < 1.0:
        # Not enough time, return best
        result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
        return result
        
    # We will run SA for the remaining time.
    # Steps per second: We estimate ~1000-5000 moves/sec depending on swap checks.
    # Let's just loop until deadline.
    
    current_mask = best_mask
    current_score = best_score
    
    # Adaptive temperature
    # T(t) = T_start * exp(-lambda * t)
    # We want T to drop to T_end at the end of the phase.
    # T_end = T_start * exp(-lambda * total_time)
    # lambda = ln(T_start/T_end) / total_time
    
    total_sa_time = remaining_time - 1.0 # Keep 1s for final polish/return
    if total_sa_time <= 0:
        total_sa_time = 0.1
        
    lam = math.log(T_start / T_end) / total_sa_time
    
    last_improve = time.time()
    
    while time.time() - start_time < DEADLINE - 1.0:
        elapsed_sa = time.time() - (start_time + (DEADLINE - remaining_time))
        # If elapsed_sa is negative (shouldn't be), clamp
        if elapsed_sa < 0: elapsed_sa = 0
        T = T_start * math.exp(-lam * elapsed_sa)
        
        # Move generation
        # 50% Flip, 50% Swap
        if random.random() < 0.5:
            x = random.randint(0, N - 1)
            if (current_mask >> x) & 1:
                # Remove
                new_mask = current_mask ^ (1 << x)
                delta = -delta_3ap_remove(current_mask, x)
                # Always valid to remove
                if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                    current_mask = new_mask
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                        last_improve = time.time()
            else:
                # Add
                if has_4ap_with_mask(current_mask, x):
                    continue
                new_mask = current_mask | (1 << x)
                delta = delta_3ap_add(current_mask, x)
                if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                    current_mask = new_mask
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                        last_improve = time.time()
        else:
            # Swap
            present = [i for i in range(N) if (current_mask >> i) & 1]
            absent = [i for i in range(N) if not (current_mask >> i) & 1]
            if not present or not absent:
                continue
            x = random.choice(present)
            y = random.choice(absent)
            
            mask_without_x = current_mask ^ (1 << x)
            if has_4ap_with_mask(mask_without_x, y):
                continue
            
            new_mask = mask_without_x | (1 << y)
            loss = delta_3ap_remove(current_mask, x)
            gain = delta_3ap_add(mask_without_x, y)
            delta = gain - loss
            
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                current_mask = new_mask
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    last_improve = time.time()
        
        # If stuck for a long time, do a quick hill climb to get back to a local optimum
        if time.time() - last_improve > 2.0:
            current_mask, current_score = local_search_hill_climb(current_mask, current_score, timeout=0.5)
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
            last_improve = time.time()

    # Final polish: Hill climb the best found
    final_mask, final_score = local_search_hill_climb(best_mask, best_score, timeout=0.5)
    if final_score > best_score:
        best_score = final_score
        best_mask = final_mask

    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}, Score: {count_3aps(sum(1 << (x-1) for x in res))}")
