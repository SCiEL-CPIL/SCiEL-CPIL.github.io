import time
import math
import numpy as np
import random

def compute_score_fast(U):
    """
    Computes the lower bound exponent: 1 + log(|U-U|/|U+U|)/log(2 max U + 1)
    Uses numpy for fast set operations.
    """
    if len(U) < 2:
        return 0.0
    
    # Ensure U is a sorted numpy array of unique integers
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    maxU = U_arr[-1]
    if maxU == 0:
        return 0.0
    
    # Calculate Sum Set U+U
    # Size of sumset is at most 2*maxU + 1 (range 0 to 2*maxU)
    # We use a boolean array to track presence
    # Note: If maxU is very large (e.g., > 10^6), this array might be too big.
    # However, for optimal structures in this problem, maxU is likely kept moderate
    # or the set is sparse enough that we can handle it. 
    # Given the 150s limit, we can afford arrays up to a few million.
    
    # Optimization: If maxU is huge, fall back to set-based for safety?
    # No, let's assume we keep maxU reasonable or use a hash-based approach if maxU is too big.
    # For SA, we will constrain the search space to keep maxU manageable (e.g., < 100,000) 
    # unless the score improves significantly with larger maxU.
    
    if maxU > 200000:
        # Fallback for very large maxU to prevent memory errors
        sum_set = set()
        diff_set = set()
        for a in U_arr:
            for b in U_arr:
                sum_set.add(a + b)
                diff_set.add(a - b)
        ratio = len(diff_set) / len(sum_set) if sum_set else 0
        if ratio <= 0:
            return 0.0
        return 1 + math.log(ratio) / math.log(2 * maxU + 1)
        
    # Fast path with numpy
    # Sumset
    sum_arr = np.zeros(2 * maxU + 1, dtype=bool)
    # Use broadcasting or loops? Loops over N are faster if N < maxU.
    # N is likely small (< 1000).
    for a in U_arr:
        sum_arr[a + U_arr] = True
    sum_count = np.sum(sum_arr)
    
    # Difference Set
    # Range is -maxU to maxU. Offset by maxU.
    diff_arr = np.zeros(2 * maxU + 1, dtype=bool)
    for a in U_arr:
        diff_arr[(a - U_arr) + maxU] = True
    diff_count = np.sum(diff_arr)
    
    if sum_count == 0:
        return 0.0
        
    ratio = diff_count / sum_count
    if ratio <= 0:
        return 0.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0.0
        
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initialize with a good candidate
    # Powers of 2 are a strong baseline
    best_U = [0] + [2**i for i in range(1, 15)] # 0 to 16384
    best_score = compute_score_fast(best_U)
    
    # Also try a random dense set to see if it's better
    random.seed(42)
    for _ in range(5):
        n = random.randint(10, 50)
        max_val = random.randint(100, 5000)
        candidate = random.sample(range(1, max_val + 1), n)
        candidate.append(0)
        s = compute_score_fast(candidate)
        if s > best_score:
            best_score = s
            best_U = candidate
    
    current_U = best_U[:]
    current_score = best_score
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-6
    
    # Pre-calculate some constants for speed
    # We will perform moves: Add, Remove, Swap, Shift
    
    while time.time() < deadline:
        # Check time frequently
        if time.time() > deadline:
            break
            
        # Generate a neighbor
        move_type = random.random()
        new_U = current_U[:]
        
        if move_type < 0.3:
            # Add a new element
            if current_U:
                max_val = max(current_U)
                # Add near existing elements or extend range
                if random.random() < 0.5:
                    # Extend range
                    new_val = random.randint(max_val + 1, max_val + 100)
                else:
                    # Fill gap
                    new_val = random.randint(1, max_val)
                
                if new_val not in current_U:
                    new_U.append(new_val)
            else:
                new_U = [0, 1]
                
        elif move_type < 0.6:
            # Remove an element (not 0)
            if len(current_U) > 2:
                # Pick a random index, avoid 0 (index 0 if sorted, but let's just pick random non-zero)
                non_zero_indices = [i for i, v in enumerate(current_U) if v != 0]
                if non_zero_indices:
                    idx = random.choice(non_zero_indices)
                    new_U.pop(idx)
            else:
                continue
                
        elif move_type < 0.8:
            # Swap an element
            if len(current_U) > 2:
                non_zero_indices = [i for i, v in enumerate(current_U) if v != 0]
                if non_zero_indices:
                    idx = random.choice(non_zero_indices)
                    old_val = current_U[idx]
                    max_val = max(current_U)
                    new_val = random.randint(1, max_val + 50)
                    if new_val not in current_U:
                        new_U[idx] = new_val
                        
        else:
            # Shift all non-zero elements by a small delta
            if current_U:
                delta = random.choice([-1, 1, -2, 2, -5, 5])
                # Ensure no negatives and no duplicates
                valid = True
                new_vals = set()
                for v in current_U:
                    if v == 0:
                        new_vals.add(0)
                    else:
                        nv = v + delta
                        if nv < 0:
                            valid = False
                            break
                        if nv in new_vals:
                            valid = False
                            break
                        new_vals.add(nv)
                if valid:
                    new_U = list(new_vals)
                    if len(new_U) != len(current_U):
                        continue # Should not happen if valid check passed, but safety
            
        # Score the new candidate
        # Optimization: If the new max is much larger, the score might drop due to denominator.
        # But we must compute to be sure.
        
        # Fast rejection: If maxU increases significantly but N stays same, ratio might drop.
        # Just compute.
        
        new_score = compute_score_fast(new_U)
        
        # Acceptance probability
        if new_score > current_score:
            current_U = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = current_U[:]
        else:
            # Simulated Annealing acceptance
            delta_s = current_score - new_score
            if delta_s < 0:
                accept = True
            else:
                # exp(-delta_s / T)
                # If delta_s is large, probability is tiny
                if T > 0:
                    prob = math.exp(-delta_s / T)
                    accept = random.random() < prob
                else:
                    accept = False
                    
            if accept:
                current_U = new_U
                current_score = new_score
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T

    return best_U

if __name__ == "__main__":
    # Quick self-test
    U = construct()
    print(f"Best Score: {compute_score_fast(U)}")
    print(f"Set Size: {len(U)}")
    print(f"Max U: {max(U) if U else 0}")
