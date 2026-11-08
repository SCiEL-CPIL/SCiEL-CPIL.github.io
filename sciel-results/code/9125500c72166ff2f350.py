"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def compute_score_fast(U):
    """
    Computes the lower bound exponent: 1 + log(|U-U|/|U+U|)/log(2*max(U) + 1)
    Optimized for speed using numpy where appropriate, or efficient set ops.
    """
    if len(U) < 2:
        return 0.0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    # For small N, set operations are very fast in Python
    # For larger N, we might need numpy, but SA usually keeps N smallish or moderate
    # Let's stick to sets for robustness and speed on small sets
    sum_set = set()
    diff_set = set()
    
    # Optimization: iterate over unique pairs? No, U+U and U-U include all pairs (u,v)
    # We can iterate u1, u2
    for u1 in U:
        for u2 in U:
            sum_set.add(u1 + u2)
            diff_set.add(u1 - u2)
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0.0
        
    log_ratio = math.log(size_diff / size_sum)
    log_range = math.log(2 * maxU + 1)
    
    return 1.0 + log_ratio / log_range

def construct():
    """
    Entry point. Returns a list of nonnegative integers containing 0.
    Uses Simulated Annealing over a binary vector representation of set membership.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for 120s limit
    
    # Define the search space bounds
    # Based on inspirations and typical set sizes for this problem,
    # let's bound the maximum value in the set to M_MAX.
    # If M is too large, log(2M+1) grows, reducing the score unless the ratio grows faster.
    # The ratio |U-U|/|U+U| is bounded by roughly 2 (since diffs are symmetric, sums are not).
    # Actually, for non-negative sets, |U-U| can be up to 2*|U|-1 (if all diffs distinct) 
    # and |U+U| can be up to |U|(|U|+1)/2.
    # Wait, |U-U| size is at most 2*max(U)-min(U)+1.
    # Let's pick a reasonable M_MAX. 
    # Inspiration 1 used brute force on small sets. 
    # Let's try M_MAX = 200 or 500. 
    # If we use a binary vector of length M_MAX+1, flipping a bit changes the set.
    
    M_MAX = 150  # Upper bound for values in U
    
    # Initial state: A random set or a known good structure
    # Let's start with a random set of moderate density
    def random_initial_set():
        # Ensure 0 is included
        mask = np.zeros(M_MAX + 1, dtype=int)
        mask[0] = 1
        # Add some random elements
        num_elements = random.randint(3, 15)
        candidates = list(range(1, M_MAX + 1))
        random.shuffle(candidates)
        for i in range(min(num_elements, len(candidates))):
            mask[candidates[i]] = 1
        return mask

    def mask_to_list(mask):
        return [i for i in range(M_MAX + 1) if mask[i] == 1]

    def score_from_mask(mask):
        U = mask_to_list(mask)
        if len(U) < 2:
            return -1000.0 # Penalty
        return compute_score_fast(U)

    # Initial mask
    current_mask = random_initial_set()
    current_score = score_from_mask(current_mask)
    
    best_mask = current_mask.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 0.001
    alpha = 0.995  # Cooling rate
    T = T_start
    
    # We need to handle the case where the set becomes too small or empty
    # The mask representation allows us to easily flip bits.
    
    while T > T_end:
        if time.time() > deadline:
            break
            
        # Generate a neighbor by flipping a random bit (except 0)
        # Sometimes flip multiple bits? No, single bit flip is standard.
        # We must ensure 0 is always present.
        
        # Pick a random index to flip
        # Weighted towards smaller numbers? Or uniform?
        # Uniform is fine.
        idx = random.randint(1, M_MAX)
        
        # Flip the bit
        old_val = current_mask[idx]
        current_mask[idx] = 1 - old_val
        
        # Calculate new score
        new_score = score_from_mask(current_mask)
        
        # Check if valid (len >= 2)
        # score_from_mask returns -1000 if invalid, so delta will be huge negative
        
        delta = new_score - current_score
        
        # Metropolis criterion
        if delta > 0:
            # Accept improvement
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask.copy()
        else:
            # Accept degradation with probability exp(-delta / T)
            # Note: delta is negative, so -delta/T is positive
            # If T is small, probability is low
            if random.random() < math.exp(-delta / T):
                current_score = new_score
            else:
                # Reject: revert
                current_mask[idx] = old_val
        
        # Cool down
        T *= alpha
        
        # Optional: Restart if stuck? 
        # Simple SA usually doesn't restart, but we can add a simple restart mechanism
        # if we detect no improvement for a long time.
        # For now, let's just run the cooling schedule.
        
        # To make it more effective, we might want to periodically reheat or restart
        # if the score hasn't improved for many steps.
        # Let's add a simple restart logic every 1000 steps if no improvement
        # But counting steps is easy.
        
        # Actually, let's just run until deadline.
        
    # Final check: ensure best_mask is valid
    best_U = mask_to_list(best_mask)
    
    # If the SA result is bad (e.g., score < 1.0), fall back to a known good candidate
    # The seed program had candidates like [0, 1, 3, 9, 27, 81]
    fallback_candidates = [
        [0, 1, 3, 9, 27, 81],
        [0, 1, 3, 9],
        [0, 1, 2, 4, 8, 16, 32],
        [0, 1, 4, 9, 16, 25] # Squares
    ]
    
    fallback_best = -1
    fallback_U = fallback_candidates[0]
    for c in fallback_candidates:
        s = compute_score_fast(c)
        if s > fallback_best:
            fallback_best = s
            fallback_U = c
            
    if best_score < fallback_best:
        return fallback_U
        
    return best_U

# EVOLVE-BLOCK-END
