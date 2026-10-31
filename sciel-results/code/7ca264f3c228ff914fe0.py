import time
import math
import numpy as np
import random

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 110.0  # Safety margin vs 150s kill

def _compute_sets_fft(u, max_val):
    """
    Compute |U+U| and |U-U| using FFT.
    u: list of nonnegative integers
    max_val: max(u)
    
    Returns: (size_sum, size_diff)
    """
    if not u:
        return 0, 0
    if len(u) == 1:
        return 1, 1
    
    M = max_val
    # N needs to be at least 2*M + 1 to avoid circular convolution aliasing for sums [0, 2M]
    # and differences [-M, M]
    N = 1
    while N < 2 * M + 1:
        N *= 2
    
    f = np.zeros(N, dtype=np.float64)
    for x in u:
        if 0 <= x < N:
            f[x] = 1.0
    
    F = np.fft.fft(f)
    
    # Sum set
    conv_sum = np.fft.ifft(F * F).real
    # Valid range for sums is [0, 2*M]
    size_sum = np.sum(conv_sum[:2 * M + 1] > 0.5)
    
    # Difference set
    # Autocorrelation r[k] = sum_x f[x] f[x+k]
    auto_diff = np.fft.ifft(F * np.conj(F)).real
    # r[k] for k in [1, M] corresponds to positive differences
    pos_diffs = np.sum(auto_diff[1:M + 1] > 0.5)
    size_diff = 1 + 2 * pos_diffs
    
    return int(size_sum), int(size_diff)

def _score(u):
    """
    Compute the score: 1 + log(|U-U|/|U+U|) / log(2*max U + 1)
    """
    if not u:
        return 0.0
    m = max(u)
    if m < 1:
        return 1.0
    
    size_sum, size_diff = _compute_sets_fft(u, m)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / float(size_sum)
    
    if ratio <= 0:
        return 0.0
        
    denom = math.log(2 * m + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def _greedy_construct(target_size, max_val):
    """
    Construct a set of size target_size by greedily adding the element
    that maximizes the score.
    """
    u = [0]
    if target_size <= 1:
        return u
    
    current_score = _score(u)
    used = {0}
    
    for _ in range(target_size - 1):
        best_candidate = None
        best_score_val = -np.inf
        
        # We can't just check all 0..max_val if max_val is large.
        # But for the constructive phase, we can limit the search space
        # or sample if max_val is huge.
        # Given the score denominator log(2M+1), small M is preferred.
        # Let's restrict candidates to [1, max_val] but sample if max_val > 500
        if max_val <= 1000:
            candidates = list(range(1, max_val + 1))
        else:
            candidates = random.sample(range(1, max_val + 1), min(500, max_val))
        
        # Shuffle to break ties randomly? No, we want deterministic best.
        # But we need to avoid O(N^2) if N is large.
        # For each candidate, we compute the new score.
        # This is expensive.
        # Optimization: We can approximate or use a heuristic.
        # However, for target_size < 50 and max_val < 200, it's feasible.
        
        for c in candidates:
            if c in used:
                continue
            
            new_u = list(u)
            new_u.append(c)
            new_u.sort()
            
            # Quick check: does this increase max too much?
            # The score penalizes large max.
            
            s = _score(new_u)
            
            if s > best_score_val:
                best_score_val = s
                best_candidate = c
        
        if best_candidate is None:
            # Fallback: pick a random unused
            unused = [x for x in range(1, max_val + 1) if x not in used]
            if not unused:
                break
            best_candidate = random.choice(unused)
        
        u.append(best_candidate)
        used.add(best_candidate)
        u.sort()
        current_score = best_score_val
        
        if time.time() > _DEADLINE - 10:
            break
            
    return u

def _greedy_rebuild(u, max_val, num_restarts=10):
    """
    Try to improve the set by removing one element and greedily re-adding
    the best possible element, or simply rebuilding from scratch with
    different initial conditions.
    """
    best_u = list(u)
    best_s = _score(u)
    
    for r in range(num_restarts):
        if time.time() > _DEADLINE - 5:
            break
            
        # Create a perturbed version: remove a random element
        if len(u) > 1:
            idx = random.randint(1, len(u) - 1) # don't remove 0
            perturbed = list(u)
            perturbed.pop(idx)
        else:
            perturbed = [0]
            
        # Now greedily add elements to restore size or exceed it
        # Target size: len(u) or slightly larger
        target_size = len(u) + random.randint(-1, 2)
        if target_size < 2: target_size = 2
        
        # Greedy add from perturbed state
        current_u = perturbed
        used = set(current_u)
        
        for _ in range(target_size - len(current_u)):
            best_c = None
            best_s_c = -np.inf
            
            # Sample candidates
            if max_val <= 1000:
                candidates = [x for x in range(1, max_val + 1) if x not in used]
            else:
                candidates = [x for x in random.sample(range(1, max_val + 1), min(200, max_val)) if x not in used]
                
            if not candidates:
                break
                
            for c in candidates:
                new_u = list(current_u)
                new_u.append(c)
                new_u.sort()
                s = _score(new_u)
                if s > best_s_c:
                    best_s_c = s
                    best_c = c
                    
            if best_c is None:
                break
                
            current_u.append(best_c)
            used.add(best_c)
            current_u.sort()
            
        if _score(current_u) > best_s:
            best_s = _score(current_u)
            best_u = current_u
            
    return best_u, best_s

def construct():
    global _DEADLINE
    _DEADLINE = max(time.time(), _START + 110.0)
    
    random.seed(42)
    
    best_u = [0, 1, 2]
    best_s = _score(best_u)
    
    # Strategy:
    # The score is 1 + log(|U-U|/|U+U|) / log(2*max U + 1).
    # To maximize, we want high ratio and small max.
    # Greedy construction tends to find locally optimal sets for a fixed size.
    # We should explore a range of sizes and max values.
    
    # Phase 1: Greedy construction for various sizes
    # Small sizes often yield better ratios if they are "sparse" enough
    # but not too sparse (need enough differences).
    
    sizes_to_try = [3, 4, 5, 6, 8, 10, 15, 20, 30]
    
    for n in sizes_to_try:
        if time.time() > _DEADLINE - 20:
            break
            
        # Try different max_val ranges
        # For small n, max_val can be small.
        # For larger n, max_val needs to be larger to fit n elements.
        # But large max_val hurts the denominator.
        # Sweet spot: max_val ~ n * k where k is small.
        
        max_vals_to_try = [n * 2, n * 3, n * 5, 50, 100]
        max_vals_to_try = sorted(set([m for m in max_vals_to_try if m >= n]))
        
        for M in max_vals_to_try:
            if time.time() > _DEADLINE - 10:
                break
                
            # Greedy construct
            u_cand = _greedy_construct(n, M)
            
            if not u_cand:
                continue
                
            s_cand = _score(u_cand)
            
            if s_cand > best_s:
                best_s = s_cand
                best_u = u_cand
                
            # Try to improve via greedy rebuild/perturbation
            if time.time() < _DEADLINE - 5:
                u_improved, s_improved = _greedy_rebuild(u_cand, M, num_restarts=5)
                if s_improved > best_s:
                    best_s = s_improved
                    best_u = u_improved

    # Final polish: Try to optimize the best found set further
    # Maybe allow max_val to grow slightly if it helps ratio significantly?
    # Or shrink max_val by removing large elements?
    
    if time.time() < _DEADLINE - 2:
        # Try removing the largest element if it doesn't hurt ratio too much
        if len(best_u) > 1:
            u_trial = list(best_u)
            u_trial.pop() # remove largest
            s_trial = _score(u_trial)
            if s_trial > best_s:
                best_s = s_trial
                best_u = u_trial

    # Ensure 0 is in best_u and all are non-negative integers
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
        
    return best_u
# EVOLVE-BLOCK-END
