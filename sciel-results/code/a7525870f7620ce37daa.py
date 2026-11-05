"""Low discrepancy on all arithmetic progressions - Greedy Worst-AP Repair with Vectorized Evaluation."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _get_max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    Returns (max_disc, worst_start, worst_step, worst_len)
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int16)
    
    max_abs = 0
    best_start = 0
    best_step = 1
    best_len = 1
    
    for d in range(1, n + 1):
        # For a given step d, starts are 0..d-1 (but must be < n)
        # We check all valid starts
        for i in range(min(d, n)):
            # Sequence is s[i], s[i+d], s[i+2d], ...
            # Length L is the number of terms
            # We need max |prefix_sum| for this sequence
            # s[i:n:d] gives the sequence
            seq = s[i:n:d]
            if len(seq) == 0:
                continue
            
            cumsum = np.cumsum(seq)
            abs_cumsum = np.abs(cumsum)
            current_max = np.max(abs_cumsum)
            
            if current_max > max_abs:
                max_abs = current_max
                best_start = i
                best_step = d
                # Find the length L corresponding to this max
                # argmax gives the index in cumsum, so length = index + 1
                best_len = np.argmax(abs_cumsum) + 1
    
    return int(max_abs), best_start, best_step, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    n = 256
    rng = np.random.default_rng(42)
    
    # Initial random signs
    signs = rng.choice([-1, 1], size=n).astype(np.int16)
    
    # Evaluate initial
    current_disc, _, _, _ = _get_max_ap_discrepancy(signs)
    best_signs = signs.copy()
    best_disc = current_disc
    
    # Greedy Worst-AP Repair Loop
    # We repeatedly find the AP with max discrepancy and try to fix it.
    
    while time.time() < deadline:
        disc, start, step, length = _get_max_ap_discrepancy(signs)
        
        if disc < best_disc:
            best_disc = disc
            best_signs = signs.copy()
        
        # If we reach 0 or 1 (theoretical minimum is likely 1 for n=256? 
        # Actually, for n=256, can we get 1? 
        # Sum of all 256 signs is even. 
        # APs of length 1 have sum +/-1. So min possible max abs is 1.
        if best_disc <= 1:
            break
            
        # Identify the specific AP: indices start, start+step, ..., start+(length-1)*step
        # Current sum S = sum(signs[idx] for idx in AP)
        # We want to flip a sign in this AP to reduce |S|.
        # Flipping sign at index j changes S by -2 * signs[j].
        # New S' = S - 2 * signs[j].
        # We want to minimize |S'|.
        # Ideally, we pick j such that signs[j] has the same sign as S.
        # Then S' = S - 2*sign(S) = S - 2*sign(S).
        # If S > 0, we want signs[j] = 1. Then S' = S - 2.
        # If S < 0, we want signs[j] = -1. Then S' = S + 2.
        
        ap_indices = np.arange(start, start + length * step, step)
        ap_signs = signs[ap_indices]
        current_sum = np.sum(ap_signs)
        
        if current_sum == 0:
            # This shouldn't happen if disc > 0 and this is the max, 
            # unless there are ties and we picked one with 0? 
            # _get_max_ap_discrepancy returns max abs. If max abs is 0, disc=0.
            # If disc > 0, there is an AP with non-zero sum.
            # If we picked one with 0, it's not the max. 
            # But due to ties, we might pick a 0-sum AP if max_abs was 0? 
            # No, if max_abs > 0, we picked an AP with abs sum == max_abs.
            # So current_sum should be non-zero.
            pass
            
        # Find indices in the AP where sign matches the sum's sign
        # If current_sum > 0, we want to flip a +1 to -1.
        # If current_sum < 0, we want to flip a -1 to +1.
        
        target_sign = 1 if current_sum > 0 else -1
        
        # Find positions in ap_indices where ap_signs == target_sign
        candidates = np.where(ap_signs == target_sign)[0]
        
        if len(candidates) > 0:
            # Pick one randomly to avoid cycling? Or pick the one that causes least collateral damage?
            # For now, random pick among candidates.
            # Actually, picking the one that is in the "middle" might be better? 
            # Or just random.
            idx_to_flip = candidates[rng.integers(0, len(candidates))]
            global_idx = ap_indices[idx_to_flip]
            
            signs[global_idx] = -signs[global_idx]
        else:
            # No candidate with matching sign? 
            # This implies all signs in the AP are opposite to the sum's sign?
            # Impossible. Sum is sum of signs. If all are -1, sum is negative. 
            # If sum is positive, at least one must be +1.
            # So this branch should not be taken.
            # But just in case, flip a random one.
            global_idx = ap_indices[rng.integers(0, length)]
            signs[global_idx] = -signs[global_idx]
            
    return best_signs.tolist()

# EVOLVE-BLOCK-END
