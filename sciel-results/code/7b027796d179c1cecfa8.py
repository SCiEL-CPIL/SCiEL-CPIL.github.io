import time
import numpy as np
import random

def _compute_max_discrepancy(signs):
    """
    Compute the maximum absolute signed sum over any arithmetic progression.
    An AP is defined by start a, step d, and length L.
    The sum is sum_{k=0}^{L-1} signs[a + k*d].
    We need to check all prefixes of all maximal APs for all steps d and starts a.
    This is equivalent to:
    For each step d:
      For each start r in [0, d):
        Consider the sequence signs[r], signs[r+d], signs[r+2d], ...
        Compute the maximum absolute prefix sum of this sequence.
    """
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1  # Singletons
    
    # We only need to check steps d up to n.
    # For a given d, the starts are r in [0, d).
    for d in range(1, n + 1):
        # For this step d, we have d sequences starting at 0, 1, ..., d-1.
        # We can process them together or individually.
        # To be fast, we can create a 2D array.
        # Max length is ceil(n/d).
        max_len = (n + d - 1) // d
        
        # Create a matrix of shape (d, max_len)
        # Initialize with 0s to handle uneven lengths
        # However, cumsum on padded zeros is fine as long as we ignore the tail
        # Or we can just loop r, which is at most 256 times.
        # Let's try the loop first, it's simpler and likely fast enough for N=256.
        # 256 steps * 256 starts = 65536 cumsums.
        # Each cumsum is on an array of size <= 256.
        # This might be ~1-5ms in Python/NumPy.
        
        for r in range(d):
            if r >= n:
                break
            subseq = arr[r::d]
            if len(subseq) == 0:
                continue
            prefix_sums = np.cumsum(subseq)
            current_max = np.max(np.abs(prefix_sums))
            if current_max > max_disc:
                max_disc = int(current_max)
                
    return max_disc

def _find_worst_ap(signs):
    """
    Returns (max_disc, start, step, length) for one of the APs achieving max_disc.
    """
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1
    best_start = 0
    best_step = 1
    best_len = 1
    
    for d in range(1, n + 1):
        for r in range(d):
            if r >= n:
                break
            subseq = arr[r::d]
            if len(subseq) == 0:
                continue
            prefix_sums = np.cumsum(subseq)
            abs_sums = np.abs(prefix_sums)
            current_max = np.max(abs_sums)
            if current_max > max_disc:
                max_disc = int(current_max)
                best_step = d
                best_start = r
                # Find the length (index in prefix_sums + 1) that achieves this max
                # There might be multiple, pick the first one
                idx = np.argmax(abs_sums)
                best_len = idx + 1
                
    return max_disc, best_start, best_step, best_len

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    seq = []
    for i in range(n):
        # Count number of 1s in binary representation
        # Python 3.1+ has bin(i).count('1')
        # Or use bit_count() in 3.10+
        # Let's use a fast method
        c = 0
        x = i
        while x:
            c += x & 1
            x >>= 1
        seq.append(1 if c % 2 == 0 else -1)
    return seq

def construct():
    n = 256
    start_time = time.time()
    deadline = 58.0  # Safety margin
    
    # Initial guess: Thue-Morse
    signs = _thue_morse(n)
    
    current_disc = _compute_max_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    rng = random.Random(42)
    
    # Local Search
    # We will try to reduce the discrepancy by flipping signs in the worst AP.
    # If we get stuck (no improvement for many steps), we can do a random flip to escape local minima.
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < deadline:
        # Find the worst AP
        disc, start, step, length = _find_worst_ap(signs)
        
        if disc < best_disc:
            best_disc = disc
            best_signs = signs.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # If we've been stuck for a while, try a random flip
        if no_improve_count > max_no_improve:
            idx = rng.randint(0, n - 1)
            signs[idx] = -signs[idx]
            no_improve_count = 0
            continue
            
        # The worst AP is indices: start, start+step, ..., start+(length-1)*step
        # We want to flip a sign in this AP to reduce the sum.
        # The current sum is S = sum(signs[start + k*step] for k in range(length))
        # abs(S) == disc.
        # If we flip a sign x_i, the new sum is S - 2*x_i.
        # We want |S - 2*x_i| < |S|.
        # This is always true if x_i has the same sign as S.
        # So, if S > 0, we should flip a +1 to -1.
        # If S < 0, we should flip a -1 to +1.
        
        # Calculate the current sum of this specific AP
        ap_indices = list(range(start, start + length * step, step))
        current_sum = sum(signs[i] for i in ap_indices)
        
        # Find candidates in this AP that have the same sign as current_sum
        # If current_sum > 0, look for 1s. If current_sum < 0, look for -1s.
        target_sign = 1 if current_sum > 0 else -1
        candidates = [i for i in ap_indices if signs[i] == target_sign]
        
        if candidates:
            # Pick a random candidate
            idx = rng.choice(candidates)
            signs[idx] = -signs[idx]
            # We don't immediately revert. We check in the next loop.
            # But we should check if this move actually improved the global discrepancy.
            # It might improve this AP but worsen another.
            # For now, accept it. If global disc increases, we'll revert later or get stuck.
            # Better: Check global disc.
            new_disc = _compute_max_discrepancy(signs)
            if new_disc > disc:
                # Revert
                signs[idx] = -signs[idx]
                # If we can't improve by flipping in the worst AP, try a random flip nearby
                # or just count as no improvement
                no_improve_count += 1
            else:
                # Keep the flip
                pass
        else:
            # No candidates? This means all signs in the AP are opposite to the sum.
            # This is impossible for a sum of non-zero magnitude unless length is 0?
            # If sum > 0, there must be at least one +1.
            # So candidates should never be empty.
            pass

    return best_signs
