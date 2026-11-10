import time
import numpy as np
import random

def _compute_max_discrepancy(signs):
    """
    Compute the maximum absolute signed sum over any arithmetic progression.
    """
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1  # Singletons
    
    for d in range(1, n + 1):
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
                idx = np.argmax(abs_sums)
                best_len = idx + 1
                
    return max_disc, best_start, best_step, best_len

def _greedy_construct(n):
    """
    Greedy construction: at each step, choose the sign that minimizes the maximum
    absolute prefix sum over all APs ending at the current position.
    """
    signs = [0] * n
    
    # We maintain for each possible AP ending at current index i, the current sum.
    # An AP ending at i is defined by (start, step) such that i = start + k*step for some k >= 0.
    # Equivalently, for each divisor-like structure, we can track the sums.
    
    # A more efficient way:
    # For each position i, the APs ending at i are those where i is the last element.
    # If the AP is (a, d, L), then i = a + (L-1)d.
    # The sum is sum_{k=0}^{L-1} signs[a + k*d].
    
    # Let's just compute the change in discrepancy if we set signs[i] to +1 vs -1.
    # But that's expensive.
    
    # Alternative greedy:
    # For each i, consider all APs that end at i.
    # For each such AP, we know the sum of the previous elements (since they are already set).
    # Let S_prev be the sum of the AP without the last element.
    # If we set signs[i] = +1, the new sum is S_prev + 1.
    # If we set signs[i] = -1, the new sum is S_prev - 1.
    # We want to minimize the maximum absolute value of these new sums, but also consider
    # the effect on other APs? No, only APs ending at i are affected by signs[i].
    # Wait, signs[i] is only the last element of APs ending at i. It is not part of any AP
    # that ends at a later index yet, and it is not part of any AP that ended earlier.
    # So the only APs whose sum changes are those ending at i.
    
    # However, we also need to consider that the "prefix sums" of longer APs that will end later
    # depend on signs[i]. But for the greedy choice at step i, we only care about the discrepancy
    # of APs that are "complete" or "currently active".
    # Actually, the discrepancy is defined on the final sequence.
    # But a common heuristic is to minimize the maximum absolute sum of APs ending at the current step.
    
    # Let's implement this:
    # For each i from 0 to n-1:
    #   Find all APs ending at i.
    #   For each such AP, compute the sum of the previous elements.
    #   Let pos_sum = sum of (S_prev + 1) for all such APs? No.
    #   We want to choose sign s in {+1, -1} that minimizes max(|S_prev + s|) over all APs ending at i.
    
    # How to find all APs ending at i?
    # An AP ending at i has form (start, step, length) such that i = start + (length-1)*step.
    # This is equivalent to: for each step d, and for each start a such that i >= a and (i-a) % d == 0,
    # the AP (a, d, (i-a)//d + 1) ends at i.
    # This is too many to iterate over naively for each i.
    
    # Better:
    # For a fixed i, the APs ending at i are in 1-1 correspondence with pairs (d, a) where
    # a = i - k*d for some k >= 0, and a >= 0.
    # Actually, for each d, the possible starts a are i, i-d, i-2d, ... down to 0 or 1.
    # So for each d, there are floor(i/d) + 1 APs ending at i.
    
    # Let's just compute the sum for each such AP.
    # But that's O(i * log i) per step? No, sum_{d=1}^i floor(i/d) ~ i log i.
    # Total complexity ~ sum_{i=1}^n i log i ~ n^2 log n. For n=256, this is ~ 256^2 * 8 = 500k operations.
    # This is very fast.
    
    for i in range(n):
        best_sign = 1
        best_max_abs = float('inf')
        
        # Evaluate +1
        max_abs_pos = 0
        # Evaluate -1
        max_abs_neg = 0
        
        for d in range(1, i + 1):
            # APs ending at i with step d
            # Starts: i, i-d, i-2d, ...
            # For each start a, the AP is (a, d, L) where L = (i-a)//d + 1
            # The sum of the previous elements is sum(signs[a + k*d] for k=0 to L-2)
            # This is the sum of the AP (a, d, L-1) if L > 1, else 0.
            
            # We can compute this sum by iterating backwards from i-d to a.
            # But we can be smarter.
            # Let's just compute the sum for each AP ending at i.
            
            a = i
            while a >= 0:
                # AP starts at a, step d, ends at i.
                # Length L = (i - a) // d + 1
                # Sum of previous elements:
                if a == i:
                    prev_sum = 0
                else:
                    prev_sum = 0
                    # Sum signs[a + k*d] for k=0 to (i-a)//d - 1
                    # This is the same as the sum of the AP (a, d, L-1)
                    # We can compute this by a loop, but that's slow.
                    # Since we are doing this greedily, maybe we can maintain a table?
                    # No, let's just compute it directly for now and see if it's fast enough.
                    # n=256 is small.
                    curr = a
                    while curr < i:
                        prev_sum += signs[curr]
                        curr += d
                
                s_pos = prev_sum + 1
                s_neg = prev_sum - 1
                
                if abs(s_pos) > max_abs_pos:
                    max_abs_pos = abs(s_pos)
                if abs(s_neg) > max_abs_neg:
                    max_abs_neg = abs(s_neg)
                
                a -= d
        
        if max_abs_pos <= max_abs_neg:
            best_sign = 1
            best_max_abs = max_abs_pos
        else:
            best_sign = -1
            best_max_abs = max_abs_neg
            
        signs[i] = best_sign
        
    return signs

def construct():
    n = 256
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    # Initial guess: Greedy construction
    signs = _greedy_construct(n)
    
    current_disc = _compute_max_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    rng = random.Random(42)
    
    # Local Search with Simulated Annealing-like restarts
    no_improve_count = 0
    max_no_improve = 100
    temperature = 1.0
    
    while time.time() - start_time < deadline:
        # Find the worst AP
        disc, start, step, length = _find_worst_ap(signs)
        
        if disc < best_disc:
            best_disc = disc
            best_signs = signs.copy()
            no_improve_count = 0
            temperature = 1.0
        else:
            no_improve_count += 1
            temperature = max(0.1, temperature * 0.99)
            
        # If we've been stuck for a while, try a random flip or restart
        if no_improve_count > max_no_improve:
            # Try a few random flips
            for _ in range(10):
                idx = rng.randint(0, n - 1)
                signs[idx] = -signs[idx]
                new_disc = _compute_max_discrepancy(signs)
                if new_disc < disc:
                    break
                else:
                    signs[idx] = -signs[idx] # revert
            no_improve_count = 0
            continue
            
        # The worst AP is indices: start, start+step, ..., start+(length-1)*step
        ap_indices = list(range(start, start + length * step, step))
        current_sum = sum(signs[i] for i in ap_indices)
        
        # Find candidates in this AP that have the same sign as current_sum
        target_sign = 1 if current_sum > 0 else -1
        candidates = [i for i in ap_indices if signs[i] == target_sign]
        
        if candidates:
            # Pick a random candidate
            idx = rng.choice(candidates)
            signs[idx] = -signs[idx]
            
            # Check global disc
            new_disc = _compute_max_discrepancy(signs)
            if new_disc > disc:
                # Revert
                signs[idx] = -signs[idx]
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
