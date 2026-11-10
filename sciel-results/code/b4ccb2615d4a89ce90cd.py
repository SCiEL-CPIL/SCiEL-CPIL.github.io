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

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    seq = []
    for i in range(n):
        c = 0
        x = i
        while x:
            c += x & 1
            x >>= 1
        seq.append(1 if c % 2 == 0 else -1)
    return seq

def _greedy_construct(n, rng):
    """
    Construct signs greedily. At each step, pick the sign that minimizes the 
    maximum discrepancy of all APs ending at the current position.
    This is O(n^3) roughly, which for n=256 is ~16M operations, feasible.
    Actually, for each new sign, we only need to check APs that include this new sign.
    An AP includes index i if i = a + k*d for some k.
    For a fixed i, the APs ending at i are those with d dividing (i-a) for some a.
    Actually, it's easier to just check all APs that have i as the last element.
    For each d in 1..i:
       For each start a such that a + k*d = i for some k >= 0.
       This is equivalent to a = i - k*d >= 0.
       So k can range from 0 to floor(i/d).
       The AP is (i-k*d, ..., i).
       We need to update the sum for this AP.
    """
    signs = [0] * n
    # To make it faster, we can maintain a dictionary of current sums for all active APs?
    # No, let's just recompute the relevant sums.
    
    for i in range(n):
        best_sign = 1
        best_disc = 10**9
        
        for s in [1, -1]:
            signs[i] = s
            # Check all APs ending at i
            max_d = i + 1
            current_max = 0
            # APs ending at i:
            # Step d, last term i.
            # Terms: i, i-d, i-2d, ... >= 0
            for d in range(1, max_d + 1):
                # Sum of signs[i], signs[i-d], ...
                # This is a prefix of the sequence r, r+d, ... where r = i % d? No.
                # The sequence is i, i-d, i-2d...
                # This is the reverse of a standard AP start.
                # Let's just sum them.
                s_val = 0
                j = i
                while j >= 0:
                    s_val += signs[j]
                    j -= d
                    if abs(s_val) > current_max:
                        current_max = abs(s_val)
                        # Early termination if we already exceed best_disc?
                        # If current_max > best_disc, we can break this d loop?
                        # No, we need to compare s=1 and s=-1.
                        # But if current_max > best_disc, this sign is worse.
                        # We can break the while loop for this d.
                        pass
                
                # Optimization: We can break the d-loop if current_max >= best_disc
                # But we need to finish the check for this sign to know if it's better.
                # Actually, if current_max > best_disc, we can stop checking this sign immediately.
                if current_max > best_disc:
                    break
            
            if current_max < best_disc:
                best_disc = current_max
                best_sign = s
        
        signs[i] = best_sign
        
    return signs

def construct():
    n = 256
    start_time = time.time()
    deadline = 58.0  # Safety margin
    
    # Start with Greedy Construction
    # Greedy is deterministic if we don't use rng, but let's use a fixed rng for reproducibility if needed.
    # Actually greedy is deterministic.
    signs = _greedy_construct(n, None)
    
    current_disc = _compute_max_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    rng = random.Random(42)
    
    # Local Search with Simulated Annealing style or just Hill Climbing with Restarts
    # We will use Hill Climbing first. If stuck, random restart from best.
    
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() - start_time < deadline:
        # Find the worst AP
        disc, start, step, length = _find_worst_ap(signs)
        
        if disc < best_disc:
            best_disc = disc
            best_signs = signs.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # If stuck, restart from best with random perturbation
        if no_improve_count > max_no_improve:
            signs = best_signs.copy()
            # Perturb a few random positions
            for _ in range(10):
                idx = rng.randint(0, n - 1)
                signs[idx] = -signs[idx]
            no_improve_count = 0
            continue
            
        # The worst AP is indices: start, start+step, ..., start+(length-1)*step
        ap_indices = list(range(start, start + length * step, step))
        current_sum = sum(signs[i] for i in ap_indices)
        
        # Find candidates in this AP that have the same sign as current_sum
        # If current_sum > 0, look for 1s. If current_sum < 0, look for -1s.
        target_sign = 1 if current_sum > 0 else -1
        candidates = [i for i in ap_indices if signs[i] == target_sign]
        
        if candidates:
            # Pick a random candidate
            idx = rng.choice(candidates)
            old_sign = signs[idx]
            signs[idx] = -signs[idx]
            
            new_disc = _compute_max_discrepancy(signs)
            if new_disc > disc:
                # Revert
                signs[idx] = old_sign
                no_improve_count += 1
            else:
                # Keep the flip
                pass
        else:
            # Should not happen
            pass

    return best_signs
