"""Direct heuristic construction for low discrepancy on arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """Compute max absolute signed sum over any arithmetic progression in [1..n]."""
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_abs = 0
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                continue
            m = (n - 1 - r) // d + 1
            if m <= 0:
                continue
            indices = np.arange(r, n, d)
            sub = signs_arr[indices]
            cum = np.cumsum(sub)
            max_sum = 0
            min_sum = 0
            running_min = 0
            running_max = 0
            for i in range(m):
                val = cum[i]
                if val < running_min:
                    running_min = val
                if val > running_max:
                    running_max = val
                diff_max = val - running_min
                diff_min = val - running_max
                if diff_max > max_sum:
                    max_sum = diff_max
                if diff_min < min_sum:
                    min_sum = diff_min
            current_max = max(abs(max_sum), abs(min_sum))
            if current_max > max_abs:
                max_abs = current_max
    return max_abs

def _fast_max_ap_discrepancy(signs, max_step=None):
    """Faster: only check steps up to max_step for quick evaluation."""
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_abs = 0
    if max_step is None:
        max_step = n - 1
    max_step = min(max_step, n - 1)
    for d in range(1, max_step + 1):
        for r in range(d):
            if r >= n:
                continue
            m = (n - 1 - r) // d + 1
            if m <= 0:
                continue
            indices = np.arange(r, n, d)
            sub = signs_arr[indices]
            cum = np.cumsum(sub)
            max_sum = 0
            min_sum = 0
            running_min = 0
            running_max = 0
            for i in range(m):
                val = cum[i]
                if val < running_min:
                    running_min = val
                if val > running_max:
                    running_max = val
                diff_max = val - running_min
                diff_min = val - running_max
                if diff_max > max_sum:
                    max_sum = diff_max
                if diff_min < min_sum:
                    min_sum = diff_min
            current_max = max(abs(max_sum), abs(min_sum))
            if current_max > max_abs:
                max_abs = current_max
    return max_abs

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    signs = []
    for i in range(n):
        cnt = bin(i).count('1')
        signs.append(1 if cnt % 2 == 0 else -1)
    return signs

def _greedy_balance(n, seed_val):
    """Greedy: assign each position to minimize worst-case AP sum increase."""
    rng = random.Random(seed_val)
    signs = [0] * n
    
    # Precompute for each position, the APs that could be affected
    # For efficiency, we track per (d, r) the current sum
    
    # Assign positions in order
    for i in range(n):
        # For each candidate sign (+1 or -1), compute the max increase
        # in discrepancy across all APs containing position i
        best_sign = 1
        best_worst = float('inf')
        
        for s in (1, -1):
            # Temporarily assign
            signs[i] = s
            # Check all APs containing position i
            # AP (d, r) contains i if i % d == r and i >= r
            max_inc = 0
            for d in range(1, i + 1):
                r = i % d
                if r > i:
                    continue
                # The AP starting at r with step d: positions r, r+d, r+2d, ...
                # Position i is at index (i - r) // d in this AP
                # We need the sum of the AP up to and including i
                # But also any sub-AP
                # For speed, just check the full AP sum so far
                m = (i - r) // d + 1
                if m <= 0:
                    continue
                indices = list(range(r, i + 1, d))
                s_sum = 0
                for idx in indices:
                    if signs[idx] != 0:
                        s_sum += signs[idx]
                # The max sub-AP sum could be higher, but as approximation
                # use |s_sum| as a lower bound
                if abs(s_sum) > max_inc:
                    max_inc = abs(s_sum)
            
            if max_inc < best_worst:
                best_worst = max_inc
                best_sign = s
        
        signs[i] = best_sign
    
    return signs

def _walsh_hadamard(n):
    """Walsh-Hadamard based construction."""
    # Use the first n rows of the Walsh matrix
    signs = []
    for i in range(n):
        # Walsh function: (-1)^(popcount(i & j)) for some j
        # Use j = i for a diagonal-like construction
        cnt = bin(i & (i ^ (i >> 1))).count('1')
        signs.append(1 if cnt % 2 == 0 else -1)
    return signs

def _multi_scale(n):
    """Multi-scale construction combining different frequencies."""
    signs = []
    for i in range(n):
        # Combine Thue-Morse with a slower oscillation
        tm_cnt = bin(i).count('1')
        tm = 1 if tm_cnt % 2 == 0 else -1
        
        # Add a slow component
        slow = 1 if (i // 8) % 2 == 0 else -1
        
        # Combine: if they agree, use that; if disagree, use TM
        if tm == slow:
            signs.append(tm)
        else:
            # Use a tiebreaker based on position
            signs.append(1 if (i % 4) < 2 else -1)
    return signs

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 256
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Thue-Morse
    candidates.append(_thue_morse(n))
    
    # 2. Multi-scale
    candidates.append(_multi_scale(n))
    
    # 3. Walsh
    candidates.append(_walsh_hadamard(n))
    
    # 4. Greedy with different seeds
    for seed in [1, 7, 13, 42, 99]:
        if time.time() > deadline - 30:
            break
        candidates.append(_greedy_balance(n, seed))
    
    # 5. Random
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        rng = random.Random(seed)
        rand = [rng.choice((-1, 1)) for _ in range(n)]
        candidates.append(rand)
    
    # 6. Block patterns
    for block in [2, 4, 8, 16, 32, 64]:
        blk = []
        for i in range(n):
            b = i // block
            w = i % block
            if b % 2 == 0:
                blk.append(1 if w < block // 2 else -1)
            else:
                blk.append(-1 if w < block // 2 else 1)
        candidates.append(blk)
    
    # 7. Sine-based
    for freq in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        sine = [1 if math.sin(2 * math.pi * freq * i / n) >= 0 else -1 for i in range(n)]
        candidates.append(sine)
    
    # Evaluate all candidates with full discrepancy
    best_signs = candidates[0][:]
    best_disc = float('inf')
    
    for c in candidates:
        if time.time() > deadline - 50:
            break
        disc = _max_ap_discrepancy(c)
        if disc < best_disc:
            best_disc = disc
            best_signs = c[:]
    
    # Aggressive local search
    improved = True
    no_improve_count = 0
    max_no_improve = 5
    
    while improved and time.time() < deadline - 15 and no_improve_count < max_no_improve:
        improved = False
        no_improve_count += 1
        
        # Find worst APs
        signs_arr = np.asarray(best_signs, dtype=np.int8)
        worst_d = 1
        worst_r = 0
        worst_val = 0
        worst_indices = []
        
        for d in range(1, n):
            for r in range(d):
                if r >= n:
                    continue
                m = (n - 1 - r) // d + 1
                if m <= 0:
                    continue
                indices = np.arange(r, n, d)
                sub = signs_arr[indices]
                cum = np.cumsum(sub)
                max_sum = 0
                min_sum = 0
                running_min = 0
                running_max = 0
                for i in range(m):
                    val = cum[i]
                    if val < running_min:
                        running_min = val
                    if val > running_max:
                        running_max = val
                    diff_max = val - running_min
                    diff_min = val - running_max
                    if diff_max > max_sum:
                        max_sum = diff_max
                    if diff_min < min_sum:
                        min_sum = diff_min
                current_max = max(abs(max_sum), abs(min_sum))
                if current_max > worst_val:
                    worst_val = current_max
                    worst_d = d
                    worst_r = r
                    worst_indices = list(range(r, n, d))
        
        # Try flipping each position in the worst AP
        for idx in worst_indices:
            if time.time() > deadline - 10:
                break
            old_val = best_signs[idx]
            best_signs[idx] = -old_val
            new_disc = _max_ap_discrepancy(best_signs)
            if new_disc < best_disc:
                best_disc = new_disc
                improved = True
                no_improve_count = 0
            else:
                best_signs[idx] = old_val
        
        # Try random flips
        rng_local = random.Random(int(time.time() * 1000) % 10000)
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            idx = rng_local.randint(0, n - 1)
            old_val = best_signs[idx]
            best_signs[idx] = -old_val
            new_disc = _max_ap_discrepancy(best_signs)
            if new_disc < best_disc:
                best_disc = new_disc
                improved = True
                no_improve_count = 0
            else:
                best_signs[idx] = old_val
        
        # Try double flips
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            i1 = rng_local.randint(0, n - 1)
            i2 = rng_local.randint(0, n - 1)
            if i1 == i2:
                continue
            old1 = best_signs[i1]
            old2 = best_signs[i2]
            best_signs[i1] = -old1
            best_signs[i2] = -old2
            new_disc = _max_ap_discrepancy(best_signs)
            if new_disc < best_disc:
                best_disc = new_disc
                improved = True
                no_improve_count = 0
            else:
                best_signs[i1] = old1
                best_signs[i2] = old2
    
    # Final pass: try each position flip
    if time.time() < deadline - 3:
        for i in range(n):
            if time.time() > deadline - 1:
                break
            old_val = best_signs[i]
            best_signs[i] = -old_val
            new_disc = _max_ap_discrepancy(best_signs)
            if new_disc < best_disc:
                best_disc = new_disc
            else:
                best_signs[i] = old_val
    
    return best_signs
# EVOLVE-BLOCK-END
