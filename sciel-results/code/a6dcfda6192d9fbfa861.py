import time
import numpy as np
import random

def _ap_discrepancy(signs):
    """Compute max |sum| over all APs."""
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1
    for d in range(1, n + 1):
        for r in range(d):
            if r >= n:
                break
            sub = arr[r::d]
            if len(sub) == 0:
                continue
            ps = np.cumsum(sub)
            cur = np.max(np.abs(ps))
            if cur > max_disc:
                max_disc = int(cur)
    return max_disc

def _all_ap_sums(signs):
    """
    Return a dict mapping (start, step, length) -> sum of that AP.
    Actually, we just need to know for each position i, which APs it affects.
    """
    pass

def _compute_all_ap_sums(signs):
    """
    For each AP (defined by start, step), compute all prefix sums.
    Returns a list of (start, step, prefix_sums_array).
    """
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    ap_data = []
    for d in range(1, n + 1):
        for r in range(d):
            if r >= n:
                break
            sub = arr[r::d]
            if len(sub) == 0:
                continue
            ps = np.cumsum(sub)
            ap_data.append((r, d, ps))
    return ap_data

def _flip_delta(signs, idx, ap_data):
    """
    Compute the change in max discrepancy if we flip signs[idx].
    This is expensive, so we use it carefully.
    Actually, let's just recompute.
    """
    pass

def _greedy_construct(n, seed=1234):
    """
    Greedy construction: minimize max abs sum of APs ending at current index.
    """
    signs = [0] * n
    rng = random.Random(seed)
    
    for i in range(n):
        best_sign = rng.choice([1, -1])
        best_max = float('inf')
        
        # Try +1
        max_pos = 0
        # Try -1
        max_neg = 0
        
        # All APs ending at i
        for d in range(1, i + 1):
            a = i
            while a >= 0:
                # AP: a, a+d, ..., i
                # Length L = (i-a)//d + 1
                # Sum of previous L-1 terms
                s_prev = 0
                curr = a
                while curr < i:
                    s_prev += signs[curr]
                    curr += d
                    
                s_plus = s_prev + 1
                s_minus = s_prev - 1
                
                if abs(s_plus) > max_pos:
                    max_pos = abs(s_plus)
                if abs(s_minus) > max_neg:
                    max_neg = abs(s_minus)
                    
                a -= d
                
        if max_pos <= max_neg:
            best_sign = 1
            best_max = max_pos
        else:
            best_sign = -1
            best_max = max_neg
            
        signs[i] = best_sign
        
    return signs

def _local_search(signs, ap_data, rng, time_limit, start_time):
    """
    Local search: flip bits to reduce max discrepancy.
    Uses incremental update if possible, or full recompute.
    """
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    current_disc = _ap_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    # We will use a simple hill-climbing / simulated annealing
    # To speed up, we can precompute which APs contain each index.
    # index -> list of (start, step, position_in_ap)
    
    # Build index map
    idx_map = [[] for _ in range(n)]
    for (r, d, ps) in ap_data:
        # r is start index
        # length = len(ps)
        # indices: r, r+d, r+2d, ...
        k = 0
        idx = r
        while idx < n:
            idx_map[idx].append((r, d, k))
            k += 1
            idx += d
            
    # Note: ap_data has prefix sums, but if we flip a bit,
    # the prefix sums for that AP from that point onwards change.
    
    # Instead of full recompute, let's just do full recompute but optimized.
    # n=256 is small.
    
    while time.time() - start_time < time_limit:
        # Find worst AP
        # We need to find the AP with max |sum|
        # And also find a bit to flip.
        
        # Let's pick a random bit, or a bit from the worst AP.
        # Find worst AP
        max_val = -1
        worst_r = -1
        worst_d = -1
        worst_k = -1
        
        for (r, d, ps) in ap_data:
            # ps is the current prefix sums
            # We need to find the max abs
            # This is O(total elements in all APs) ~ O(n^2 log n)
            # This is too slow to do every iteration if we do full scan.
            
            # Alternative: maintain current max?
            # Or just sample.
            
            pass
            
        # Let's use a different strategy:
        # 1. Compute current_disc.
        # 2. Find the specific AP that achieves it.
        # 3. Flip a random element in that AP.
        
        # To find the specific AP efficiently:
        # We can just scan all APs again, it's ~ 500k ops.
        # 500k ops in Python is ~ 0.05s.
        # So we can do ~1000 iterations in 60s? Maybe less.
        
        # Let's just do full scan to find worst AP.
        max_disc = 1
        worst_r = 0
        worst_d = 1
        worst_k = 0
        
        for (r, d, ps) in ap_data:
            # ps is array of sums
            # Find index of max abs
            # np.argmax(np.abs(ps))
            # But we need the value
            idx_max = np.argmax(np.abs(ps))
            val = abs(ps[idx_max])
            if val > max_disc:
                max_disc = int(val)
                worst_r = r
                worst_d = d
                worst_k = idx_max
                
        # Now we have worst AP: r, d, ending at k (index in AP)
        # The indices involved are r, r+d, ..., r+k*d
        # We can pick one of these to flip.
        
        # Let's pick the last one (r+k*d) or a random one.
        # Flipping the last one is most likely to reduce the sum if the sign matches the sum.
        idx_flip = worst_r + worst_k * worst_d
        
        # Determine sign to flip
        # If sum > 0, we want to flip a +1 to -1.
        # If sum < 0, we want to flip a -1 to +1.
        
        # Get the sum
        # We need the actual sum value, not just abs.
        # We didn't store it. Let's recompute or store it.
        # Actually, ps[worst_k] is the sum.
        current_sum = int(np.array([p for p in [0] if False] or [0])) # placeholder
        # Let's just recompute the sum for this specific AP
        s = 0
        curr = worst_r
        for _ in range(worst_k + 1):
            s += signs[curr]
            curr += worst_d
            
        if s > 0:
            # Flip a +1 to -1
            # Find indices in this AP that are +1
            candidates = []
            curr = worst_r
            for _ in range(worst_k + 1):
                if signs[curr] == 1:
                    candidates.append(curr)
                curr += worst_d
        else:
            # Flip a -1 to +1
            candidates = []
            curr = worst_r
            for _ in range(worst_k + 1):
                if signs[curr] == -1:
                    candidates.append(curr)
                curr += worst_d
                
        if not candidates:
            # Should not happen
            idx_flip = rng.randint(0, n-1)
        else:
            idx_flip = rng.choice(candidates)
            
        # Flip
        signs[idx_flip] = -signs[idx_flip]
        
        # Recompute
        new_disc = _ap_discrepancy(signs)
        
        if new_disc < current_disc:
            current_disc = new_disc
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
        else:
            # Revert
            signs[idx_flip] = -signs[idx_flip]
            
    return best_signs

def construct():
    n = 256
    start_time = time.time()
    deadline = 55.0
    
    rng = random.Random(42)
    
    # Greedy init
    signs = _greedy_construct(n, seed=42)
    
    current_disc = _ap_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    # Local search
    signs_ls = _local_search(signs, [], rng, deadline, start_time)
    
    # Check result
    disc_ls = _ap_discrepancy(signs_ls)
    if disc_ls < best_disc:
        best_disc = disc_ls
        best_signs = signs_ls
    
    # Try a few different seeds for greedy
    while time.time() - start_time < deadline - 5.0:
        seed = rng.randint(0, 10000)
        signs_try = _greedy_construct(n, seed=seed)
        disc_try = _ap_discrepancy(signs_try)
        
        if disc_try < best_disc:
            best_disc = disc_try
            best_signs = signs_try
            # Do some local search on this
            signs_ls = _local_search(signs_try, [], rng, min(deadline, time.time() + 5.0), start_time)
            disc_ls = _ap_discrepancy(signs_ls)
            if disc_ls < best_disc:
                best_disc = disc_ls
                best_signs = signs_ls
                
    return best_signs
