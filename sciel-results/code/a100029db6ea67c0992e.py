"""
Structural rewrite: simulated annealing with bitmask-vectorized scoring
to maximize distinct 3-AP differences for 32 integers in 1..512.
"""
import time
import random
import numpy as np

def _score_bitmask(mask):
    """
    Fast score using a 512-bit integer bitmask.
    mask: int with bit i set iff (i+1) is in the set (i ranges 0..511).
    Returns number of distinct d>0 such that a, a+d, a+2d all in set.
    """
    valid_ds = set()
    # For each element position i (0-indexed, value = i+1),
    # for each other element j > i, d = j - i (in index space = value space)
    # Check if (i - d) >= 0 and bit (i-d) is set, OR (j + d) < 512 and bit (j+d) is set
    # But we need to iterate over set bits.
    
    # Extract positions of set bits
    positions = []
    m = mask
    idx = 0
    while m:
        if m & 1:
            positions.append(idx)
        m >>= 1
        idx += 1
    
    n = len(positions)
    pos_set = set(positions)
    
    for i in range(n):
        x = positions[i]
        for j in range(i + 1, n):
            y = positions[j]
            d = y - x
            # x-d in set means (x-d, x, y) is a 3-AP with diff d
            # y+d in set means (x, y, y+d) is a 3-AP with diff d
            if (x - d) in pos_set or (y + d) in pos_set:
                valid_ds.add(d)
    
    return len(valid_ds)


def _score_fast(S):
    """Score a set of 32 integers (values 1..512) using bitmask."""
    mask = 0
    for v in S:
        mask |= (1 << (v - 1))
    return _score_bitmask(mask)


def _score_numpy(S):
    """
    Vectorized score using numpy.
    S: list of 32 integers in 1..512.
    Returns count of distinct d > 0 with a 3-AP.
    """
    arr = np.array(S, dtype=np.int32)
    n = len(arr)
    # All pairs (i,j) with i < j
    # d = arr[j] - arr[i]
    # Check if arr[i] - d is in S  => 2*arr[i] - arr[j] in S
    # Check if arr[j] + d is in S  => 2*arr[j] - arr[i] in S
    
    valid_ds = set()
    # For each pair, compute d and check
    for i in range(n):
        for j in range(i + 1, n):
            d = arr[j] - arr[i]
            # a, a+d, a+2d: need a = arr[i]-d = 2*arr[i]-arr[j], or a = arr[i]
            # If a = arr[i]: need arr[i]+2d = arr[j]+d in S
            # If a = arr[i]-d: need arr[i] and arr[i]+d=arr[j] already, check arr[i]-d in S
            candidate1 = arr[i] - d  # = 2*arr[i] - arr[j]
            candidate2 = arr[j] + d  # = 2*arr[j] - arr[i]
            if candidate1 in S or candidate2 in S:
                valid_ds.add(d)
    return len(valid_ds)


def _score_set(S):
    """Score using a Python set for membership (faster than list)."""
    S_set = set(S)
    valid_ds = set()
    n = len(S)
    for i in range(n):
        x = S[i]
        for j in range(i + 1, n):
            y = S[j]
            d = y - x
            if (x - d) in S_set or (y + d) in S_set:
                valid_ds.add(d)
    return len(valid_ds)


def _make_initial_set(rng, style):
    """Create diverse initial sets."""
    if style == 0:
        # Random
        return list(rng.sample(range(1, 513), 32))
    elif style == 1:
        # Arithmetic progression based
        start = rng.randint(1, 30)
        step = rng.randint(10, 15)
        s = set()
        x = start
        while len(s) < 32 and x <= 512:
            s.add(x)
            x += step
        # Fill remaining randomly
        remaining = [v for v in range(1, 513) if v not in s]
        s.update(rng.sample(remaining, 32 - len(s)))
        return list(s)
    elif style == 2:
        # Two interleaved APs
        start1 = rng.randint(1, 20)
        step1 = rng.randint(12, 18)
        start2 = rng.randint(1, 20)
        step2 = rng.randint(12, 18)
        s = set()
        x = start1
        while len(s) < 16 and x <= 512:
            s.add(x)
            x += step1
        x = start2
        while len(s) < 32 and x <= 512:
            s.add(x)
            x += step2
        remaining = [v for v in range(1, 513) if v not in s]
        if len(s) < 32:
            s.update(rng.sample(remaining, 32 - len(s)))
        return list(s)
    elif style == 3:
        # Geometric-like (powers then fill)
        s = set()
        base = rng.randint(2, 4)
        x = 1
        while x <= 512 and len(s) < 32:
            s.add(x)
            x *= base
        remaining = [v for v in range(1, 513) if v not in s]
        s.update(rng.sample(remaining, 32 - len(s)))
        return list(s)
    else:
        # Clustered: groups in different regions
        regions = [(1, 128), (129, 256), (257, 384), (385, 512)]
        s = set()
        for lo, hi in regions:
            count = 8
            s.update(rng.sample(range(lo, hi + 1), count))
        return list(s)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    rng = random.Random(12345)
    
    best_set = None
    best_score = 0
    
    # Multiple restarts with different initializations
    num_restarts = 8
    restarts_done = 0
    
    for restart in range(num_restarts):
        if time.time() >= deadline:
            break
        
        style = restart % 5
        current_set = _make_initial_set(rng, style)
        current_score = _score_set(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)
        
        # Simulated annealing on this set
        T = 1.0
        T_min = 0.001
        cooling = 0.9995
        accepted = 0
        total_steps = 0
        
        while time.time() < deadline and T > T_min:
            # Generate a random move: pick one element to replace
            idx = rng.randrange(32)
            old_val = current_set[idx]
            
            # Choose new value: biased toward nearby values or random
            if rng.random() < 0.3:
                # Local move: perturb by small amount
                delta = rng.randint(-20, 20)
                new_val = old_val + delta
                if new_val < 1 or new_val > 512:
                    new_val = rng.randint(1, 512)
            else:
                new_val = rng.randint(1, 512)
            
            if new_val in current_set:
                continue
            
            current_set[idx] = new_val
            new_score = _score_set(current_set)
            
            delta_score = new_score - current_score
            total_steps += 1
            
            if delta_score > 0 or (T > T_min and rng.random() < np.exp(delta_score / T)):
                current_score = new_score
                accepted += 1
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
            else:
                # Reject
                current_set[idx] = old_val
            
            T *= cooling
        
        restarts_done += 1
    
    # Final: if we have time, do a focused local search on best_set
    if best_set is not None and time.time() < deadline - 5.0:
        current_set = list(best_set)
        current_score = _score_set(current_set)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for idx in range(32):
                if time.time() >= deadline - 2.0:
                    break
                old_val = current_set[idx]
                # Try a few candidate replacements
                for _ in range(10):
                    new_val = rng.randint(1, 512)
                    if new_val in current_set:
                        continue
                    current_set[idx] = new_val
                    new_score = _score_set(current_set)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current_set)
                        break
                    else:
                        current_set[idx] = old_val
    
    return list(best_set) if best_set else list(range(1, 33))


# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Result: {sorted(result)}")
    print(f"Score: {_score_set(result)}")
    print(f"Time: {t1 - t0:.1f}s")
    print(f"Count: {len(result)}, Distinct: {len(set(result))}")
    print(f"Range: {min(result)}..{max(result)}")
