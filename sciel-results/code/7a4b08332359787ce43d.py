import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 128
    target_lo = 128
    target_hi = 256
    
    def compute_multiplicities(A_set):
        vec = np.zeros(n + 1, dtype=np.float64)
        for x in A_set:
            if 0 <= x <= n:
                vec[x] = 1.0
        size = 256
        f = np.fft.fft(vec, size=size)
        conv = np.fft.ifft(f * f).real
        mult = np.zeros(target_hi + 1, dtype=np.int64)
        for s in range(target_lo, target_hi + 1):
            if s < 0 or s > 2 * n:
                mult[s] = 0
                continue
            ordered = int(round(conv[s]))
            if s % 2 == 0 and (s // 2) in A_set:
                unordered = (ordered + 1) // 2
            else:
                unordered = ordered // 2
            mult[s] = unordered
        return mult[target_lo:target_hi + 1]
    
    def get_peak(mult):
        return int(np.max(mult))
    
    # Start with interval [64,128]
    A = set(range(64, 129))
    mult = compute_multiplicities(A)
    best_peak = get_peak(mult)
    best_A = set(A)
    
    random.seed(42)
    
    # Try various structured initial sets
    candidates = []
    
    # Interval-based with gaps
    for start in range(40, 70):
        for end in range(100, 129):
            if time.time() > deadline:
                break
            A_try = set(range(start, end + 1))
            if min(A_try) > 64 or max(A_try) < 128:
                continue
            mult_try = compute_multiplicities(A_try)
            if np.all(mult_try > 0):
                peak_try = get_peak(mult_try)
                if peak_try < best_peak:
                    best_peak = peak_try
                    best_A = set(A_try)
    
    # Random sets with structure
    for size in range(45, 75, 3):
        for attempt in range(3):
            if time.time() > deadline:
                break
            # Ensure coverage: include some low and high elements
            low_part = random.sample(range(0, 65), min(10, size // 3))
            high_part = random.sample(range(100, 129), min(10, size // 3))
            mid_part = random.sample(range(50, 100), max(0, size - len(low_part) - len(high_part)))
            A_try = set(low_part + high_part + mid_part)
            if min(A_try) > 64 or max(A_try) < 128:
                continue
            mult_try = compute_multiplicities(A_try)
            if np.all(mult_try > 0):
                peak_try = get_peak(mult_try)
                if peak_try < best_peak:
                    best_peak = peak_try
                    best_A = set(A_try)
    
    # Local search on best_A
    A = set(best_A)
    mult = compute_multiplicities(A)
    peak = get_peak(mult)
    
    # Efficient local search: try swaps that target high-multiplicity sums
    no_improve = 0
    while time.time() < deadline and no_improve < 100:
        no_improve += 1
        improved = False
        
        # Find the target with highest multiplicity
        max_idx = int(np.argmax(mult))
        max_val = mult[max_idx]
        target_sum = target_lo + max_idx
        
        # Try to reduce multiplicity at target_sum by swapping
        # Elements that contribute to target_sum: pairs (a, target_sum - a) where both in A
        contributors = []
        for a in A:
            b = target_sum - a
            if b in A and a <= b:
                contributors.append((a, b))
        
        # Try removing one contributor element and adding a non-contributor
        for (a, b) in contributors:
            if time.time() > deadline:
                break
            for y in range(0, n + 1):
                if y in A:
                    continue
                # Check if y would help with other high-multiplicity targets
                A_new = set(A)
                if a == b:
                    A_new.remove(a)
                else:
                    A_new.remove(a)
                A_new.add(y)
                
                mult_new = compute_multiplicities(A_new)
                if not np.all(mult_new > 0):
                    continue
                peak_new = get_peak(mult_new)
                if peak_new < peak:
                    A = A_new
                    mult = mult_new
                    peak = peak_new
                    best_A = set(A)
                    best_peak = peak
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        
        if not improved:
            # Try random swaps
            for _ in range(50):
                if time.time() > deadline:
                    break
                x = random.choice(list(A))
                y = random.randint(0, n)
                if y == x:
                    continue
                A_new = set(A)
                A_new.remove(x)
                A_new.add(y)
                mult_new = compute_multiplicities(A_new)
                if not np.all(mult_new > 0):
                    continue
                peak_new = get_peak(mult_new)
                if peak_new < peak:
                    A = A_new
                    mult = mult_new
                    peak = peak_new
                    best_A = set(A)
                    best_peak = peak
                    improved = True
                    no_improve = 0
                    break
                elif peak_new == peak:
                    # Accept with some probability (simulated annealing-like)
                    if random.random() < 0.3:
                        A = A_new
                        mult = mult_new
    
    return sorted(best_A)
