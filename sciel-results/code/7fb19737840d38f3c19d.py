import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    min_sum = 128
    max_sum = 256
    
    def evaluate(arr):
        """Returns (peak, covered) using convolution."""
        conv = np.convolve(arr, arr, 'full')
        target_conv = conv[min_sum:max_sum+1].astype(np.float64)
        s_vals = np.arange(min_sum, max_sum + 1)
        even_mask = (s_vals % 2 == 0)
        diag = np.zeros(len(s_vals))
        even_s = s_vals[even_mask]
        halves = even_s // 2
        valid = halves <= n
        diag[even_mask] = np.where(valid, arr[halfes[valid]], 0)
        unordered = (target_conv + diag) / 2.0
        if np.any(unordered < 0.5):  # < 1
            return float('inf'), False
        return int(np.ceil(np.max(unordered))), True
    
    best_peak = float('inf')
    best_arr = None
    rng = random.Random(123)
    
    # Try multiple initial configurations
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        # Generate initial set
        arr = np.zeros(n + 1, dtype=np.int32)
        arr[128] = 1  # Must have 128 for sum 256
        
        # Add random elements
        size = rng.randint(15, 40)
        candidates = list(range(0, 128))
        rng.shuffle(candidates)
        for i in range(min(size - 1, len(candidates))):
            arr[candidates[i]] = 1
        
        # Ensure coverage by adding elements
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            peak, covered = evaluate(arr)
            if covered:
                break
            # Find an uncovered sum and add an element to cover it
            conv = np.convolve(arr, arr, 'full')
            target_conv = conv[min_sum:max_sum+1]
            s_vals = np.arange(min_sum, max_sum + 1)
            even_mask = (s_vals % 2 == 0)
            diag = np.zeros(len(s_vals))
            even_s = s_vals[even_mask]
            halves = even_s // 2
            valid = halves <= n
            diag[even_mask] = np.where(valid, arr[halfes[valid]], 0)
            unordered = (target_conv + diag) / 2.0
            uncovered = s_vals[unordered < 1]
            if len(uncovered) == 0:
                break
            s = rng.choice(uncovered.tolist())
            # Try to add x such that x or s-x is in A
            added = False
            for x in range(0, 129):
                if arr[x]:
                    continue
                y = s - x
                if 0 <= y <= 128 and (arr[y] or x == y):
                    arr[x] = 1
                    added = True
                    break
            if not added:
                # Add random
                zeros = np.where(arr == 0)[0]
                if len(zeros) > 0:
                    arr[rng.choice(zeros)] = 1
        
        peak, covered = evaluate(arr)
        if not covered:
            continue
        
        if peak < best_peak:
            best_peak = peak
            best_arr = arr.copy()
        
        # Local search
        cur = arr.copy()
        cur_peak = peak
        stagnant = 0
        
        while time.time() < deadline - 5:
            if stagnant > 500:
                break
            
            if rng.random() < 0.4 and np.sum(cur) > 12:
                # Remove
                idxs = np.where(cur > 0)[0]
                idxs = idxs[idxs != 128]
                if len(idxs) == 0:
                    break
                idx = rng.choice(idxs.tolist())
                cur[idx] = 0
                new_peak, cov = evaluate(cur)
                if cov and new_peak <= cur_peak:
                    cur_peak = new_peak
                    if new_peak < best_peak:
                        best_peak = new_peak
                        best_arr = cur.copy()
                        stagnant = 0
                    else:
                        stagnant += 1
                else:
                    cur[idx] = 1
                    stagnant += 1
            else:
                # Add
                zeros = np.where(cur == 0)[0]
                if len(zeros) == 0:
                    break
                idx = rng.choice(zeros.tolist())
                cur[idx] = 1
                new_peak, cov = evaluate(cur)
                if cov and new_peak <= cur_peak:
                    cur_peak = new_peak
                    if new_peak < best_peak:
                        best_peak = new_peak
                        best_arr = cur.copy()
                        stagnant = 0
                    else:
                        stagnant += 1
                else:
                    cur[idx] = 0
                    stagnant += 1
    
    if best_arr is not None:
        return list(np.where(best_arr > 0)[0])
    return list(range(64, 129))
