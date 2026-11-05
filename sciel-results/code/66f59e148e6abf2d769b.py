import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    min_sum = 128
    max_sum = 256
    S = max_sum - min_sum + 1 # 129 sums
    
    def init_counts(mask):
        m = len(mask)
        N = 1 << (2 * m - 1).bit_length()
        f = np.fft.rfft(mask.astype(np.float64), n=N)
        conv = np.fft.irfft(f * f, n=N)[:2 * m - 1]
        conv = np.rint(conv).astype(np.int64)
        
        sums = np.arange(min_sum, max_sum + 1)
        ordered_counts = conv[sums]
        
        even_mask = (sums % 2 == 0)
        half_sums = sums // 2
        diag = np.where(even_mask, 
                        np.where((half_sums >= 0) & (half_sums <= n), 
                                mask[half_sums], 0), 
                        0)
        
        counts = (ordered_counts + diag) // 2
        return counts

    def get_cost(counts):
        max_c = int(np.max(counts))
        uncovered = int(np.sum(counts == 0))
        return max_c + 100 * uncovered

    best_score = float('inf')
    best_mask = None
    best_counts = None

    def update_best(mask, counts):
        nonlocal best_score, best_mask, best_counts
        if np.min(counts) > 0:
            val_score = int(np.max(counts))
            if val_score < best_score:
                best_score = val_score
                best_mask = mask.copy()
                best_counts = counts.copy()

    # Initialization
    random.seed(42)
    
    for _ in range(20):
        size = random.randint(30, 70)
        A = random.sample(range(n + 1), size)
        mask = np.zeros(n + 1, dtype=np.int64)
        for x in A:
            mask[x] = 1
        counts = init_counts(mask)
        update_best(mask, counts)

    if best_mask is None:
        mask = np.zeros(n + 1, dtype=np.int64)
        mask[64:129] = 1
        counts = init_counts(mask)
        update_best(mask, counts)

    # Simulated Annealing with Incremental Updates
    current_mask = best_mask.copy() if best_mask is not None else np.zeros(n+1, dtype=np.int64)
    if np.any(current_mask) == 0:
        current_mask[64:129] = 1
    current_counts = init_counts(current_mask)
    current_cost = get_cost(current_counts)
    
    T = 10.0
    cooling = 0.999
    
    # Precompute list of elements in set for fast iteration? 
    # We just flip one element v.
    
    while time.time() < deadline - 5 and T > 0.1:
        v = random.randint(0, n)
        
        # Calculate delta cost for flipping v
        # We need to update counts for sums involving v.
        # Sums: 2v, v+u for u in A, u != v.
        
        # Let's just do the incremental update on a copy of counts?
        # No, we can calculate the change in cost directly.
        
        # Get current counts for affected sums
        # And determine new counts
        
        # This is tricky to do without modifying the array and reverting.
        # Given the speed of FFT, maybe it's faster to just use FFT for SA?
        # FFT takes ~0.1-0.5ms. 
        # 10000 steps * 0.5ms = 5s. 
        # So FFT is fine.
        
        trial_mask = current_mask.copy()
        trial_mask[v] = 1 - trial_mask[v]
        
        if not np.any(trial_mask):
            continue
            
        trial_counts = init_counts(trial_mask)
        trial_cost = get_cost(trial_counts)
        
        delta = trial_cost - current_cost
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_mask = trial_mask
            current_counts = trial_counts
            current_cost = trial_cost
            update_best(current_mask, current_counts)
            
        T *= cooling

    # Final local search
    if best_mask is not None:
        m = best_mask.copy()
        c = best_counts.copy()
        max_c = int(np.max(c))
        
        in_set = [i for i in range(n+1) if m[i]]
        random.shuffle(in_set)
        
        for v in in_set:
            if time.time() > deadline - 2:
                break
            affected = []
            for u in in_set:
                if u == v:
                    s = 2*v
                elif u < v:
                    s = u + v
                else:
                    continue
                if min_sum <= s <= max_sum:
                    affected.append(s)
            
            safe = True
            for s in affected:
                if c[s - min_sum] == 1:
                    safe = False
                    break
            
            if safe:
                peaks = [s for s in range(min_sum, max_sum+1) if c[s - min_sum] == max_c]
                all_peaks_affected = True
                for s in peaks:
                    if s == 2 * v:
                        pass
                    else:
                        u = s - v
                        if u < 0 or u > n or not m[u]:
                            all_peaks_affected = False
                            break
                
                if all_peaks_affected:
                    for s in affected:
                        c[s - min_sum] -= 1
                    m[v] = 0
                    in_set.remove(v)
                    max_c -= 1
                    if max_c < best_score:
                        best_score = max_c
                        best_mask = m.copy()
                        best_counts = c.copy()

    if best_mask is None:
        return list(range(64, 129))

    return [i for i in range(n + 1) if best_mask[i]]
