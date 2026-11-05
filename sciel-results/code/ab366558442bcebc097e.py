import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    
    def get_counts_fast(A):
        A_arr = np.array(sorted(A), dtype=np.int64)
        k = len(A_arr)
        if k == 0:
            return np.zeros(max_sum + 1, dtype=int)
        if k <= 50:
            A_i = A_arr[:, None]
            A_j = A_arr[None, :]
            sums = A_i + A_j
            mask = np.triu(np.ones((k, k), dtype=bool), k=0)
            valid_sums = sums[mask]
            valid_sums = valid_sums[valid_sums <= max_sum]
            if len(valid_sums) > 0:
                counts = np.bincount(valid_sums, minlength=max_sum + 1)
            else:
                counts = np.zeros(max_sum + 1, dtype=int)
            return counts
        else:
            counts = np.zeros(max_sum + 1, dtype=int)
            for i in range(k):
                ai = A_arr[i]
                s = ai + ai
                if s <= max_sum:
                    counts[s] += 1
                for j in range(i + 1, k):
                    aj = A_arr[j]
                    s = ai + aj
                    if s <= max_sum:
                        counts[s] += 1
            return counts

    def get_cost(A):
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum + 1]
        if np.any(target_counts == 0):
            return 1000 + np.sum(target_counts == 0)
        return float(np.max(target_counts))

    # ... (incremental cost functions remain similar but return float) ...
    
    # Use Simulated Annealing
    # ...
