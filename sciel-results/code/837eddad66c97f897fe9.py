"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
sys.setrecursionlimit(10000)

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer

    # Precompute APs
    ap_list = []
    index_aps = [[] for _ in range(n)]
    
    for d in range(1, n):
        max_len = (n - 1) // d + 1
        for L in range(1, max_len + 1):
            max_start = n - (L - 1) * d
            for a in range(max_start):
                indices = [a + k * d for k in range(L)]
                ap_id = len(ap_list)
                ap_list.append(indices)
                for idx in indices:
                    index_aps[idx].append(ap_id)

    ap_lengths = [len(ap) for ap in ap_list]
    
    best_s = None
    best_B = float('inf')
    
    B = 1
    while time.time() - start_time < time_limit:
        s = [-1] * n
        ap_sums = [0] * len(ap_list)
        unassigned_counts = ap_lengths[:]
        
        t_start_B = time.time()
        
        def is_consistent(k, sigma):
            for ap_id in index_aps[k]:
                new_sum = ap_sums[ap_id] + sigma
                remaining = unassigned_counts[ap_id] - 1
                low = new_sum - remaining
                high = new_sum + remaining
                if high < -B or low > B:
                    return False
            return True

        def bt(k):
            nonlocal s, ap_sums, unassigned_counts
            if time.time() - t_start_B > 5.0:
                return False
            if k == n:
                return True
            for sigma in [1, -1]:
                if is_consistent(k, sigma):
                    s[k] = sigma
                    for ap_id in index_aps[k]:
                        ap_sums[ap_id] += sigma
                        unassigned_counts[ap_id] -= 1
                    if bt(k + 1):
                        return True
                    for ap_id in index_aps[k]:
                        ap_sums[ap_id] -= sigma
                        unassigned_counts[ap_id] += 1
                    s[k] = -1
            return False

        if bt(0):
            best_s = s[:]
            best_B = B
            break
        else:
            B += 1
            if B > 10:
                break

    if best_s is None:
        rng = random.Random(42)
        best_s = [rng.choice([-1, 1]) for _ in range(n)]
        
    return best_s

# EVOLVE-BLOCK-END
