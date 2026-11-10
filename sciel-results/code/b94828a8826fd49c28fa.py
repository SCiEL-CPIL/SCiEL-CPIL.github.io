"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 80.0

    n = 12
    indices = list(range(1, n + 1))

    # Precompute popcount for all 4096 masks
    popcount = [0] * 4096
    for mask in range(1, 4096):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)

    def hall_check(m, L):
        freq = [0] * 4096
        for v in range(m + 1, m + L + 1):
            mask = 0
            for i in indices:
                if v % i == 0:
                    mask |= (1 << (i - 1))
            freq[mask] += 1

        subset_sum = freq[:]
        for bit in range(12):
            for mask in range(4096):
                if mask & (1 << bit):
                    subset_sum[mask] += subset_sum[mask ^ (1 << bit)]

        for S in range(1, 4096):
            pc = popcount[S]
            if pc > L:
                continue
            complement = (~S) & 4095
            n_covered = L - subset_sum[complement]
            if n_covered < pc:
                return False
        return True

    def f_of_m(m):
        lo, hi = n, 144
        if not hall_check(m, hi):
            return -1
        while lo < hi:
            mid = (lo + hi) // 2
            if hall_check(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    best_m = 0
    best_f = -1
    period = 27720

    for m in range(0, period):
        if time.time() - start_time > deadline:
            break
        f_val = f_of_m(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m

    return int(best_m)
# EVOLVE-BLOCK-END
