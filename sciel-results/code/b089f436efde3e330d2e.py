import time
import sys

def construct():
    deadline = time.time() + 85
    
    P = 27720  # lcm(1..12)
    N = 12
    max_L = 144
    
    # For each m in [0, P-1], find the minimal L such that a perfect matching exists.
    # Use Hall's theorem with efficient subset-sum counting.
    
    # Precompute popcount
    popcount = [0] * 4096
    for i in range(1, 4096):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    def check_hall_fast(m, L):
        """Check Hall's condition for window (m, m+L]. Returns True if matching exists."""
        if L < N:
            return False
        
        # For each k in 1..L, compute bitmask of i's that divide (m+k)
        # Bit i-1 set if i divides (m+k)
        masks_k = []
        for k in range(1, L + 1):
            val = m + k
            mask = 0
            for i in range(1, 13):
                if val % i == 0:
                    mask |= (1 << (i - 1))
            masks_k.append(mask)
        
        # Frequency of each mask
        freq = [0] * 4096
        for mk in masks_k:
            freq[mk] += 1
        
        # SOS DP: subset_sum[T] = sum of freq[mm] for mm subset of T
        subset_sum = freq[:]
        for bit in range(12):
            for T in range(4096):
                if T & (1 << bit):
                    subset_sum[T] += subset_sum[T ^ (1 << bit)]
        
        full_mask = (1 << 12) - 1
        
        # Check Hall: for all non-empty S, |union| >= |S|
        # |union| = L - subset_sum[~S & full_mask]
        for S in range(1, 4096):
            pc = popcount[S]
            if pc > L:
                return False  # Can't have union size >= pc if pc > L
            comp = (~S) & full_mask
            union_size = L - subset_sum[comp]
            if union_size < pc:
                return False
        
        return True
    
    def find_min_L(m):
        """Binary search for minimal L."""
        # Lower bound: at least N=12
        # Upper bound: 144
        lo, hi = N, max_L
        
        # First check if hi works
        if not check_hall_fast(m, hi):
            return hi + 1  # Should not happen
        
        while lo < hi:
            mid = (lo + hi) // 2
            if check_hall_fast(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # We can't afford to check all 27720 m values with full binary search.
    # Let's sample and find a good candidate.
    
    import random
    random.seed(12345)
    
    best_m = 0
    best_L = 0
    
    # Sample m values
    num_samples = 3000
    sampled = random.sample(range(P), min(num_samples, P))
    
    for m in sampled:
        if time.time() > deadline - 15:
            break
        L = find_min_L(m)
        if L > best_L:
            best_L = L
            best_m = m
    
    # Local search around best
    if time.time() < deadline - 10:
        for dm in range(-100, 101):
            if time.time() > deadline - 5:
                break
            m = (best_m + dm) % P
            if m == best_m:
                continue
            L = find_min_L(m)
            if L > best_L:
                best_L = L
                best_m = m
    
    return int(best_m % P)

if __name__ == "__main__":
    m = construct()
    print(f"m = {m}")
