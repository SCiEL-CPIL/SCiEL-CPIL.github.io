import time
from math import log
import numpy as np

def _fast_score(U):
    """
    Fast computation of the score using FFT for sumset and difference set sizes.
    Score = 1 + log(|U-U|/|U+U|)/log(2 max U + 1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # We need to compute sizes of U+U and U-U.
    # U+U is contained in [0, 2*max_u]. Size of range is 2*max_u + 1.
    # U-U is contained in [-max_u, max_u]. Size of range is 2*max_u + 1.
    
    # Using autocorrelation of the indicator function.
    # Let f be the indicator vector of U.
    # The sumset size is the number of non-zero entries in the linear convolution of f with f (reversed? no, f with f).
    # Actually, |A+B| is the support size of the convolution of 1_A and 1_B.
    # For A=B=U, it's the support size of 1_U * 1_U.
    # The difference set size |A-A| is the support size of the autocorrelation (cross-correlation with reverse).
    
    # To avoid circular convolution artifacts, we need padding.
    # Max value in sum is 2*max_u. So we need length at least 2*max_u + 1.
    # Let's pick N as a power of 2 >= 2*(2*max_u + 1) to be safe for both.
    
    m = 2 * max_u + 1
    N = 1
    while N < 2 * m:
        N <<= 1
        
    f = np.zeros(N, dtype=np.float64)
    f[U] = 1.0
    
    F = np.fft.rfft(f)
    
    # Sumset: Convolution of f with f. 
    # In frequency domain: F * F.
    # Note: Standard convolution c_k = sum f_i g_{k-i}. 
    # For sets, this counts representations. We just need support.
    sum_conv = np.fft.irfft(F * F)
    # The sumset elements are in [0, 2*max_u]. 
    # Due to circular convolution, if N is large enough, the linear part is at the start.
    sumset_size = int(np.count_nonzero(sum_conv[:m] > 0.5))
    
    # Difference set: Autocorrelation.
    # d_k = sum f_i f_{i+k} ? No, that's cross correlation.
    # |A-A| is the support of the set {a-b}.
    # This is equivalent to the support of the cross-correlation of f with f_reversed.
    # Or simply: autocorrelation R_k = sum f_i f_{i+k} gives counts of differences?
    # Let's check: if f = delta_0 + delta_1. 
    # R_0 = 1+1=2. R_1 = f_0 f_1 + f_1 f_2 = 1. R_{-1} = f_1 f_0 = 1.
    # The FFT of f gives F. The FFT of autocorrelation is |F|^2.
    # irfft(|F|^2) gives the circular autocorrelation.
    # The lags are 0, 1, ..., N/2. 
    # The difference set is symmetric. 
    # The support of the autocorrelation corresponds to the differences.
    diff_conv = np.fft.irfft(F * np.conj(F))
    
    # The differences range from -max_u to max_u.
    # In the circular autocorrelation array of length N:
    # Index 0 is lag 0.
    # Indices 1 to N/2 are positive lags 1 to N/2.
    # Indices N/2+1 to N-1 are negative lags.
    # We care about lags -max_u to max_u.
    # Since max_u <= N/4 (because N >= 4*max_u + 4), there is no wrap-around ambiguity for the central block.
    # The positive differences are in diff_conv[1:max_u+1].
    # The negative differences are in diff_conv[N-max_u:N].
    # 0 is in diff_conv[0].
    
    # We just need to count non-zeros in these ranges.
    # Note: diff_conv[0] is always non-zero if U is not empty.
    
    # Positive part
    pos_diffs = diff_conv[1:max_u+1]
    # Negative part
    neg_diffs = diff_conv[N-max_u:N]
    
    diffset_size = int(np.count_nonzero(pos_diffs > 0.5)) + int(np.count_nonzero(neg_diffs > 0.5)) + 1 # +1 for 0
    
    # Sanity check: |U-U| must be at least |U|
    # And |U+U| must be at least |U|
    
    if sumset_size == 0:
        return 1.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 1.0:
        return 1.0
        
    return 1.0 + log(ratio) / log(2 * max_u + 1)

def compute_score(U):
    """
    Wrapper that tries fast score, falls back to slow exact score if needed.
    """
    try:
        s = _fast_score(U)
        # Optional: verify with slow score for small sets to ensure FFT precision isn't an issue
        # For this task, FFT is generally robust enough for integers < 10^7
        return s
    except Exception:
        # Fallback to exact O(n^2) computation
        U = sorted(set(U))
        if 0 not in U:
            return 0.0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 1:
            return 1.0
        return 1.0 + log(ratio) / log(2 * max_u + 1)

def construct():
    start = time.time()
    deadline = start + 115  # 15s safety margin under 130s limit (assuming 120s hard kill or similar, 115 is safe)
    
    def time_ok():
        return time.time() < deadline

    best_U = [0, 1, 3]
    best_score = compute_score(best_U)
    
    def consider(U):
        nonlocal best_U, best_score
        if not time_ok():
            return
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
            
    # 1. Structured Families
    # Powers of b: {0, b, b^2, ...}
    # These have large gaps, which helps |U-U|/|U+U|.
    # Score approx 1 + ln(2)/ln(2*b^k)
    # To maximize, we want small b and large k? 
    # If b=2, k=10 -> max=512. Score ~ 1 + 0.69/ln(1025) ~ 1 + 0.69/6.9 ~ 1.1
    # If b=10, k=5 -> max=100000. Score ~ 1 + 0.69/ln(200001) ~ 1 + 0.69/12.2 ~ 1.05
    # So small bases with moderate size seem better for the constant factor, but we can go larger.
    
    # Let's scan bases 2 to 100
    for b in range(2, 50):
        if not time_ok():
            break
        # Generate powers
        U = [1]
        val = 1
        for i in range(1, 20):
            val *= b
            if val > 10**7: # Keep max_u reasonable to keep FFT fast
                break
            U.append(val)
        if 0 not in U:
            U.insert(0, 0)
        consider(U)
        
        # Also try {0, b-1, (b-1)^2, ...}
        if b > 2:
            U2 = [1]
            val = 1
            base = b - 1
            for i in range(1, 20):
                val *= base
                if val > 10**7:
                    break
                U2.append(val)
            if 0 not in U2:
                U2.insert(0, 0)
            consider(U2)
            
    # 2. Greedy Local Search
    # Start from several good seeds
    seeds = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 9],
        [0, 1, 4, 10],
        [0, 1, 2, 4, 8], # Powers of 2
        [0, 1, 3, 9, 27] # Powers of 3
    ]
    
    # Add best from structured scan
    seeds.append(best_U[:])
    
    def greedy_grow(U):
        U = sorted(set(U))
        if 0 not in U:
            U.insert(0, 0)
        cur_score = compute_score(U)
        improved = True
        # Limit size to prevent explosion of candidates
        max_elements = 30 
        
        while improved and time_ok() and len(U) < max_elements:
            improved = False
            mx = max(U)
            
            # Candidate generation:
            # 1. Small integers near existing gaps?
            # 2. Large jumps to maintain lacunarity.
            
            # Try adding small numbers first to fill in if it helps? 
            # Usually, for this metric, we want SPARSE sets.
            # So we should prioritize large numbers.
            
            candidates = set()
            
            # Local small candidates (to fix local defects)
            # Range: [1, mx//2] is too big. 
            # Let's try [1, 20] and [mx-10, mx+10]
            candidates.update(range(1, min(20, mx+1)))
            if mx > 10:
                candidates.update(range(max(1, mx-10), mx+11))
                
            # Large structured candidates
            # Multiples of max
            for mult in [2, 3, 4, 5, 7, 10, 16, 32, 64]:
                c = mx * mult
                if c <= 10**7:
                    candidates.add(c)
                    
            # Powers related to max
            candidates.add(mx * mx)
            if mx > 1:
                candidates.add(int(mx ** 0.5) + 1)
                
            # Random large candidates
            for _ in range(5):
                c = int(np.random.uniform(mx, mx * 4))
                if c <= 10**7:
                    candidates.add(c)
            
            # Filter and sort
            valid_candidates = [c for c in candidates if c > 0 and c not in U]
            # Sort by "value" - we want to try large ones first? 
            # Or small ones? 
            # Let's sort by potential impact. 
            # Adding a large number increases denominator log(2*max+1) but also increases |U-U|.
            # Usually, adding a large number is good if it's "far" from others.
            # Let's try all, but break early if we find a good improvement.
            
            # Sort candidates: Mix of small and large
            # Let's just sort ascending.
            valid_candidates.sort()
            
            best_cand = None
            best_cand_score = cur_score
            
            for c in valid_candidates:
                if not time_ok():
                    break
                new_U = U + [c]
                # Quick check: if c is too close to existing, it might not help much
                # But compute_score is fast enough.
                ns = compute_score(new_U)
                if ns > best_cand_score:
                    best_cand_score = ns
                    best_cand = c
            
            if best_cand is not None:
                U.append(best_cand)
                U = sorted(set(U))
                cur_score = best_cand_score
                improved = True
                
        return U, cur_score

    for seed in seeds:
        if not time_ok():
            break
        U, s = greedy_grow(seed)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    # Final polish: try growing from the absolute best
    if time_ok():
        U, s = greedy_grow(best_U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    return best_U

if __name__ == "__main__":
    # For local testing
    U = construct()
    print("Result:", U)
    print("Score:", compute_score(U))
