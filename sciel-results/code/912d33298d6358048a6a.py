import time
import numpy as np
from itertools import product
import sys

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    if len(arr) > 0 and arr[0] != 0:
        # Ensure 0 is in the set for correctness of sum/diff sets relative to 0
        # The task says U contains 0. If not, we should probably add it or handle it.
        # The seed program assumes 0 is there or adds it.
        # Let's strictly follow the task: U contains 0.
        # If the user passes a set without 0, the score might be undefined or different.
        # But construct() must return a set containing 0.
        pass
        
    arr = np.unique(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    # Size of sumset U+U: values in [0, 2*mx]
    # Size of diffset U-U: values in [-mx, mx], shift by mx -> [0, 2*mx]
    N = 2 * mx + 1
    
    # Efficient counting via FFT
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6) # Sums range 0..2*mx which is 0..N-1
    
    # Diffset
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    diffset_size = np.sum(S_diff > 1e-6)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + np.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Strategy:
    # We construct U as a mixed-radix set: U = { sum_{i=0}^{K-1} d_i M_i }
    # where M_0 = 1, M_{i+1} = 2 * max(U_i) + 1.
    # This ensures the sumset and diffset sizes factorize:
    # |U+U| = prod |D_i+D_i|
    # |U-U| = prod |D_i-D_i|
    # Max(U) = sum_{i=0}^{K-1} max(D_i) M_i
    #
    # The score is:
    # S = 1 + (sum_{i=0}^{K-1} log(|D_i-D_i|/|D_i+D_i|)) / log(2 Max(U) + 1)
    #
    # For large K, the score converges to the "efficiency" of the atom D:
    # Eff(D) = log(|D-D|/|D+D|) / log(2 max(D) + 1)  <-- Wait, the denominator is log(2 Max(U)+1).
    # Let's analyze the asymptotic behavior.
    # Let r_D = |D-D|/|D+D| and s_D = max(D).
    # Max(U_K) approx (s_D+1) * M_K / 2 ? No.
    # M_{i+1} = 2 M_i s_D + 1.
    # M_K approx C * (2 s_D)^K.
    # Max(U_K) = sum d_i M_i approx M_K * (1 + 1/(2s_D) + ...) = M_K * (2s_D)/(2s_D-1).
    # So Max(U) is proportional to (2s_D)^K.
    # log(2 Max(U) + 1) approx K * log(2 s_D).
    # Numerator = K * log(r_D).
    # So asymptotic score = 1 + log(r_D) / log(2 s_D).
    #
    # We want to maximize log(r_D) / log(2 s_D) over all integer sets D containing 0.
    # Let's search for the best "Atom" D.
    #
    # Instead of random search, let's do a systematic search for D up to a certain size/span.
    # We can also consider "Super-Atoms" which are products of two small sets,
    # but the factorization property means we can just treat the product as a single larger D
    # if we re-calculate its r and s.
    # Actually, if D = A x B (mixed radix), then r_D = r_A r_B and s_D = s_A s_B + s_A + s_B.
    # The efficiency is additive in the numerator but the denominator is log(2s_D).
    # log(2 (s_A s_B + s_A + s_B)) is not exactly log(2s_A) + log(2s_B).
    # However, for large s, it is close.
    
    # Let's find the single best atom D.
    # Then, we might improve by mixing different atoms if their efficiencies are close,
    # or by using a "tail" atom that has a slightly worse efficiency but helps the finite-K boundary terms?
    # Actually, for a fixed K, we want to maximize the total log-ratio for a given Max(U).
    # This is a knapsack-like problem if we view each level as an item.
    # But since we can repeat the same atom, the optimal strategy for large K is to repeat the most efficient atom.
    # For small K, we might use a different atom.
    
    # Step 1: Find the best atom D.
    # We'll search D with size up to 10 and max element up to 50.
    # To make it faster, we can restrict D to be "sum-free" or have other properties?
    # No, we want high diffset/sumset ratio.
    # Sets with high ratio are typically "thin" in the middle or have specific gaps.
    
    best_atom = None
    best_eff = 0.0
    
    # Precompute for all subsets of [1..30] of size up to 5?
    # 30 choose 5 is ~140k. 30 choose 4 is ~27k.
    # Let's go up to size 6 and span 30.
    # We can prune: if r <= 1, it's useless (score < 1). We need r > 1.
    
    from itertools import combinations
    
    max_span = 40
    max_size = 6
    
    # We can also try to optimize the "shape" of D.
    # A common good shape is {0, a, b, c...} where elements are spaced out.
    
    # Let's use a smarter search:
    # 1. Generate candidate atoms.
    # 2. Evaluate their efficiency.
    # 3. Pick the best one.
    # 4. Try to construct U using K copies of the best atom.
    # 5. Try to improve by replacing the last 1-2 atoms with other good atoms.
    
    candidates = []
    
    # Generate atoms
    # To speed up, we can skip atoms where max(D) is too small or too large relative to size.
    # Also, we can sort candidates by size.
    
    for size in range(2, max_size + 1):
        # Combinations of 1..max_span
        # This might be slow if max_span is 40 and size is 6.
        # 40C6 = 3,838,380. That's too many for Python loops.
        # We need to reduce the search space.
        
        # Heuristic: Good atoms often have elements that are "powers" or "multiples".
        # Or we can just search smaller spans for larger sizes.
        
        if size == 2:
            span_limit = max_span
        elif size == 3:
            span_limit = max_span
        elif size == 4:
            span_limit = 25
        elif size == 5:
            span_limit = 15
        else:
            span_limit = 10
            
        # Actually, let's just do a random search for atoms if the exhaustive is too slow.
        # But we have 110 seconds.
        # Let's try a hybrid: Exhaustive for small sizes, random for larger.
        
        if size <= 4:
            for comb in combinations(range(1, span_limit + 1), size - 1):
                D = [0] + list(comb)
                # Calculate r and s
                sumset = set()
                diffset = set()
                for a in D:
                    for b in D:
                        sumset.add(a + b)
                        diffset.add(a - b)
                r = len(diffset) / len(sumset)
                if r > 1.0:
                    s = max(D)
                    eff = np.log(r) / np.log(2 * s + 1)
                    if eff > best_eff:
                        best_eff = eff
                        best_atom = D
        else:
            # Random search for larger sizes
            for _ in range(5000):
                D = [0] + list(np.random.choice(range(1, span_limit + 1), size=size-1, replace=False))
                D.sort()
                sumset = set()
                diffset = set()
                for a in D:
                    for b in D:
                        sumset.add(a + b)
                        diffset.add(a - b)
                r = len(diffset) / len(sumset)
                if r > 1.0:
                    s = max(D)
                    eff = np.log(r) / np.log(2 * s + 1)
                    if eff > best_eff:
                        best_eff = eff
                        best_atom = D

    if best_atom is None:
        return [0, 1]
        
    # Now we have the best atom D*.
    # Let's build U using K copies of D*.
    # We need to choose K to maximize the score within the time limit and memory constraints.
    # |U| = (|D*|)^K.
    # If |D*| = 5, K=10 -> 5^10 = 9,765,625. This is large but manageable for FFT?
    # FFT size would be ~ 2 * Max(U). Max(U) grows as (2s)^K.
    # If s=10, K=10, Max(U) ~ 20^10 ~ 10^13. FFT size 2*10^13 is impossible.
    # So we are limited by Max(U), not just |U|.
    # Max(U) must be small enough for FFT.
    # FFT size N_fft ~ 2 * Max(U).
    # Memory for FFT: 2 * N_fft * 8 bytes.
    # If N_fft = 10^6, mem = 16MB.
    # If N_fft = 10^7, mem = 160MB.
    # Let's cap Max(U) at 10^6 or 10^7.
    
    # Let's find the maximum K such that Max(U_K) < 10^7.
    # Max(U_K) approx (2 s_D)^K.
    # K_max = log(10^7) / log(2 s_D).
    
    s_D = max(best_atom)
    log_base = np.log(2 * s_D)
    K_max = int(np.log(10**7) / log_base)
    
    # Let's try K from 1 to K_max.
    # For each K, we can construct U and score it.
    # But constructing U for large K might be slow in Python if we do it naively.
    # We can compute the score analytically for the mixed-radix set!
    # Score = 1 + K * log(r_D) / log(2 Max(U_K) + 1).
    # Max(U_K) can be computed exactly.
    
    def calc_max_u(K, D):
        max_u = 0
        m = 1
        s_D = max(D)
        for i in range(K):
            max_u += s_D * m
            m = 2 * max_u + 1
        return max_u
    
    # Let's evaluate the score for K=1..K_max analytically.
    r_D = 0
    sumset = set()
    diffset = set()
    for a in best_atom:
        for b in best_atom:
            sumset.add(a + b)
            diffset.add(a - b)
    r_D = len(diffset) / len(sumset)
    log_r_D = np.log(r_D)
    
    best_K = 1
    best_score_analytic = 0
    best_U_analytic = None
    
    for K in range(1, K_max + 1):
        max_u = calc_max_u(K, best_atom)
        if max_u > 10**7:
            break
        denom = np.log(2 * max_u + 1)
        score = 1.0 + (K * log_r_D) / denom
        if score > best_score_analytic:
            best_score_analytic = score
            best_K = K
            
    # Now, construct the actual set U for best_K.
    # And then, do a local search on the last few levels to see if we can improve.
    # The analytic score assumes all levels are identical.
    # Maybe the last level should be different?
    # Let's generate a pool of good atoms (top 10 by efficiency).
    
    # Re-run atom search to get top 10
    top_atoms = []
    # We already found best_atom. Let's find a few more.
    # We'll just reuse the search logic but store top 10.
    # To save time, we'll just use best_atom and maybe its "neighbors" (adding/removing elements).
    
    # Actually, let's just stick with the best atom for now.
    # The gain from mixing is likely small.
    
    U = [0]
    M = 1
    for i in range(best_K):
        new_U = []
        for u in U:
            for d in best_atom:
                new_U.append(u + d * M)
        U = new_U
        M = 2 * max(U) + 1
        
        # Check time
        if time.time() > deadline - 10:
            break
            
    # Final verification
    # The analytic score might be slightly off due to the "2 max U + 1" term vs "2 Max(U_i) + 1".
    # But it's a very good approximation.
    # Let's compute the real score.
    
    real_score = _score_set(U)
    
    # If the real score is lower than expected, maybe we should try K-1?
    # Or maybe we can do a quick local search on the last element?
    # No, let's keep it simple.
    
    return U

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")
