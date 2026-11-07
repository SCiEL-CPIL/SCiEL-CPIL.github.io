import time
import math
from fractions import Fraction
import numpy as np

def construct():
    deadline = time.time() + 80  # Safety margin
    MAXN = 10_000_000
    TARGET = Fraction(3, 2)
    
    # Initial best (known small pair)
    # sigma(11) = 12, sigma(6) = 12. Ratio 11/6 = 1.833. Error |1.833 - 1.5| = 0.333.
    # sigma(14) = 24, sigma(15) = 24. Ratio 14/15 = 0.933. Error |0.933 - 1.5| = 0.566.
    # sigma(16) = 31 (prime).
    # Let's find a better initial one or just rely on the search.
    # 11, 6 is a safe fallback.
    best_a, best_b = 11, 6
    best_err = abs(Fraction(11, 6) - TARGET)
    
    # 1. Compute sigma(n) for n in [1, MAXN]
    # Using NumPy for speed
    sigma_arr = np.zeros(MAXN + 1, dtype=np.int64)
    
    # Sieve method: for each d, add d to all multiples of d
    # This is O(N log N)
    # We can optimize by only iterating d up to MAXN
    # The slice assignment sigma[d::d] += d is very fast in NumPy
    
    # To save time, we can stop if time is running out, but 10^7 is usually fast enough.
    # Let's run the full sieve.
    for d in range(1, MAXN + 1):
        sigma_arr[d::d] += d
        
    if time.time() > deadline - 10:
        return [int(best_a), int(best_b)]
        
    # 2. Search for best pair
    # We want to minimize |a/b - 3/2|.
    # a ~ 1.5 b.
    # We iterate over b and check nearby a's.
    
    # Vectorized approach:
    # B = 1..MAXN
    # A_candidate = round(1.5 * B)
    # Check A_candidate + k for small k
    
    B = np.arange(1, MAXN + 1, dtype=np.int64)
    A_base = np.round(1.5 * B).astype(np.int64)
    
    # We will check offsets k from -5 to 5
    # This covers a range of |2a - 3b| roughly.
    # If a = 1.5b + delta, then 2a - 3b = 2delta.
    # We want small |2delta|.
    # So checking a close to 1.5b is correct.
    
    best_err_val = float(best_err)
    
    for k in range(-10, 11):
        A = A_base + k
        # Valid range for A
        mask = (A >= 1) & (A <= MAXN) & (A != B)
        if not np.any(mask):
            continue
            
        A_valid = A[mask]
        B_valid = B[mask]
        
        # Check if sigma(A) == sigma(B)
        # Note: sigma_arr is 0-indexed, but we allocated MAXN+1 so index n is valid for n<=MAXN.
        sigma_A = sigma_arr[A_valid]
        sigma_B = sigma_arr[B_valid]
        
        match_mask = (sigma_A == sigma_B)
        
        if np.any(match_mask):
            A_match = A_valid[match_mask]
            B_match = B_valid[match_mask]
            
            # Calculate errors
            # err = |a/b - 3/2| = |2a - 3b| / (2b)
            # We can compute this exactly using integers to avoid float precision issues initially,
            # then convert to Fraction for final comparison or just use float if we are careful.
            # The task asks for the binary64 serialization of the rational error.
            # So we need the exact Fraction to determine the best one, then return the pair.
            
            # To find the minimum, we can compare fractions.
            # num_err = |2a - 3b|
            # den_err = 2b
            # We want to minimize num_err / den_err.
            
            num_err = np.abs(2 * A_match - 3 * B_match)
            den_err = 2 * B_match
            
            # We can sort by the ratio or just iterate through matches.
            # The number of matches might be small.
            # Let's find the index of the minimum ratio.
            # ratio = num_err / den_err
            # To avoid division by zero (den_err is at least 2), this is safe.
            
            # We can compute the ratio as float for a quick filter, then verify with Fraction.
            # Or just use Fraction for the candidates found.
            
            # There might be many matches. Let's filter for small num_err first.
            # Ideally num_err = 1.
            
            # Let's just compute the float ratio and find the min index.
            ratios = num_err.astype(np.float64) / den_err.astype(np.float64)
            
            min_idx = np.argmin(ratios)
            current_a = int(A_match[min_idx])
            current_b = int(B_match[min_idx])
            current_err = abs(Fraction(current_a, current_b) - TARGET)
            
            if current_err < best_err:
                best_err = current_err
                best_a, best_b = current_a, current_b
                best_err_val = float(current_err)
                
                # If we find error 0 (impossible for integers a!=b, 2a!=3b unless a=3k, b=2k, 
                # but then sigma(3k) == sigma(2k)?), we can't stop because 0 is invalid?
                # "raw value of exactly 0 cannot be scored".
                # So we must find the *smallest non-zero* error.
                # If current_err == 0, we should skip it?
                # But 2a=3b implies a=3k, b=2k.
                # If sigma(3k) == sigma(2k), then error is 0.
                # The task says "raw value of exactly 0 ... counts as invalid".
                # So we must ensure we don't return a pair with error 0.
                # We should find the next best if we hit 0.
                
                if current_err == 0:
                    # We need to find the next best in this batch or other batches.
                    # Since we are iterating k, we might find non-zero ones later or in other k's.
                    # But if this is the global min, we need the runner up.
                    # Let's collect all valid non-zero errors for this k and find the min.
                    pass # Handled below

    # The above loop might miss the global min if the global min has error 0 and we skip it,
    # or if we just take the min of the current k.
    # Let's refine: Collect all candidate pairs (a,b) with sigma(a)==sigma(b) and a!=b.
    # Then find the min non-zero error.
    
    # Given the speed, we can just store the best non-zero error found so far.
    # If we find 0, we ignore it and keep looking for the smallest non-zero.
    
    # Let's re-run the search logic more carefully to handle the 0 case.
    
    best_a, best_b = 11, 6
    best_err = abs(Fraction(11, 6) - TARGET)
    
    # We will store the best non-zero error.
    
    for k in range(-20, 21):
        A = A_base + k
        mask = (A >= 1) & (A <= MAXN) & (A != B)
        if not np.any(mask):
            continue
            
        A_valid = A[mask]
        B_valid = B[mask]
        
        sigma_A = sigma_arr[A_valid]
        sigma_B = sigma_arr[B_valid]
        
        match_mask = (sigma_A == sigma_B)
        
        if np.any(match_mask):
            A_match = A_valid[match_mask]
            B_match = B_valid[match_mask]
            
            num_err = np.abs(2 * A_match - 3 * B_match)
            den_err = 2 * B_match
            
            # Exclude num_err == 0
            non_zero_mask = (num_err > 0)
            if not np.any(non_zero_mask):
                continue
                
            A_nz = A_match[non_zero_mask]
            B_nz = B_match[non_zero_mask]
            num_nz = num_err[non_zero_mask]
            den_nz = den_err[non_zero_mask]
            
            # Find min ratio
            ratios = num_nz.astype(np.float64) / den_nz.astype(np.float64)
            min_idx = np.argmin(ratios)
            
            current_a = int(A_nz[min_idx])
            current_b = int(B_nz[min_idx])
            current_err = abs(Fraction(current_a, current_b) - TARGET)
            
            if current_err < best_err:
                best_err = current_err
                best_a, best_b = current_a, current_b

    if time.time() > deadline - 5:
        return [int(best_a), int(best_b)]

    return [int(best_a), int(best_b)]
