import time
import random
import numpy as np

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    # Size of sumset U+U: values in [0, 2*mx]
    # Size of diffset U-U: values in [-mx, mx], shift by mx -> [0, 2*mx]
    N = 2 * mx + 1
    # Use next power of 2 for FFT efficiency if needed, but let's just use N or slightly larger
    # Actually, convolution size is n + n - 1. We care about counts > 0.
    
    # Efficient counting via FFT
    # Create signal of ones at positions in U
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    # Sumset: convolution of sig with sig
    # Result index k corresponds to sum k
    # Diffset: convolution of sig with reversed sig
    # Let rev_sig be sig reversed. 
    # Conv(sig, rev_sig)[k] counts pairs (a,b) such that a + (mx - b) = k => a - b = k - mx.
    # So diffset size is count of non-zero entries in conv(sig, rev_sig).
    
    # To avoid circular convolution artifacts, we need size >= 2*N - 1? 
    # No, max sum is 2*mx = N-1. Max diff is mx.
    # Linear convolution of two length N signals has length 2N-1.
    # We only care about indices 0..2*mx for sum (which is 0..N-1).
    # For diff, we care about indices 0..2*mx in the reversed convolution?
    # Let's just do full linear convolution via FFT size 2N or next power of 2.
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6) # Sums range 0..2*mx which is 0..N-1
    
    # Diffset
    # rev_sig[i] = sig[N-1-i]
    # We want counts of a-b.
    # Standard trick: conv(sig, flip(sig))
    # Let's construct flip_sig
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    # The index k in S_diff corresponds to a + (N-1-b) = k => a-b = k - (N-1).
    # We want to know if a-b exists.
    # a-b ranges from -mx to mx.
    # k ranges from 0 to 2*mx.
    # Let's check non-zero entries in S_diff.
    # Specifically, we want to map back to valid differences.
    # Since U contains 0, min diff is -mx, max diff is mx.
    # The convolution covers all combinations.
    # diffset_size = np.sum(S_diff > 1e-6) is safe because only valid diffs will have mass.
    # Wait, is it possible for invalid indices to have mass?
    # a in [0, mx], b in [0, mx].
    # a + (N-1-b) = a - b + N - 1.
    # Min: 0 - mx + N - 1 = N - 1 - mx.
    # Max: mx - 0 + N - 1 = N - 1 + mx.
    # So the non-zero part is in [N-1-mx, N-1+mx].
    # Total range is 2*mx+1 = N.
    # So diffset_size is simply the count of non-zero entries in S_diff.
    
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
    
    # Start with a small base set
    # [0, 1] is minimal. [0, 1, 3] is the seed.
    # Let's start with a set that allows growth.
    # Geometric sequences often have small sumsets.
    # U = {0, 1, 2, 4, 8, ...}
    # Sumset size is roughly 2*max. Diffset size is roughly 2*max. Ratio ~ 1. Score ~ 1.
    # We want Ratio > 1.
    # This happens if sumset is "small" (additive structure) but diffset is "large".
    # But for finite sets, |U-U| >= |U+U| is not always true?
    # Actually, for symmetric sets U=-U, |U-U|=|U+U|.
    # If U is not symmetric, |U-U| can be larger.
    # Example: U = {0, 10, 11}.
    # U+U = {0, 10, 11, 20, 21, 22}. Size 6.
    # U-U = {-11, -10, -1, 0, 1, 10, 11}. Size 7.
    # Ratio 7/6 = 1.16.
    # Max U = 11. 2*11+1 = 23. log(23) ~ 3.1. log(1.16) ~ 0.15.
    # Score = 1 + 0.15/3.1 = 1.05.
    
    # Strategy: Start with a set, try to add elements.
    # Candidate elements: powers of 2, or random, or local neighbors.
    
    best_set = [0, 1]
    best_score = _score_set(best_set)
    
    # Try to grow the set
    # We will maintain a list of candidates and try adding them.
    # If adding improves score, keep it. If not, try removing or swapping.
    
    # Let's use a randomized hill climbing.
    # State: sorted list of integers.
    
    # Initial candidates
    candidates = set()
    # Add powers of 2
    p = 1
    while p < 10000:
        candidates.add(p)
        p *= 2
    # Add some randoms
    for _ in range(100):
        candidates.add(random.randint(1, 500))
    
    current_set = [0, 1]
    current_score = best_score
    
    # Sort current set
    current_set.sort()
    
    # Local search
    while time.time() < deadline:
        improved = False
        
        # 1. Try adding candidates
        # Filter candidates to those not in current_set and > max(current_set) * 0.5 maybe?
        # Or just any candidate.
        # To avoid O(N^2) check, we can just iterate candidates.
        
        # Sort candidates by "potential" (e.g., smaller ones first to keep max U small initially?)
        # Or just random order.
        
        # Let's try adding the candidate that gives best immediate improvement.
        # This is expensive. Let's do random restarts of local search or random moves.
        
        # Move type 1: Add random candidate
        if len(candidates) > 0:
            c = random.choice(list(candidates))
            if c not in current_set:
                new_set = sorted(current_set + [c])
                new_score = _score_set(new_set)
                if new_score > current_score + 1e-9:
                    current_set = new_set
                    current_score = new_score
                    improved = True
                    # Remove from candidates? No, maybe we want to add more nearby.
                    # Add neighbors of c to candidates
                    for d in range(1, 10):
                        if c - d > 0:
                            candidates.add(c - d)
                        candidates.add(c + d)
        
        # Move type 2: Remove random element (except 0)
        if len(current_set) > 2:
            # Pick random index > 0
            idx = random.randint(1, len(current_set) - 1)
            new_set = current_set[:idx] + current_set[idx+1:]
            new_score = _score_set(new_set)
            if new_score > current_score + 1e-9:
                current_set = new_set
                current_score = new_score
                improved = True
        
        # Move type 3: Swap
        if len(current_set) > 3:
            idx1 = random.randint(1, len(current_set) - 1)
            idx2 = random.randint(1, len(current_set) - 1)
            if idx1 != idx2:
                val = current_set[idx1]
                new_val = random.randint(1, max(current_set) * 2 + 10)
                if new_val not in current_set:
                    new_set = current_set[:]
                    new_set[idx1] = new_val
                    new_set.sort()
                    new_score = _score_set(new_set)
                    if new_score > current_score + 1e-9:
                        current_set = new_set
                        current_score = new_score
                        improved = True
        
        if not improved:
            # Stuck, randomize a bit
            # Try adding a random new large number
            c = random.randint(1, 5000)
            if c not in current_set:
                new_set = sorted(current_set + [c])
                new_score = _score_set(new_set)
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                else:
                    # Give up on this direction, maybe restart from a smaller set?
                    # Or just accept if it's not worse? No, we want to maximize.
                    pass
        
        if current_score > best_score:
            best_set = current_set[:]
            best_score = current_score
            
        # Prune candidates to keep list manageable
        if len(candidates) > 5000:
            # Keep only candidates near current max or small
            mx = max(current_set) if current_set else 1
            # Keep candidates < mx * 2
            candidates = {c for c in candidates if c < mx * 2}
            if len(candidates) == 0:
                candidates = set(range(1, 100))
                
    return best_set

# Test locally
if __name__ == "__main__":
    # Quick test
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")
