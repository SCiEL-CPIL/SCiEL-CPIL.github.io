"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin before 60s kill

    # Precompute all arithmetic progressions as (start, step, indices)
    # For each step d >= 1 and start s, collect all indices s, s+d, s+2d, ... < n
    # We need max |sum| over any contiguous subsequence of the progression
    # Key insight: for a fixed (start, step), the AP is a sequence. The max abs sum
    # over any subsequence = max over all prefix differences.
    # Actually, "any arithmetic progression contained in [1,256]" means:
    # pick start a, step d>=1, length L>=1 such that a+(L-1)*d <= n.
    # The sum is x[a] + x[a+d] + ... + x[a+(L-1)*d].
    # We want to minimize max over all (a,d,L) of |that sum|.
    
    # Efficient evaluation: for each (a, d), compute the sequence of signs along
    # that progression, then compute prefix sums. The max absolute sum over all
    # contiguous subsequences = max(prefix) - min(prefix) where prefix[0]=0.
    # But we also need |sum|, so it's max(max_prefix - min_prefix, |max_prefix|, |min_prefix|)
    # Actually no: the sum of elements from position i to j-1 is prefix[j] - prefix[i].
    # Max absolute value = max over i<j of |prefix[j]-prefix[i]| = max(prefix) - min(prefix).
    
    # Precompute: for each (start, step), store the list of 0-based indices
    ap_data = []
    for step in range(1, n + 1):
        for start in range(n):
            if start + step * 0 >= n:
                break
            # Collect indices: start, start+step, start+2*step, ... < n
            indices = list(range(start, n, step))
            if len(indices) >= 1:
                ap_data.append((start, step, indices))
    
    # But this is too many. Let's think about what matters.
    # The number of APs is sum over step d of (n - d) * something... 
    # Actually for step d, start ranges 0..n-1, and length is floor((n-1-start)/d)+1 if start < n.
    # Total APs: sum_{d=1}^{n} sum_{s=0}^{n-1} max(1, floor((n-1-s)/d)+1) ... this is O(n^2 log n) or so.
    # For n=256, this is manageable but let's be smart.
    
    # Actually, let me reconsider. The AP is defined by (a, d, L) where a is 1-based start.
    # 0-based: a0 = a-1. Indices: a0, a0+d, ..., a0+(L-1)*d, all < n.
    # For a fixed (a0, d), the maximum length is floor((n-1-a0)/d)+1.
    # Any shorter AP starting at a0 with step d is a prefix of this.
    # So for each (a0, d), we just need the full sequence and compute max abs subarray sum.
    
    # Let me restructure: group by (start, step)
    ap_list = []  # list of arrays of indices
    for step in range(1, n + 1):
        for start in range(n):
            if start >= n:
                break
            # indices: start, start+step, start+2*step, ... < n
            idx = np.arange(start, n, step)
            if len(idx) >= 1:
                ap_list.append(idx)
    
    # Number of APs
    n_ap = len(ap_list)
    
    def evaluate(signs):
        """Return max absolute AP sum for given signs array."""
        max_abs = 0
        for idx in ap_list:
            s = signs[idx]
            prefix = np.concatenate([[0], np.cumsum(s)])
            m = prefix.max() - prefix.min()
            if m > max_abs:
                max_abs = m
        return max_abs
    
    # Initialize with Thue-Morse sequence (0-based: t(i) = parity of popcount(i))
    # Thue-Morse: t(0)=0, t(1)=1, t(2)=1, t(3)=0, ...
    # Map to signs: 0 -> +1, 1 -> -1
    def thue_morse_signs(n):
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            # Count bits set
            cnt = bin(i).count('1')
            signs[i] = 1 if cnt % 2 == 0 else -1
        return signs
    
    signs = thue_morse_signs(n)
    best_cost = evaluate(signs)
    
    # Local search: try flipping individual signs
    rng = random.Random(42)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle order of positions to try
        positions = list(range(n))
        rng.shuffle(positions)
        for pos in positions:
            if time.time() >= deadline - 5.0:
                break
            # Try flipping pos
            signs[pos] = -signs[pos]
            new_cost = evaluate(signs)
            if new_cost < best_cost:
                best_cost = new_cost
                improved = True
            else:
                signs[pos] = -signs[pos]  # revert
    
    # Try a few more random restarts with perturbation if time allows
    if time.time() < deadline - 10.0:
        for restart in range(20):
            if time.time() >= deadline - 5.0:
                break
            # Perturb best solution
            perturbed = signs.copy()
            for _ in range(5):
                p = rng.randint(0, n - 1)
                perturbed[p] = -perturbed[p]
            pert_cost = evaluate(perturbed)
            if pert_cost < best_cost:
                signs = perturbed
                best_cost = pert_cost
            # Quick local search on perturbed
            for pos in range(n):
                if time.time() >= deadline - 5.0:
                    break
                perturbed[pos] = -perturbed[pos]
                nc = evaluate(perturbed)
                if nc < best_cost:
                    best_cost = nc
                    signs = perturbed.copy()
                else:
                    perturbed[pos] = -perturbed[pos]
    
    return [int(x) for x in signs]
# EVOLVE-BLOCK-END
