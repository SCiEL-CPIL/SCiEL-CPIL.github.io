import time
import math
import zlib
from collections import Counter

_start_time = None
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def _time_ok():
    if _start_time is None:
        return True
    return (time.time() - _start_time) < (_DEADLINE - _SAFETY_MARGIN)

def priority(vector: tuple, n: int) -> float:
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    if not _time_ok():
        return 0.0

    # Strategy: Structural change towards "Line-Blocking Potential" proxy via spectral analysis.
    # 
    # Instead of local balance (symbol counts), we look at the global geometric structure.
    # In Z_3^n, the size of the cap set is limited by the additive structure.
    # Vectors with "high frequency" components (in the DFT sense over Z_3) are
    # more "random" or "dispersed" and less likely to align in low-frequency arithmetic
    # progressions (lines).
    # 
    # However, computing full DFT is expensive. We can approximate the "dispersion" or
    # "chaos" of a vector by looking at its projection onto specific high-frequency characters.
    # 
    # A simpler, faster structural proxy:
    # The "Line-Blocking" intuition suggests that vectors which are "orthogonal" to many
    # potential line directions are good. 
    # 
    # Let's try a different structural angle: **Local Independence**.
    # A vector is "good" if it is not too close (Hamming distance) to the "origin" of
    # structured subspaces.
    # 
    # Actually, let's look at the inspirations. Inspiration 1 used DFT magnitude.
    # Let's try a **Spectral Dispersion** score.
    # 
    # For a vector v, we can compute the power spectrum in a reduced sense.
    # But for speed, we can use a heuristic: 
    # Vectors with high "autocorrelation" are bad. 
    # Vectors that look like random noise are good.
    # 
    # Let's use a **Hash-based Perturbation** combined with **Hamming Weight Variance**.
    # 
    # Wait, the directive asks for "Line-blocking potential".
    # A vector v blocks a line {a, a+d, a+2d} if v is one of these points? No,
    # if we ADD v, we block any line that CONTAINS v.
    # The number of lines containing v is 3^(n-1). This is constant for all v!
    # So "blocking potential" in terms of count is constant.
    # 
    # The difference is WHICH lines are blocked.
    # If we block "popular" lines (lines that are likely to be formed by other high-priority vectors),
    # we might be more effective.
    # 
    # This is a second-order effect. 
    # 
    # Let's try **Conflict Minimization** via pairwise distance.
    # We want vectors that are "far" from each other? No, cap sets can have close points.
    # We want vectors that do NOT form lines.
    # 
    # Let's go back to the **DFT** idea from Inspiration 1, but optimize it.
    # Instead of full DFT, use the **Spectral Radius** of the vector's coordinate distribution?
    # 
    # Let's try **Entropy of Pairwise Differences**.
    # 
    # Actually, the most robust structural change that often works in these FunSearch tasks is
    # to use a **Deterministic Pseudorandom Priority** seeded by the vector's content,
    # but correlated with a structural property like **Balancedness of Substrings**.
    # 
    # Let's try: **Sum of Squares of Local Imbalances**.
    # Divide the vector into blocks. Calculate imbalance in each block.
    # Penalize vectors that have high local imbalance.
    # This encourages vectors that are locally balanced, which might reduce the chance
    # of forming local lines.
    
    # Block size choice: adaptive or fixed? Fixed 3 or 5?
    # Let's use a window size of 3 (matching Z_3).
    
    block_size = 3
    n_blocks = n // block_size
    
    local_penalty = 0.0
    for i in range(0, n_blocks * block_size, block_size):
        block = vector[i:i+block_size]
        c0 = block.count(0)
        c1 = block.count(1)
        c2 = block.count(2)
        # Ideal is 1,1,1. Penalty for deviation.
        # Variance of counts.
        mean_count = 1.0
        var = (c0 - mean_count)**2 + (c1 - mean_count)**2 + (c2 - mean_count)**2
        local_penalty += var

    # Handle remainder if n is not divisible by 3
    rem = n % block_size
    if rem > 0:
        block = vector[n_blocks*block_size:]
        c0 = block.count(0)
        c1 = block.count(1)
        c2 = block.count(2)
        mean_count = rem / 3.0
        var = (c0 - mean_count)**2 + (c1 - mean_count)**2 + (c2 - mean_count)**2
        local_penalty += var

    # Add a small global balance term to prevent local optima that are globally bad
    global_counts = [0, 0, 0]
    for v in vector:
        global_counts[v] += 1
    global_mean = n / 3.0
    global_penalty = sum((c - global_mean)**2 for c in global_counts)

    # Combine: Weight local balance heavily?
    # If local balance is good, the vector is "locally cap-like".
    # Priorities: Higher is better. So we want LOW penalty.
    # Return negative penalty.
    
    # Let's mix local and global.
    # local_penalty is O(1) per block, so O(n).
    # global_penalty is O(1) effectively (sum of squares).
    
    # Scaling:
    # local_penalty magnitude: ~ n_blocks * 2 (if all 0s, var=2 per block). ~ 2n/3.
    # global_penalty magnitude: ~ n (if all 0s, (n-n/3)^2 + ... ~ 2n^2/9? No, counts are ~n. (n - n/3)^2 = 4n^2/9. Sum ~ 8n^2/9.
    # This is quadratic in n. Local is linear.
    # We need to normalize.
    
    norm_local = local_penalty / max(1, n_blocks)
    norm_global = global_penalty / max(1, n)
    
    # Priority = - (alpha * norm_local + beta * norm_global)
    # Let's try alpha=1.0, beta=0.1
    score = -(1.0 * norm_local + 0.1 * norm_global)
    
    return score
