import time
import zlib

_start_time = time.time()
_DEADLINE = _start_time + 280.0

# Precompute pseudo-random weights for the bilinear form
# We use a fixed seed to ensure determinism
import random
_rng = random.Random(12345)
_MAX_N = 64
# Weights for interactions between coordinates i and j
# We only need i <= j to save space/computation, or just use a matrix
# To keep it fast, we use a diagonal-like or sparse interaction model?
# No, a dense matrix of size n x n is O(n^2). n is likely small (< 20 or 30 in these tasks).
# If n is large, O(n^2) might be slow.
# Let's check typical n. FunSearch cap set usually n=8, 9, 10... maybe up to 12.
# O(n^2) is fine.

# However, we don't know n ahead of time.
# We can generate weights on the fly based on indices if we use a deterministic formula.
# Or we can use a fixed set of "basis" vectors for the interaction.

# Let's use a simpler, faster structural metric that captures "complexity":
# The "Rigidity" of the vector.
# A vector is rigid if it has high variance in its coordinate differences.

# Let's stick to a robust heuristic:
# 1. Balance (Symbol counts)
# 2. Positional Weighted Sum (Linear term with random weights)
# 3. Pairwise Difference Sum (Quadratic term)

# Pre-generate random linear weights for up to max_n
# This helps break symmetry
_MAX_N_SUPPORT = 100
_linear_weights = [_rng.uniform(-1.0, 1.0) for _ in range(_MAX_N_SUPPORT)]

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    # 1. Compute Symbol Counts (Balance)
    c0, c1, c2 = 0, 0, 0
    # Also compute linear weighted sum for symmetry breaking
    lin_score = 0.0
    
    for i, val in enumerate(vector):
        if val == 0:
            c0 += 1
        elif val == 1:
            c1 += 1
        elif val == 2:
            c2 += 1
        
        # Use modulo indexing if n is larger than precomputed weights
        idx = i % _MAX_N_SUPPORT
        lin_score += val * _linear_weights[idx]
        
    # Balance Score: How close to n/3 of each?
    ideal = n / 3.0
    # Variance from ideal
    var = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    # Normalize: Max variance is when one count is n, others 0 -> (n-n/3)^2 + ... 
    # Actually, max var is roughly 2/3 n^2.
    # We want high balance, so we negate or invert.
    # balance_score is in [0, 1]
    max_var = (2.0 * n * n) / 3.0
    if max_var > 0:
        balance_score = 1.0 - (var / max_var)
    else:
        balance_score = 1.0

    # 2. Compute Pairwise Interaction Score (Quadratic Proxy)
    # This term captures "structure" that might form APs.
    # We compute sum_{i<j} (v_i - v_j)^2.
    # This is O(n^2).
    # Note: (v_i - v_j)^2 in integers is not mod 3, but it captures "spread".
    # A vector with all same values has spread 0.
    # A vector with mixed values has high spread.
    # Cap sets often benefit from "spread" in the Hamming sense, but in Z_3,
    # we want to avoid linear dependencies.
    
    # To make it O(n), we can use:
    # Sum (v_i - v_j)^2 = n * Sum v_i^2 - (Sum v_i)^2
    # Let's use this algebraic identity for speed.
    
    sum_v = sum(vector)
    sum_v_sq = sum(val * val for val in vector)
    
    # Spread measure
    spread = n * sum_v_sq - sum_v * sum_v
    
    # Normalize spread
    # Max spread is when half are 0 and half are 2? Or 0 and 1?
    # For n elements, max variance is when values are 0 and 2 (distance 2).
    # Max spread approx 2/3 n^2?
    # Let's just normalize by n^2 to get a 0..1 scale roughly.
    if n > 1:
        spread_score = spread / (n * n)
    else:
        spread_score = 0.0

    # 3. Hash for Tie-Breaking
    # Use zlib.crc32 for a fast, strong-ish hash
    # Convert vector to bytes for hashing
    # Vector values are 0, 1, 2.
    # We can create a bytes object
    try:
        # Optimization: only hash if n is small enough to not be slow?
        # crc32 is very fast.
        # Pad or truncate? Just pack the values.
        # Python int to bytes conversion might be slow if done naively.
        # Let's use a simple manual hash which is faster for small tuples
        h = 0
        for val in vector:
            h = (h * 131 + val) & 0xFFFFFFFF
        hash_score = h / 4294967296.0
    except Exception:
        hash_score = 0.0

    # Combine Scores
    # We want high Balance.
    # We want High or Low Spread?
    # "Smooth" vectors (low spread) are often the "center" and might be picked first,
    # but they might form APs with other smooth vectors.
    # "Noisy" vectors (high spread) are more random.
    # The directive suggests penalizing structures prone to APs.
    # Additive energy is high for sets with high "correlation".
    # Let's assume we want to *prioritize* vectors that are "structurally distinct"
    # from the average.
    
    # Let's try: Score = 2.0 * Balance + 0.5 * (1.0 - spread_score) + 0.1 * lin_score + 0.01 * hash_score
    # Why (1 - spread)? Because we might prefer "balanced" in value distribution too?
    # Actually, let's look at the inspirations again.
    # Inspiration 1: Spectral Smoothness (penalizing high freq).
    # Low freq = Smooth = Low Spread?
    # If we penalize high freq, we prefer smooth vectors.
    # So let's prefer LOW spread (smooth vectors).
    
    # However, if ALL vectors are smooth, they might form APs?
    # The Balance term already ensures symbol counts are good.
    # The Spread term adds a secondary preference for "coherent" vectors.
    
    # Let's invert the spread preference to match "Smoothness" (Inspiration 1)
    # Smoothness = Low high-frequency content = Low variance of differences = Low Spread.
    
    smoothness_score = 1.0 - spread_score # Higher is smoother
    
    # Linear score is a random walk, adds variety.
    
    score = 3.0 * balance_score + 1.0 * smoothness_score + 0.5 * lin_score + 0.05 * hash_score
    
    return float(score)
