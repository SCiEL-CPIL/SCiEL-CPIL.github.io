import time
import math

_T0 = time.time()
_DEADLINE = _T0 + 280.0

def _mix64(h):
    """64-bit MurmurHash3 finalizer for high-quality mixing."""
    h ^= h >> 33
    h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    return h

# Precompute weights for the "spectral" penalty.
# The idea is to penalize vectors that are "low frequency" in a certain sense.
# In F_3^n, the all-ones vector (1,1,...,1) and its multiples are low frequency.
# Vectors with many 1s are "closer" to the center.
# Let's define a penalty based on the deviation from a "random" distribution of 1s.
# Or better, let's look at the "energy" of the vector.
# A vector v has "energy" sum(v_i^2).
# For {0,1,2}, 0^2=0, 1^2=1, 2^2=4.
# If we want to penalize low frequency (correlation with all-ones), we might look at sum(v_i).
# But the inspiration suggests penalizing high correlation to the all-ones vector.
# The all-ones vector is (1,1,...,1).
# Correlation is proportional to sum(v_i - 1).
# Let's try a different structural feature:
# The number of 1s is a good proxy for "distance from corners".
# But let's combine it with a hash-based perturbation that is scaled by the "rarity" of the composition.
# This ensures that rare compositions (like all 1s or all 0s) get more hash-driven randomness,
# while common compositions (like balanced ones) are more strictly ordered by their structural score.

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    # 1. Structural Score: Balance of 1s
    # Count 1s
    count_1 = 0
    count_0 = 0
    count_2 = 0
    for v in vector:
        if v == 1:
            count_1 += 1
        elif v == 0:
            count_0 += 1
        else:
            count_2 += 1
    
    # Target is n/2 for count_1.
    # We want to prioritize vectors with count_1 close to n/2.
    # Let's use a score that is high for balanced vectors.
    # base_score = n - abs(2*count_1 - n)
    
    # Let's refine the structural score to be more "spectral".
    # In the Fourier domain on Z_3^n, the function f(x) = 1 if x is in the cap set, else 0.
    # We want the cap set to have small Fourier coefficients for non-zero frequencies.
    # A greedy heuristic: pick points that are "orthogonal" to the current set's Fourier support?
    # Too complex for a single priority function.
    
    # Let's stick to the composition-based structural score but add a twist.
    # The "spectral" inspiration suggests penalizing low-frequency components.
    # The DC component (frequency 0) is always present.
    # The first non-zero frequencies correspond to single-coordinate changes.
    # Vectors that are "smooth" (slowly varying) are low frequency.
    # Vectors that are "noisy" are high frequency.
    # In {0,1,2}, a vector with many 1s is not necessarily smooth.
    # Let's use the variance of the coordinates as a proxy for "frequency content".
    # High variance -> high frequency -> potentially better for cap sets?
    # Or low variance -> low frequency -> bad?
    
    # Let's calculate the sum of squares (energy).
    # Energy = sum(v_i^2) = count_0*0 + count_1*1 + count_2*4
    energy = count_1 + 4 * count_2
    
    # Mean energy for a random vector:
    # E[v_i^2] = (0 + 1 + 4)/3 = 5/3
    # Expected energy = 5n/3
    
    # Deviation from expected energy.
    # Let's try to prioritize vectors that are "average" in energy, i.e., not too extreme.
    # This is similar to the balance of 1s.
    
    # Let's combine the balance of 1s and the energy deviation.
    # balance_score = n - abs(2*count_1 - n)  # Range [0, n]
    # energy_dev = abs(energy - 5*n/3)
    # We want to minimize energy_dev.
    
    # Let's use a composite structural score.
    # Primary: Balance of 1s (count_1 close to n/2).
    # Secondary: Balance of 0s and 2s (count_0 and count_2 close to n/4).
    
    # Let's define a "rarity" score.
    # The number of vectors with a given (count_0, count_1, count_2) is multinomial.
    # We can approximate the log-rarity.
    # log_rarity = log(n!) - log(count_0!) - log(count_1!) - log(count_2!)
    # Using Stirling's approximation or just precomputing factorials for small n.
    # Since n is likely small (e.g., 8-10), we can use math.log(math.factorial(x)).
    
    try:
        log_fact_0 = math.log(math.factorial(count_0))
        log_fact_1 = math.log(math.factorial(count_1))
        log_fact_2 = math.log(math.factorial(count_2))
        log_fact_n = math.log(math.factorial(n))
        
        log_rarity = log_fact_n - log_fact_0 - log_fact_1 - log_fact_2
        # Normalize log_rarity to [0, 1] roughly.
        # Max log_rarity is for balanced composition.
        # Min log_rarity is for extreme compositions.
        # We want to prioritize common compositions? Or rare ones?
        # If we prioritize common compositions, we fill the "core" of the cube first.
        # This might lead to a dense core, which is bad for cap sets.
        # If we prioritize rare compositions, we pick corners first.
        # Corners form a cap set {0,2}^n.
        # So picking corners first is a good strategy to get a large cap set.
        # Therefore, we should prioritize RARE compositions (low log_rarity).
        # So structural_score = -log_rarity.
        
        # But wait, the inspiration said "Balanced-composition routing" (prioritize common) worked well (1.0586).
        # Why? Maybe because it avoids the "corner" subspace saturation?
        # Let's stick to the proven "Balanced" strategy but refine it.
        # Let's use the balance of 1s as the primary key.
        
        structural_score = n - abs(2 * count_1 - n)
        
        # Add a small penalty for extreme energy?
        # energy_dev = abs(energy - 5*n/3)
        # structural_score -= 0.1 * energy_dev / n
        
    except Exception:
        structural_score = n - abs(2 * count_1 - n)

    # 2. Hash-based Perturbation
    # Compute the index in base 3.
    idx = 0
    for v in reversed(vector):
        idx = idx * 3 + v
    
    # Mix with n to ensure different n's have different orderings.
    seed_val = idx ^ (n * 0x9E3779B97F4A7C15)
    mixed = _mix64(seed_val)
    
    # Scale hash to [0, 1)
    hash_fraction = mixed / (2**64)
    
    # Combine structural score and hash.
    # We want the structural score to dominate.
    # But we want the hash to break ties effectively.
    # Also, we want to introduce some "noise" into the structural score to avoid rigid layers.
    # Let's scale the hash by a factor that depends on the "rarity" or "balance".
    # If the composition is very common, the layer is large, so we need good mixing.
    # If the composition is rare, the layer is small, so mixing is less critical.
    
    # Let's use a fixed small perturbation.
    # perturbation = hash_fraction * 0.1
    # result = structural_score + perturbation
    
    # Let's try a different combination:
    # result = structural_score * 10 + hash_fraction
    # This ensures that a difference of 1 in structural score is always larger than any hash difference.
    
    result = structural_score * 10.0 + hash_fraction
    
    return float(result)

# EVOLVE-BLOCK-END
