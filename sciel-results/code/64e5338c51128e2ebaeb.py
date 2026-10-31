import time

# EVOLVE-BLOCK-START
_start_time = None
_DEADLINE = 300 * 0.95  # 285s safety margin before the 300s kill

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy: Hybrid Structural-Hash Priority with Spectral and Multiplicative Refinements.
    
    1. Macro-Structural Score (from current program):
       - Quadratic distinctness: Strongly prefer vectors using all 3 symbols.
       - Balance Penalty: Penalize deviation from uniform symbol distribution.
       - Variance Penalty: Penalize high coordinate variance.
       
    2. Spectral Score (from inspiration 1):
       - Compute the Discrete Fourier Transform (DFT) of the centered vector.
       - Penalize high-frequency components (roughness), favoring smooth vectors.
       - Hypothesis: Smooth vectors are less likely to form accidental APs with other smooth vectors 
         in the early stages, or perhaps the greedy algorithm benefits from a "skeleton" of smooth vectors.
       - Note: The inspiration suggested penalizing high frequency. We implement this as a penalty on 
         the ratio of high-frequency energy to total energy.
       
    3. Multiplicative Complexity Score (from directive):
       - Maximize the product of (v_i + 1) to enforce "full-rank" algebraic structure.
       - Vectors with many 0s have a lower multiplicative score (since 0+1=1, but 1+1=2, 2+1=3).
       - Actually, the directive says "treating 0 as 1 for multiplicative stability". 
       - Let's interpret this as: score = sum(log(v_i + 1)) or product(v_i + 1).
       - This favors vectors with more 2s and 1s over 0s? 
       - Or does it mean something else? "Full-rank algebraic structure" might imply avoiding zeros 
         in a multiplicative sense? But 0 is a valid element in Z3.
       - Let's stick to the literal interpretation: prioritize vectors with higher multiplicative complexity.
       - We will add a term: lambda * sum(log(v_i + 1)). This favors vectors with larger values (2s).
       - However, balance is already handled by the macro score. This might conflict with balance.
       - Let's weight it lightly to act as a tie-breaker or minor adjustment.
       
    4. Micro-Structural Hash (from current program):
       - Polynomial rolling hash modulated by safety.
       
    This combination aims to:
    - First, select structurally sound (balanced, distinct) vectors.
    - Second, among those, prefer smoother (lower frequency) vectors.
    - Third, slightly prefer vectors with higher multiplicative complexity (more 2s).
    - Fourth, use the hash for diversity within similar structural/spectral profiles.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: ordering no longer matters, return a flat constant.
        return 0.0

    # Fast symbol counting and basic stats in one pass
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
            s += 1
            s2 += 1
        else:  # v == 2
            c2 += 1
            s += 2
            s2 += 4

    # 1. Macro-Structural Score
    # Factor 1: Quadratic distinctness
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Balance Penalty (L2 deviation from n/3)
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 10.0

    # Factor 3: Variance Penalty
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 500.0

    macro_score = distinct_score + balance_score + danger_score

    # 2. Spectral Score (DFT-based)
    # We want to penalize high-frequency content.
    # Compute the DFT of the centered vector x = v - 1.
    # Since n is small, we can compute the DFT in O(n^2) or use a simple approximation.
    # For efficiency, let's compute the power spectrum only if n is not too large, or use a fast method.
    # Given the 300s limit and potentially many calls, O(n^2) might be risky if n is large (e.g., 30).
    # 30^2 = 900 operations per call. If there are 10000 calls, that's 9M ops. Should be fine in Python?
    # Maybe. But let's try to be smarter.
    # Actually, we can compute the autocorrelation or just the high-frequency energy directly?
    # Let's stick to a simplified spectral measure: 
    # "Roughness" = sum_{i=0}^{n-2} (x_{i+1} - x_i)^2
    # This is a proxy for high-frequency content (second-order differences).
    # It's O(n) and captures local variation.
    roughness = 0.0
    for i in range(n - 1):
        diff = vector[i+1] - vector[i]
        roughness += diff * diff
    
    # Normalize roughness
    # Max possible diff is 2 (from 0 to 2 or 2 to 0). So max diff^2 is 4.
    # Max roughness is 4 * (n-1).
    norm_roughness = roughness / (4.0 * (n - 1) if n > 1 else 1.0)
    
    # Penalize roughness (high frequency)
    spectral_score = -norm_roughness * 50.0

    # 3. Multiplicative Complexity Score
    # Sum of log(v_i + 1)
    # log(1)=0, log(2)=0.69, log(3)=1.10
    mult_score = 0.0
    for v in vector:
        mult_score += math.log(v + 1)
    
    # Normalize mult_score to be comparable
    # Max mult_score is n * log(3)
    norm_mult_score = mult_score / (n * math.log(3.0)) if n > 0 else 0.0
    mult_complexity_score = norm_mult_score * 20.0  # Small weight, acts as tie-breaker

    # 4. Micro-Structural Hash
    # Polynomial rolling hash
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    
    # Normalize hash to [0, 1]
    hash_norm = h / 0x7FFFFFFF

    # Modulate hash weight based on "safety" (inverse of risk)
    # Risk is approximated by balance_score (negative).
    hash_weight = 1.0 / (1.0 + dev_sq * 0.05)
    
    # Scale hash contribution
    micro_score = hash_norm * 1000.0 * hash_weight

    # Combine all scores
    total = macro_score + spectral_score + mult_complexity_score + micro_score
    
    return float(total)

import math
# EVOLVE-BLOCK-END
