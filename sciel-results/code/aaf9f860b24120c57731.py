"""Cap set greedy priority function with pair entropy and hash-based diversity."""
import time
import zlib
import math

# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Priority for greedy cap set construction in {0,1,2}^n.
    
    Structural Change:
    Instead of 1D position spread, we use 2D Pair Entropy.
    Vectors with diverse coordinate pairs are more "generic" and less likely
    to belong to structured subspaces (like {0,2}^n) that form large but
    sub-optimal cap sets.
    
    Components:
    1. Pair Entropy: Reward vectors where all 9 possible pairs (0,0)..(2,2) 
       appear with roughly equal frequency.
    2. Symbol Balance: Reward near-uniform distribution of 0,1,2.
    3. Hash Diversity: Small random-looking term to break symmetries.
    """
    v = vector
    
    # 1. Pair Entropy
    # Count occurrences of each pair (v[i], v[j]) for i != j
    # There are 9 possible pairs.
    pair_counts = [0] * 9
    
    # Optimization: Only need to count i < j and double, or just i != j.
    # For entropy, we need the distribution over all ordered pairs or unordered?
    # APs are ordered in a sense (a,b,c), but the pair difference a-b is directed.
    # Let's use ordered pairs (v[i], v[j]) for i != j.
    # Total pairs = n * (n - 1)
    
    # Fast loop for small n
    for i in range(n):
        vi = v[i]
        for j in range(n):
            if i != j:
                vj = v[j]
                idx = vi * 3 + vj
                pair_counts[idx] += 1
                
    total_pairs = n * (n - 1)
    if total_pairs == 0:
        entropy = 0.0
    else:
        entropy = 0.0
        for count in pair_counts:
            if count > 0:
                p = count / total_pairs
                entropy -= p * math.log2(p)
        
        # Normalize entropy to [0,1]. Max entropy is log2(9) ~ 3.17
        max_entropy = math.log2(9)
        entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # 2. Symbol Balance
    counts = [0, 0, 0]
    for val in v:
        counts[val] += 1
        
    ideal = n / 3.0
    balance_penalty = 0.0
    for c in counts:
        balance_penalty += (c - ideal) ** 2
        
    # Normalize balance penalty. Max penalty when all in one bin: n^2.
    # So penalty / n^2 is in [0,1].
    balance_score = 1.0 - (balance_penalty / (n * n + 1e-12))

    # 3. Hash Diversity
    # Create a deterministic pseudo-random number based on the vector
    # This breaks symmetry without introducing actual randomness (which would be non-deterministic across runs if not seeded)
    # The evaluator might run multiple times, so deterministic is safer for reproducibility, 
    # but "pseudo-random" based on content is fine.
    vec_str = ''.join(map(str, v))
    h = zlib.crc32(vec_str.encode('utf-8')) & 0xFFFFFFFF
    random_score = (h / 0xFFFFFFFF)
    
    # Combine
    # Weight entropy heavily as it captures the 2D structure
    # Weight balance moderately
    # Small weight for random to break ties
    priority_val = 2.0 * entropy + 1.0 * balance_score + 0.05 * random_score
    
    return float(priority_val)
# EVOLVE-BLOCK-END
