"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Priority for greedy cap set construction.
    Lower priority value = higher preference (added first).
    
    Uses a bitwise parity-based approach on the ternary representation
    combined with structural heuristics to break correlation between
    local symbol balance and global line formation.
    """
    # Convert to numpy array for efficient computation
    v = np.array(vector, dtype=np.int64)
    
    # Count coordinate frequencies
    counts = np.bincount(v, minlength=3)
    
    # Entropy of coordinate distribution: more uniform = more dangerous
    probs = counts / counts.sum() if counts.sum() > 0 else np.array([1/3, 1/3, 1/3])
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # L1 distance from uniform: vectors far from uniform are "safer"
    uniform_dist = np.sum(np.abs(counts - n/3))
    
    # Number of zero coordinates
    num_zeros = counts[0]
    
    # Base structural score: prefer vectors with low entropy (less symmetric)
    # and high zero count (less likely to form lines)
    base_score = entropy * 1.0 - uniform_dist * 0.3 - num_zeros * 0.2
    
    # Bitwise parity-based chaotic interleaving
    # Convert vector to a single integer in base-3, then use XOR with a mask
    # and count bits to create a deterministic but chaotic ordering
    if n == 0:
        return base_score
    
    # Compute the integer value of the vector in base-3
    # v[i] contributes v[i] * 3^(n-1-i)
    # Use a fixed mask derived from n to create position-dependent chaos
    mask = (1 << min(n, 30)) - 1  # Mask for lower bits
    mask = (mask * 2654435761) & 0xFFFFFFFF  # Knuth multiplicative hash
    
    # Compute base-3 integer value efficiently
    # For large n, we only use a hash of the vector
    if n <= 15:
        # Direct computation for small n
        val = 0
        for i in range(n):
            val = val * 3 + v[i]
    else:
        # Use a hash for large n - combine coordinates with weights
        # Use the first 15 coordinates for the hash
        val = 0
        for i in range(min(n, 15)):
            val = (val * 3 + v[i]) & 0xFFFFFFFF
    
    # XOR with mask and count bits (popcount)
    xored = val ^ mask
    # Count set bits in xored
    bit_count = bin(xored).count('1')
    
    # Normalize bit count to [0, 1] range
    max_bits = min(n * 2, 30)  # Approximate max bits
    bit_norm = bit_count / max_bits if max_bits > 0 else 0.5
    
    # Combine: use bit parity as a tiebreaker/chaotic factor
    # Vectors with certain bit patterns get slight priority adjustments
    # This breaks the correlation between simple coordinate balance and line formation
    chaos_bonus = (bit_norm - 0.5) * 0.1
    
    # Also add a secondary hash-based term for additional chaos
    # Use a different mixing
    val2 = (val * 1103515245 + 12345) & 0x7FFFFFFF
    bit_count2 = bin(val2).count('1')
    chaos_bonus2 = (bit_count2 / 31.0 - 0.5) * 0.05
    
    score = base_score + chaos_bonus + chaos_bonus2
    return score
# EVOLVE-BLOCK-END
