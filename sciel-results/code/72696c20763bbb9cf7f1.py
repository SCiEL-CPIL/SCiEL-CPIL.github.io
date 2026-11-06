# EVOLVE-BLOCK-START
import struct

def priority(vector, n):
    """
    Assign a deterministic, high-quality pseudo-random priority using a fast
    64-bit mixing function (SplitMix64 finalizer) on the base-3 integer
    representation of the vector. This is significantly faster than SHA-256,
    allowing the greedy algorithm to process more vectors within the time limit
    and potentially explore a more uniform ordering of the hypercube.
    """
    # Convert vector to a single integer in base 3
    val = 0
    for v in vector:
        val = val * 3 + v
        
    # Apply SplitMix64 finalizer for good avalanche properties
    # This is a fast, deterministic way to mix the bits of 'val'
    x = val
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb
    x = x ^ (x >> 31)
    
    # Ensure we are working with a 64-bit unsigned integer
    x &= (1 << 64) - 1
    
    # Map to [0, 1)
    return x / (1 << 64)
# EVOLVE-BLOCK-END
