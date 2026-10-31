"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import hashlib
import struct
from itertools import product

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Use a deterministic pseudo-random hash based on the vector's coordinates.
    # This avoids structural biases (like coordinate sum mod 3) and distributes
    # priority values uniformly, allowing the greedy algorithm to explore the space
    # more evenly.
    
    # Pack the vector into a byte string for hashing.
    # Each coordinate is 0, 1, or 2. We can pack them efficiently.
    # For simplicity and speed, we can just convert to a string or use a simple hash.
    # However, we need it to be fast.
    
    # A simple, fast hash:
    # Sum of (coordinate * prime^index) mod large_prime
    # But we need a float in [0, 1).
    
    # Let's use a simple polynomial hash.
    h = 0
    prime = 31
    mod = (1 << 32) - 1  # 2^32 - 1
    
    for i, val in enumerate(vector):
        h = (h * prime + val) % mod
    
    # Normalize to [0, 1)
    return h / mod

# EVOLVE-BLOCK-END
