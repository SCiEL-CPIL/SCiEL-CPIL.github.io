import numpy as np
import zlib

def priority(vector, n):
    # Convert to numpy array for fast operations
    v = np.array(vector, dtype=np.int8)
    
    # Count occurrences of 0, 1, 2
    n_0 = np.count_nonzero(v == 0)
    n_1 = np.count_nonzero(v == 1)
    n_2 = np.count_nonzero(v == 2)
    
    # Heuristic: Prioritize vectors with a balanced distribution of 0s, 1s, and 2s.
    # This avoids vectors that are heavily skewed towards 0s or 2s, which might
    # force more constraints on the rest of the set due to the arithmetic progression
    # structure (e.g., if b has many 0s, a and c must also have 0s in those positions).
    target = n / 3.0
    
    # Calculate deviation from the ideal balanced state
    dev = (n_0 - target)**2 + (n_1 - target)**2 + (n_2 - target)**2
    
    # Base priority is negative deviation (higher is better for more balanced vectors)
    base_priority = -dev
    
    # Tie-breaking: Use CRC32 for a fast, deterministic, non-lexicographic order
    v_bytes = bytes(v)
    h = zlib.crc32(v_bytes)
    # Normalize hash to [0, 1)
    tie_break = (h % 10000) / 10000.0
    
    return base_priority + tie_break * 0.01
