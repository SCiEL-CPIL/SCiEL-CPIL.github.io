"""
Cap set task: Maximize the size of a cap set in {0,1,2}^n.
The greedy algorithm selects vectors in descending order of priority.
We use a high-quality, bit-mixed hash to randomize the selection order,
avoiding the structural biases of lexicographic ordering.
"""

def priority(vector, n):
    """
    Computes a priority for a vector to be used in greedy cap set construction.
    Uses a deterministic, high-quality hash function to ensure a uniform
    random-looking ordering of vectors, which typically yields larger
    independent sets (cap sets) than structured orders like lexicographic.
    """
    # Initialize hash
    h = 0x811C9DC5  # FNV offset basis
    
    # Process each coordinate
    for v in vector:
        # FNV-1a step
        h ^= v
        h = (h * 0x01000193) & 0xFFFFFFFFFFFF  # 64-bit FNV prime
        
    # Apply a finalizer to scramble the bits (MurmurHash3 finalizer style)
    # This ensures that even small differences in the vector lead to
    # significantly different priorities, breaking any linear structure.
    h ^= (h >> 33)
    h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    h ^= (h >> 33)
    h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    h ^= (h >> 33)
    
    # Convert to float in [0, 1)
    return h / 0xFFFFFFFFFFFFFFFF
