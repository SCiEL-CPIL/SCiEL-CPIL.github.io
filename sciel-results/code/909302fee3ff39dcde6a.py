def priority(vector, n):
        # Compute base-3 index
        idx = 0
        for v in vector:
            idx = idx * 3 + v
        # Use a simple hash function to scramble the index
        # This ensures that neighbors in index space are not neighbors in priority space
        # We want a deterministic pseudo-random float
        h = idx
        # Simple bit mixing
        h ^= (h >> 16)
        h *= 0x85ebca6b
        h ^= (h >> 13)
        h *= 0xc2b2ae35
        h ^= (h >> 16)
        # Normalize to [0, 1)
        return (h & 0x7FFFFFFF) / 0x7FFFFFFF
