def priority(vector, n):
        # Convert vector to a string or use a hash
        # To ensure determinism, we use a fixed seed or just the vector values.
        # A simple hash: sum of (i * v[i])?
        # Or better, use Python's built-in hash on a tuple? 
        # Python's hash is randomized per run unless PYTHONHASHSEED is set.
        # We should use a deterministic hash.
        
        # Let's use a simple polynomial hash.
        h = 0
        base = 31
        for val in vector:
            h = (h * base + val) % (10**9 + 7)
        return float(h)
