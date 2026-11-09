import hashlib
import time

# EVOLVE-BLOCK-START
_START_TIME = time.monotonic()
_DEADLINE = _START_TIME + 320.0

def priority(vector, n):
    if time.monotonic() > _DEADLINE:
        return 0.0

    # Convert vector to bytes for hashing
    # Vector is a tuple of ints 0, 1, 2
    b = bytes(vector)
    
    # Use SHA-256 for a strong, uniform pseudo-random ordering
    # This effectively randomizes the greedy selection process.
    # Random greedy algorithms are known to perform well for cap sets,
    # and this avoids potential biases in structural heuristics.
    h = hashlib.sha256(b).digest()
    
    # Convert the first 8 bytes to a 64-bit integer, then to a float in [0, 1)
    # This provides a high-resolution random priority.
    val = int.from_bytes(h[:8], 'big')
    score = val / (2**64)
    
    # Add a small structural bias: prefer vectors with "intermediate" weight
    # Weight = sum of coordinates.
    # In F_3^n, vectors with weight n/2 or 3n/2 might be "central".
    # However, pure randomness is often robust.
    # Let's add a tiny bit of weight bias to see if it helps.
    # weight = sum(vector)
    # bias = -abs(weight - n) / n  # Penalize extreme weights
    # score += bias * 1e-6
    
    return score
# EVOLVE-BLOCK-END
