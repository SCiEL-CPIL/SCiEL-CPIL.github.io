import hashlib
import time

def priority(vector, n):
    # Compute a stable, high-quality hash of the vector
    # Convert vector to a string representation for hashing
    # Using bytes for efficiency
    vec_bytes = bytes(vector)
    h = hashlib.md5(vec_bytes).digest()
    # Convert hash to a float in [0, 1)
    hash_val = int.from_bytes(h[:8], 'big') / (2**64)
    
    # Heuristic: Slightly favor vectors with more zeros (lower weight)
    # This is a hypothesis that "sparser" vectors are easier to place
    # without creating immediate conflicts for future vectors.
    weight = sum(vector)
    # Normalize weight to [0, 1]
    norm_weight = weight / n if n > 0 else 0.0
    
    # Combine hash and heuristic
    # The heuristic term is small to preserve the randomization benefit
    # while introducing a structural bias.
    bias = 0.01 * (1.0 - norm_weight)
    
    return hash_val + bias
