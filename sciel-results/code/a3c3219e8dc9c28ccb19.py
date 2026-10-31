import random
import time

# Cache for random vectors per dimension n
_random_vectors_cache = {}

def priority(vector, n):
    """
    Assigns a priority to a vector in {0,1,2}^n based on its projection
    onto a fixed random vector with entries in {-1, 1}.
    """
    # Get or generate the random vector for this dimension n
    if n not in _random_vectors_cache:
        # Use a deterministic seed based on n to ensure reproducibility
        # but different orderings for different n
        rng = random.Random(n)
        # Generate a vector of random signs {-1, 1}
        r = tuple(rng.choice([-1, 1]) for _ in range(n))
        _random_vectors_cache[n] = r
    else:
        r = _random_vectors_cache[n]
    
    # Compute the dot product (projection)
    # vector is a tuple of 0, 1, 2
    # r is a tuple of -1, 1
    # The result is an integer
    score = 0
    for i in range(n):
        score += vector[i] * r[i]
        
    return float(score)
