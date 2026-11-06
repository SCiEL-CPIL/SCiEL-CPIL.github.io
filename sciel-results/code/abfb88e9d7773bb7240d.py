"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import hashlib
from itertools import product

_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def _elapsed():
    """Return elapsed time since start."""
    return time.time() - _START_TIME

def _is_cap_set(vecs):
    """Check if a list of vectors (tuples of 0,1,2) forms a cap set."""
    s = set(vecs)
    n = len(vecs[0]) if vecs else 0
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            a = vecs[i]
            b = vecs[j]
            pass
    pass

def _check_cap_set_with_new(vecs, new_vec):
    """Check if adding new_vec to vecs (a set of tuples) keeps it a cap set."""
    n = len(new_vec)
    s = set(vecs)
    for u in s:
        w = tuple((-new_vec[k] - u[k]) % 3 for k in range(n))
        if w in s and w != u and w != new_vec:
            return False
    return True

def _hash_priority(vector, n):
    """Generate a deterministic pseudo-random priority for a vector."""
    val = 0
    for i, v in enumerate(vector):
        val = val * 3 + v
    return (val * 2654435761) % (3**n)

def _spectral_collision_risk(vector, n):
    """
    Estimate collision risk using coordinate-wise overlap patterns.
    
    For a vector v, the 'collision risk' is high if there exist pairs (u, w)
    in the selected set such that u + w + v ≡ 0 mod 3 coordinate-wise.
    
    We approximate this by counting how many coordinate patterns in v
    are 'compatible' with common line structures. Vectors with many
    coordinates equal to 0, 1, or 2 in a balanced way tend to have
    higher collision risk.
    
    We use a smooth surrogate: sum of squared deviations from uniform
    distribution across coordinates, weighted by a hash-based position.
    """
    # Count occurrences of each value
    counts = [0, 0, 0]
    for v in vector:
        counts[v] += 1
    
    # Uniform distribution would be n/3 for each
    uniform = n / 3.0
    
    # Squared deviation from uniform (higher = more skewed = potentially higher risk)
    deviation = sum((c - uniform) ** 2 for c in counts)
    
    # Add a position-based hash to break ties and add variety
    base3_val = 0
    for i, v in enumerate(vector):
        base3_val = base3_val * 3 + v
    
    # Use a smooth function to combine: risk increases with deviation
    # but we want to MINIMIZE risk, so priority should be inversely related
    # We use a soft penalty: 1 / (1 + deviation * scale)
    scale = 1.0 / max(n, 1)
    penalty = 1.0 / (1.0 + deviation * scale * 10.0)
    
    # Combine with hash for deterministic ordering
    hash_component = (base3_val * 2654435761) % (3**n)
    hash_norm = hash_component / float(3**n)
    
    # Blend: 70% spectral risk, 30% hash for diversity
    return 0.7 * penalty + 0.3 * hash_norm

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Uses spectral collision risk estimation with hash-based tie-breaking.
    
    Higher priority means the vector is added earlier. We want to add
    vectors with LOW collision risk first, so we invert the risk.
    """
    # Deadline guard: if we're running out of time, return a simple hash
    if _elapsed() > _DEADLINE:
        val = 0
        for v in vector:
            val = val * 3 + v
        return float((val * 2654435761) % (3**n))
    
    # Compute spectral collision risk (higher = more risky)
    risk = _spectral_collision_risk(vector, n)
    
    # Invert so that low risk = high priority
    # Map risk from [0, 1] to priority [0, 1] with inversion
    return 1.0 - risk

# EVOLVE-BLOCK-END
