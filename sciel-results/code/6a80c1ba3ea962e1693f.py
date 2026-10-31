"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import zlib

# EVOLVE-BLOCK-START

# Cache for cap set membership checks per n
_cap_cache = {}

def _is_cap_set(vectors, n):
    """Check if a set of vectors in {0,1,2}^n is a cap set (no three collinear).
    Three points a,b,c are collinear in F_3^n iff a + b + c = (0,...,0) mod 3
    with a,b,c distinct. Equivalently, for any two distinct a,b, the third point
    c = (2*0 - a - b) mod 3 = (-a-b) mod 3 must not be in the set.
    """
    vec_set = set(vectors)
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            a = vectors[i]
            b = vectors[j]
            # c_k = (0 - a_k - b_k) % 3 = (-a_k - b_k) % 3
            c = tuple((-a[k] - b[k]) % 3 for k in range(n))
            if c in vec_set:
                return False
    return True

def _add_vector_preserves_cap(existing, new_vec, n):
    """Check if adding new_vec to existing set keeps it a cap set."""
    # For each existing vector a, check if the 'third point' with a and new_vec is in existing
    for a in existing:
        c = tuple((-a[k] - new_vec[k]) % 3 for k in range(n))
        if c in existing:
            return False
    return True

def _generate_all_vectors(n):
    """Generate all vectors in {0,1,2}^n."""
    return list(itertools.product(range(3), repeat=n))

def _balanced_score(vector):
    """Score based on balance of symbols 0,1,2. More balanced = higher priority."""
    counts = [0, 0, 0]
    for v in vector:
        counts[v] += 1
    n = len(vector)
    if n == 0:
        return 0.0
    # Variance-like measure: lower variance = more balanced = higher score
    mean = n / 3.0
    variance = sum((c - mean) ** 2 for c in counts)
    return -variance  # More negative variance (closer to 0) means more balanced

def _hash_priority(vector, n):
    """Fast deterministic hash-based priority using CRC32."""
    try:
        b = bytes(vector)
        h = zlib.crc32(b) & 0xFFFFFFFF
    except Exception:
        # Fallback if bytes conversion fails (shouldn't happen for 0,1,2)
        h = 0
        for v in vector:
            h = (h * 31 + v) & 0xFFFFFFFF
    return h / 0xFFFFFFFF

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Vectors are added in descending priority order if they keep the set a cap set.
    
    Strategy:
    1. Prefer vectors with UNbalanced symbol distribution (high variance)
       Hypothesis: Highly balanced vectors (like 0,1,2 permutations) form lines
       among themselves, so picking them first wastes capacity.
    2. Break ties with a CRC32-based deterministic ordering
    """
    # Primary: unbalanced distribution (higher variance = higher priority)
    counts = [0, 0, 0]
    for v in vector:
        counts[v] += 1
    n_vec = len(vector)
    if n_vec == 0:
        return 0.0
    mean = n_vec / 3.0
    variance = sum((c - mean) ** 2 for c in counts)
    
    # Secondary: hash for deterministic tie-breaking
    h = _hash_priority(vector, n)
    
    # Combine: variance dominates, hash breaks ties
    return variance * 1000.0 + h

# EVOLVE-BLOCK-END
