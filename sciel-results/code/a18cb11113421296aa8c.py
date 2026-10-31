"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

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
    """Fast deterministic hash-based priority."""
    # Use a simple polynomial hash
    h = 0
    base = 31
    for v in vector:
        h = (h * base + v) & 0xFFFFFFFF
    return h / 0xFFFFFFFF

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Vectors are added in descending priority order if they keep the set a cap set.
    
    Strategy:
    1. Prefer vectors with balanced symbol distribution (0,1,2 roughly equal)
    2. Break ties with a hash-based deterministic ordering
    """
    # Primary: balanced distribution (higher = more balanced)
    bal = _balanced_score(vector)
    # Secondary: hash for deterministic tie-breaking
    h = _hash_priority(vector, n)
    # Combine: balance dominates, hash breaks ties
    return bal * 1000.0 + h

# EVOLVE-BLOCK-END
