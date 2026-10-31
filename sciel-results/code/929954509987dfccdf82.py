"""Cap set greedy construction with structured priority."""
import time
import itertools

# EVOLVE-BLOCK-START

# Deadline guard: stop searching after this many seconds
_DEADLINE_SECONDS = 280.0

def _is_cap_set_check(vec, existing_set):
    """
    Check if adding vec to existing_set keeps it a cap set.
    A cap set has no three distinct elements a, b, c with a+b+c = 0 mod 3.
    Equivalently, no three distinct elements form an affine line:
    for any two distinct points x, y in the set, the third point z = 2x - y mod 3
    (or equivalently the point completing the line x, y, z where x+y+z ≡ 0 mod 3)
    must not be in the set.
    
    More precisely: a set S is a cap set iff for all distinct a, b in S,
    the third point c = (-a - b) mod 3 is not in S.
    """
    for a in existing_set:
        # For each a in existing_set, check if b = (-a - vec) mod 3 is in existing_set
        # If so, then a, b, vec form a line (a + b + vec ≡ 0 mod 3)
        b_candidate = tuple((-a[i] - vec[i]) % 3 for i in range(len(vec)))
        if b_candidate in existing_set:
            return False
    return True

def priority(vector, n):
    """
    This function is called for each vector to determine its priority.
    Higher priority = selected earlier in greedy insertion.
    
    Strategy: Use a multi-criteria priority that balances:
    1. Distance from origin (vectors with more 1s and 2s, fewer 0s tend to be 
       more "spread out" and less likely to form lines with other spread vectors)
    2. A structured hash-based ordering to break ties and avoid symmetry traps
    3. Weighted sum using a fixed pseudo-random vector to create non-local ordering
    
    The key insight: vectors that are "far" from each other in the ternary grid
    are less likely to form arithmetic progressions. We prioritize vectors that
    maximize pairwise distance from already-selected vectors, but since we can't
    know the final set, we use a proxy: vectors with balanced coordinate distributions
    and high "entropy" in their pattern.
    """
    # Convert vector to a tuple of ints
    vec = tuple(int(x) for x in vector)
    
    # Criterion 1: Number of non-zero coordinates (sparsity penalty)
    # Vectors with more non-zeros are more "spread out"
    nonzero_count = sum(1 for x in vec if x != 0)
    
    # Criterion 2: Sum of coordinates mod 3 (avoid clustering around sum=0)
    coord_sum = sum(vec) % 3
    
    # Criterion 3: Weighted projection onto a fixed pattern
    # Use a deterministic pseudo-random weight based on position
    # This creates a non-local ordering that breaks lattice symmetries
    weight = 0.0
    for i, x in enumerate(vec):
        # Fixed weights based on position - creates chaotic ordering
        w = ((i * 7 + 3) % 5) - 2  # weights in {-2, -1, 0, 1, 2}
        weight += w * x
    
    # Criterion 4: Product of non-zero coordinates (favors balanced vectors)
    prod = 1
    for x in vec:
        if x != 0:
            prod *= x
    
    # Combined priority:
    # Primary: higher nonzero_count is better (more spread)
    # Secondary: prefer coord_sum != 0 (less likely to be in a line with two others)
    # Tertiary: weighted projection for tie-breaking
    # Quaternary: product as final tie-breaker
    
    # Scale to create a total ordering
    priority_val = 0.0
    priority_val += nonzero_count * 10.0  # dominant term
    priority_val += (2.0 if coord_sum != 0 else 0.0) * 5.0  # avoid sum=0
    priority_val += weight * 1.0  # non-local ordering
    priority_val += (prod % 10) * 0.1  # fine tie-breaking
    
    return priority_val

# EVOLVE-BLOCK-END
