"""Cap set task: greedy construction with SHA256-based random priority."""
import hashlib
import time
import itertools

# EVOLVE-BLOCK-START

def _sha256_priority(vector, n):
    """
    Generate a priority for a vector using SHA256 hash.
    The vector is encoded as a string of digits (each element is 0, 1, or 2).
    We hash the string and use the first 8 bytes as a 64-bit integer for ordering.
    """
    # Encode vector as a string of characters '0', '1', '2'
    s = ''.join(str(v) for v in vector)
    h = hashlib.sha256(s.encode('ascii')).digest()
    # Use first 8 bytes as a 64-bit integer
    priority_int = int.from_bytes(h[:8], byteorder='big')
    return priority_int

def _is_cap_set(vectors):
    """
    Check if a set of vectors in {0,1,2}^n forms a cap set.
    A cap set has no three distinct elements a, b, c such that a+b+c = (0,0,...,0) mod 3.
    Equivalently, no three distinct elements form an arithmetic progression:
    a + c = 2*b mod 3, i.e., for each coordinate, a_i + c_i ≡ 2*b_i (mod 3).
    
    We check all triples. For efficiency, we check: for each pair (a, c) with a != c,
    compute b = (a + c) / 2 mod 3, i.e., b_i = (a_i + c_i) * 2^{-1} mod 3 = (a_i + c_i) * 2 mod 3.
    Then check if b is in the set and b != a and b != c.
    
    Actually, the condition for a 3-term AP in {0,1,2}^n is:
    There exist distinct a, b, c in the set such that a + c = 2b (mod 3).
    
    We'll use a set for O(1) lookup.
    """
    vec_set = set(vectors)
    n = len(vectors[0]) if vectors else 0
    
    for i, a in enumerate(vectors):
        for j, c in enumerate(vectors):
            if i == j:
                continue
            # Compute b = (a + c) * 2 mod 3, coordinate-wise
            # Since 2 * 2 = 4 ≡ 1 mod 3, multiplying by 2 is the inverse of 2.
            # So b_i = (a_i + c_i) * 2 mod 3
            b = tuple((a[k] + c[k]) * 2 % 3 for k in range(n))
            if b in vec_set and b != a and b != c:
                return False
    return True

def _try_add(vector, current_set, n):
    """
    Try to add vector to current_set. Return True if the result is still a cap set.
    """
    # Check if adding vector creates a 3-term AP
    # For each pair (u, v) in current_set, check if (u + v) * 2 mod 3 == vector
    # Or if (u + vector) * 2 mod 3 == v, etc.
    # More simply: for each u in current_set, compute w = (u + vector) * 2 mod 3.
    # If w is in current_set and w != u and w != vector, then we have a 3-term AP.
    # Also need to check: for each pair (u, v) in current_set, if (u + v) * 2 mod 3 == vector.
    
    vec_set = set(current_set)
    
    # Case 1: vector is the middle term: u + w = 2*vector => w = (2*vector - u) mod 3 = (2*vector + 2*u) mod 3? No.
    # Actually: u + w ≡ 2*vector (mod 3) => w ≡ 2*vector - u (mod 3)
    # Since we're in mod 3, -u ≡ 2u (mod 3), so w ≡ 2*vector + 2*u (mod 3)
    # But let's just compute directly: w_i = (2*vector[i] - u[i]) % 3
    
    for u in current_set:
        # Check if vector is the middle term: u + w = 2*vector
        w = tuple((2 * vector[k] - u[k]) % 3 for k in range(n))
        if w in vec_set and w != u and w != vector:
            return False
        # Check if vector is an endpoint: u + vector = 2*w => w = (u + vector) / 2 = (u + vector) * 2 mod 3
        w2 = tuple((u[k] + vector[k]) * 2 % 3 for k in range(n))
        if w2 in vec_set and w2 != u and w2 != vector:
            return False
    
    return True

def priority(vector, n):
    """
    Main entry point. The evaluator calls this to get a priority for each vector.
    Vectors are added greedily in descending priority order.
    
    We use SHA256 to generate a pseudo-random priority for each vector.
    The key insight: we need to return a priority such that when vectors are sorted
    by this priority in descending order, the greedy cap set construction yields a large cap set.
    
    However, the task says "vectors of {0,1,2}^n are added greedily in descending priority
    when they keep the set a cap set". This means the evaluator:
    1. Generates all 3^n vectors
    2. Calls priority(vector, n) for each
    3. Sorts vectors by priority in descending order
    4. Greedily adds vectors if they maintain the cap set property
    5. Returns the size of the resulting cap set
    
    So our priority function should assign higher priority to vectors that are more "useful"
    for forming a large cap set.
    
    Strategy: Use a hash-based random ordering. SHA256 gives us a good pseudo-random permutation.
    """
    # Use SHA256 to generate a deterministic pseudo-random priority
    s = ''.join(str(v) for v in vector)
    h = hashlib.sha256(s.encode('ascii')).digest()
    # Use all 32 bytes for maximum randomness
    priority_int = int.from_bytes(h, byteorder='big')
    # Normalize to [0, 1)
    return priority_int / (1 << 256)

# EVOLVE-BLOCK-END
