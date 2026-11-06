"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import product

def _is_cap_set(vecs):
    """Check if a list of vectors (tuples of 0,1,2) forms a cap set."""
    s = set(vecs)
    n = len(vecs[0]) if vecs else 0
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            a = vecs[i]
            b = vecs[j]
            # Check if a, b, c form a line: c_k = (a_k + 2*b_k) % 3 or (2*a_k + b_k) % 3
            # Actually, three points a,b,c form a line if they are distinct and c = (a + 2b) % 3
            # or more generally, for each coordinate, the values are all same or all different.
            # A line in {0,1,2}^n is a set {a, b, c} where for each coordinate k,
            # a_k, b_k, c_k are all equal or all distinct.
            # Equivalently, c = (a + 2b) % 3 is not quite right.
            # The condition is: a_k + b_k + c_k ≡ 0 (mod 3) for all k, and a,b,c distinct.
            # But for greedy insertion, we just need to check if adding new vec v to existing set S
            # creates a line. A line involving v and two existing points u, w means u, w, v are collinear.
            # u, w, v collinear iff for all k, u_k + w_k + v_k ≡ 0 mod 3.
            # So for new v, we need to check if there exist u, w in S (u != w) such that
            # for all k, u_k + w_k + v_k ≡ 0 mod 3.
            # This is equivalent to w = (-v - u) mod 3 for all coordinates.
            # So for each u in S, compute w = tuple((-v[k] - u[k]) % 3 for k in range(n))
            # and check if w is in S and w != u.
            # But we also need u != w, and if u == w then 2*u + v ≡ 0 mod 3, which means
            # v = -2u mod 3. This would be a "line" of length 1? No, a line has 3 distinct points.
            # So we need u, w, v all distinct.
            pass
    # The above is O(n^2 * n) per check which is too slow for large sets.
    # Let's implement a proper check.
    pass

def _check_cap_set_with_new(vecs, new_vec):
    """Check if adding new_vec to vecs (a set of tuples) keeps it a cap set."""
    n = len(new_vec)
    s = set(vecs)
    for u in s:
        # Compute w such that u + w + new_vec ≡ 0 mod 3
        w = tuple((-new_vec[k] - u[k]) % 3 for k in range(n))
        if w in s and w != u and w != new_vec:
            return False
    return True

def _hash_priority(vector, n):
    """Generate a deterministic pseudo-random priority for a vector."""
    # Convert vector to a base-3 number
    val = 0
    for i, v in enumerate(vector):
        val = val * 3 + v
    # Use a simple hash function to create a priority
    # We want a permutation-like ordering
    return (val * 2654435761) % (3**n)

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    Uses a deterministic hash-based priority to avoid local clustering.
    """
    # The evaluator calls this to determine the order in which vectors are added.
    # We return a float; higher priority means added earlier.
    # We use a hash-based priority that is deterministic and spread out.
    return float(_hash_priority(vector, n))

# EVOLVE-BLOCK-END
