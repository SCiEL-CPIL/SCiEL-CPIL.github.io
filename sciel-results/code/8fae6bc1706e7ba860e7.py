import time
import itertools

# EVOLVE-BLOCK-START
def _paleyn_orientation(p):
    """
    Construct Paley tournament on p vertices.
    Returns list of bits for pairs (i, j) i < j.
    0 means i -> j, 1 means j -> i.
    """
    squares = set(x * x % p for x in range(1, p))
    orientation = []
    append = orientation.append
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            append(0 if diff in squares else 1)
    return orientation

def _verify_paley_exact(p, orientation):
    """
    Exact verification of the 4-dominator property.
    Uses Python integer bitsets for speed.
    """
    in_bits = [0] * p
    idx = 0
    for i in range(p):
        for j in range(i + 1, p):
            if orientation[idx] == 0:
                in_bits[j] |= (1 << i)
            else:
                in_bits[i] |= (1 << j)
            idx += 1

    # Precompute pairwise intersections to reduce work per 4-tuple
    # pair_int[a][b] = in_bits[a] & in_bits[b] for a < b
    # Then for 4-tuple (a,b,c,d): check if pair_int[a][b] & pair_int[c][d] != 0
    # But we need consistent ordering. Let's just use direct 4-way AND.
    
    # Use local variables for speed
    ib = in_bits
    for a, b, c, d in itertools.combinations(range(p), 4):
        if (ib[a] & ib[b] & ib[c] & ib[d]) == 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    candidates = [31, 43, 47, 59, 67, 71, 79, 83]

    best_n = 83
    best_orientation = _paleyn_orientation(83)

    for p in candidates:
        if time.time() > deadline:
            break
        orientation = _paleyn_orientation(p)
        if _verify_paley_exact(p, orientation):
            best_n = p
            best_orientation = orientation
            break

    return {'n': best_n, 'orientation': best_orientation}
# EVOLVE-BLOCK-END
