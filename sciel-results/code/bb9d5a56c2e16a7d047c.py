import time
import math
import zlib

_start_time = time.time()
_DEADLINE = _start_time + 280.0

# Cache for entropy of count triples?
# Counts are (c0, c1, c2) where c0+c1+c2=n.
# Entropy H = -sum(p_i * log2(p_i))
# We can compute this quickly.

def _calc_entropy(counts, n):
    """
    Calculate Shannon entropy of the distribution of values in the vector.
    counts: tuple (c0, c1, c2)
    n: total length
    """
    if n == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            h -= p * math.log2(p)
    return h

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    # 1. Calculate counts of 0, 1, 2
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # 2. Calculate Entropy of the vector itself
    ent_v = _calc_entropy((c0, c1, c2), n)
    
    # 3. Calculate "Local Context" Score
    # Look at neighbors (Hamming distance 1).
    # There are 2n neighbors.
    # For each neighbor, calculate entropy.
    # Sum or Average them.
    
    # Optimization: Instead of generating full neighbor vectors,
    # we can update the counts directly.
    # Neighbor of v[i] changing from val to val+1 or val-1.
    
    neighbor_ent_sum = 0.0
    num_neighbors = 0
    
    # To avoid modifying the original vector, we just simulate the count changes.
    # For each index i:
    #   Current val = vector[i]
    #   Neighbor 1: val -> (val+1)%3
    #   Neighbor 2: val -> (val+2)%3 (which is -1)
    
    for i in range(n):
        val = vector[i]
        
        # Neighbor 1: val -> (val+1)%3
        # Decrement count of val, increment count of (val+1)%3
        nc0, nc1, nc2 = c0, c1, c2
        if val == 0:
            nc0 -= 1
            nc1 += 1
        elif val == 1:
            nc1 -= 1
            nc2 += 1
        else:
            nc2 -= 1
            nc0 += 1
        neighbor_ent_sum += _calc_entropy((nc0, nc1, nc2), n)
        num_neighbors += 1
        
        # Neighbor 2: val -> (val-1)%3
        nc0, nc1, nc2 = c0, c1, c2
        if val == 0:
            nc0 -= 1
            nc2 += 1
        elif val == 1:
            nc1 -= 1
            nc0 += 1
        else:
            nc2 -= 1
            nc1 += 1
        neighbor_ent_sum += _calc_entropy((nc0, nc1, nc2), n)
        num_neighbors += 1
        
    avg_neighbor_ent = neighbor_ent_sum / num_neighbors if num_neighbors > 0 else 0.0
    
    # 4. Combine Scores
    # We want high entropy for v.
    # We want high entropy for neighbors? 
    # If neighbors are high entropy, the region is "uniform".
    # If neighbors are low entropy, the region is "structured".
    # Cap sets avoid structure. So we prefer regions that are uniform?
    # Or do we prefer to be DIFFERENT from neighbors to avoid lines?
    # Lines are formed by 3 points. If v and its neighbors are all high entropy, 
    # they are "random-looking".
    
    # Let's try: Score = ent_v + 0.5 * avg_neighbor_ent
    
    # 5. Tie-Breaker: CRC32 Hash
    # Convert vector to bytes for hashing
    # vector elements are 0, 1, 2.
    # We can pack them into bytes.
    # For speed, just use a simple rolling hash with a large prime.
    h = 0
    for v in vector:
        h = (h * 31 + v) & 0xFFFFFFFF
    # Use CRC32 for better distribution?
    # zlib.crc32 requires bytes.
    # Creating bytes for every call might be slow.
    # Let's stick to a faster, stronger polynomial hash.
    # h = (h * 131 + v) & 0xFFFFFFFF
    # Or use a hash of the tuple? tuple(vector) is fast in Python.
    # hash(tuple(vector)) is randomized per process, so not deterministic across runs?
    # Wait, the evaluator runs in one process. But we need deterministic for reproducibility?
    # The task doesn't explicitly require cross-run determinism, but it's good practice.
    # zlib.crc32 is deterministic.
    # Let's try to optimize the hash.
    
    # A simple high-quality hash:
    # h = 0
    # for v in vector:
    #     h ^= (v + 0x9e3779b9 + (h<<6) + (h>>2))
    
    # Let's use the existing simple hash but with a better multiplier.
    h = 0
    for v in vector:
        h = (h * 37 + v) & 0xFFFFFFFF
        
    score = ent_v * 10.0 + avg_neighbor_ent * 5.0 + (h / 4294967296.0)
    
    return float(score)
