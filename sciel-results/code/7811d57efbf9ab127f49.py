import time
import math

_T0 = time.time()
_DEADLINE = _T0 + 280.0

def priority(vector, n):
    """
    Priority for a vector in {0,1,2}^n for greedy cap-set construction.
    
    Strategy: Balanced-composition routing.
    In F_3^n, the "center" of the cube (vectors with many 1s) is the densest
    region in terms of potential 3-term APs. However, for greedy construction,
    we want to pick points that are "safe" to add. 
    
    The key insight from the inspiration is that routing through the high-density
    center first allows the greedy algorithm to keep more early vectors. 
    Why? Because early vectors define the "structure". If we pick random or
    corner-heavy vectors first, we might accidentally create a dense subspace
    that saturates quickly. By picking "balanced" vectors (many 1s) first, we
    are effectively sampling the most "constrained" points first. If they fit,
    they fit well. If they don't, the algorithm skips them. This tends to
    result in a more uniform distribution of the final cap set, avoiding
    large empty subspaces that would have allowed more points.
    
    Specifically, we prioritize vectors with a count of 1s close to n/2.
    Vectors with all 0s or all 2s are "corners" and form large subspaces
    (e.g., the {0,2}^n subspace is a cap set of size 2^n). If we pick these
    first, we fill up that subspace, but we might miss opportunities in the
    middle. Conversely, if we pick the middle first, we might block some
    corners, but the overall density is higher.
    
    Actually, let's re-evaluate. The {0,2}^n subspace is a cap set. Its size is 2^n.
    The total size of the cube is 3^n.
    For n=8, 2^8=256, 3^8=6561.
    Known cap set sizes:
    n=8: 252 (or close to it, R(8) = 252? No, R(8) is 252? 
    Let's check known values.
    R(1)=3, R(2)=4, R(3)=9, R(4)=20, R(5)=45, R(6)=112, R(7)=280, R(8)=?
    Actually, R(8) is 252? No, that's the size of {0,2}^8.
    Wait, the {0,2}^n subspace is a cap set. So R(n) >= 2^n.
    For n=8, 2^8 = 256.
    Is R(8) > 256?
    According to literature, R(8) is 252? No, that can't be right if 2^8=256 is a cap set.
    Ah, the {0,2}^n subspace contains no 3-APs. So R(n) >= 2^n.
    For n=8, R(8) >= 256.
    Actually, R(8) is known to be 252? Let me check.
    Wait, 2^8 = 256. If R(8) was 252, that would contradict the existence of the {0,2}^8 cap set.
    Maybe I am misremembering the values.
    Let's assume the seed program (random) got 0.9922.
    The inspiration says "Balanced-composition priority ... Measured: 1 -> 1.238".
    So the score jumped from ~1.0 to 1.238.
    This implies the cap set size increased by ~24%.
    
    So the strategy is: Prioritize vectors with a number of 1s close to n/2.
    Why does this work?
    If we pick corners first, we fill the {0,2}^n subspace. This is a valid cap set of size 2^n.
    But maybe we can do better by mixing in 1s?
    If we pick vectors with many 1s, we are picking points that are "far" from the corners in Hamming distance (over {0,1,2}).
    
    Let's implement the priority as:
    1. Count the number of 1s in the vector.
    2. Calculate the distance from n/2.
    3. Priority is higher for smaller distance (i.e., closer to n/2).
    4. Tie-breaker: Pseudo-random hash to ensure determinism and break symmetry.
    
    This will route the greedy search through the "middle" of the cube.
    """
    if time.time() > _DEADLINE:
        return 0.0

    # Count 1s
    count_1 = 0
    for v in vector:
        if v == 1:
            count_1 += 1
    
    # Target is n/2
    # We want vectors with count_1 close to n/2 to have HIGH priority.
    # So priority should be inversely related to |count_1 - n/2|.
    # Let's use a Gaussian-like decay or just simple inverse distance.
    # dist = abs(count_1 - n/2.0)
    # base_priority = 1.0 / (1.0 + dist)
    
    # However, we also need to break ties.
    # We can combine the structural priority with a hash.
    # But if the structural priority dominates, the order is deterministic based on composition.
    # This might be too rigid.
    # Let's mix:
    # Primary key: count_1 close to n/2.
    # Secondary key: hash.
    
    # To return a single float, we can scale the hash to be small relative to the structural score.
    # Or, we can just return the structural score + a small epsilon based on hash.
    
    # Let's try:
    # score = (n - abs(2*count_1 - n))  # Max score is n (when count_1 = n/2), min is 0.
    # Then add a small random component.
    
    # But wait, if we just sort by this, all vectors with the same count_1 will be grouped.
    # Within that group, we want a good order.
    
    # Let's compute the index for the hash.
    idx = 0
    for v in reversed(vector):
        idx = idx * 3 + v

    def mix64(h):
        h ^= h >> 33
        h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 33
        h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 33
        return h

    seed_val = idx ^ (n * 0x9E3779B97F4A7C15)
    mixed = mix64(seed_val)
    
    # Structural score
    # We want count_1 close to n/2.
    # Let's map count_1 to a score in [0, 1].
    # Max score when count_1 is closest to n/2.
    # dist = abs(count_1 - n/2.0)
    # max_dist = n/2.0
    # norm_score = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0
    
    # Let's just use the count itself.
    # For n=8, n/2 = 4.
    # count_1 = 4 -> best.
    # count_1 = 0 or 8 -> worst.
    
    # Let's return:
    # base = (n - abs(2*count_1 - n))
    # This ranges from 0 to n.
    # We want higher priority for higher base.
    
    # To add the hash, we can do:
    # result = base + (mixed / (2**64)) * 0.1
    # This ensures that vectors with higher base are always preferred,
    # and ties are broken by hash.
    
    base = n - abs(2 * count_1 - n)
    
    # Add a small perturbation to break ties deterministically
    # The perturbation should be small enough not to override the base score difference of 1.
    # So scale hash to [0, 0.5)
    perturb = (mixed / (2**64)) * 0.5
    
    return float(base + perturb)
