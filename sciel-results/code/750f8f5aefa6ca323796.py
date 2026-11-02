"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

TIME_LIMIT = 115.0  # safety margin before 120s kill


def _generate_candidates():
    """
    Generate a diverse pool of candidate integer vectors in R^11.
    We focus on low-weight vectors (weight 2, 3, 4) with small coefficients
    to maximize the chance of finding large kissing configurations.
    
    To break symmetry and reduce search space, we enforce:
    - The first non-zero coordinate is positive (canonical sign).
    - We include both positive and negative pairs/triples for the rest.
    
    We also include some weight-1 vectors (unit vectors) as they are always compatible
    with many others, but note: in a kissing configuration, you can't have both
    e_i and -e_i, and e_i is incompatible with any vector having a non-zero i-th
    component that makes the distance too small. Actually, let's think:
    
    Condition: min pairwise distance >= max norm.
    Let M = max ||v||^2. For any pair u, v: ||u - v||^2 >= M.
    ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u·v >= M.
    
    If we normalize to norm 2, this becomes kissing number problem.
    The standard kissing number in 11D is 432.
    
    The seed program uses weight-2 vectors (entries ±1 in two positions, 0 elsewhere).
    ||v||^2 = 2. So M = 2.
    For two such vectors u, v: ||u - v||^2 >= 2.
    ||u - v||^2 = 2 + 2 - 2 u·v = 4 - 2 u·v >= 2  =>  u·v <= 1.
    
    For weight-2 vectors, u·v can be -2, -1, 0, 1, 2.
    So we need u·v <= 1. This excludes u·v = 2 (same vector) and u·v = -2 (negative of same).
    Actually u·v = 2 means u = v. u·v = -2 means u = -v.
    u·v = 1 means they share one coordinate with same sign.
    u·v = 0 means no shared coordinates or opposite signs in shared.
    u·v = -1 means share one coordinate with opposite sign.
    u·v = -2 means u = -v.
    
    So the constraint is: no two vectors can be identical or negatives.
    That means we can take at most one from each {v, -v} pair.
    But wait, u·v = 1 is allowed. So vectors sharing a coordinate are fine.
    
    Total weight-2 vectors: C(11,2) * 4 = 55 * 4 = 220.
    Pairs {v, -v}: 110.
    So max kissing number using only weight-2 is 110? But the seed gets 110?
    Wait, the seed generates all 220. But v and -v have u·v = -2, which violates u·v <= 1? 
    No: -2 <= 1 is true. Wait, I made an error.
    
    Let me re-check: ||u - v||^2 >= M where M = max ||w||^2.
    If all vectors have norm^2 = 2, then M = 2.
    ||u - v||^2 = 4 - 2 u·v >= 2  =>  -2 u·v >= -2  =>  u·v <= 1.
    
    So u·v = -2 is allowed! That means v and -v are compatible.
    So we can have all 220 weight-2 vectors? Let's check:
    u = e_1 + e_2, v = -e_1 - e_2. u·v = -1 - 1 = -2. <= 1. OK.
    u = e_1 + e_2, v = e_1 - e_2. u·v = 1 - 1 = 0. <= 1. OK.
    u = e_1 + e_2, v = e_1 + e_3. u·v = 1. <= 1. OK.
    
    So ALL weight-2 vectors are mutually compatible! That gives k = 220.
    But the seed returns 220 vectors. And score is 1.0, meaning seed = 220.
    
    Can we do better with mixed weights?
    If we add a weight-3 vector w with ||w||^2 = 3, then M = 3.
    Now for any pair u, v: ||u - v||^2 >= 3.
    For two weight-2 vectors: 4 - 2 u·v >= 3 => u·v <= 0.5 => u·v <= 0 (integer).
    So weight-2 vectors must have u·v <= 0.
    For w (weight 3) and u (weight 2): ||w - u||^2 = 3 + 2 - 2 w·u = 5 - 2 w·u >= 3 => w·u <= 1.
    For two weight-3 vectors w1, w2: 3 + 3 - 2 w1·w2 >= 3 => w1·w2 <= 1.5 => w1·w2 <= 1.
    
    This is much more restrictive. Mixing weights is tricky.
    
    Let's stick with weight-2 vectors but see if we can add weight-4 vectors.
    Weight-4: ||v||^2 = 4. M = 4.
    Two weight-2: 4 - 2 u·v >= 4 => u·v <= 0.
    Weight-2 and weight-4: 2 + 4 - 2 u·v >= 4 => u·v <= 1.
    Two weight-4: 4 + 4 - 2 w1·w2 >= 4 => w1·w2 <= 2.
    
    So if we use weight-2 and weight-4, the constraint between two weight-2 is u·v <= 0.
    This means weight-2 vectors must be orthogonal (or opposite sign in one shared coord).
    u·v = 0 means no shared coords or opposite signs.
    u·v = -1 means one shared coord with opposite sign.
    
    This is a different problem. Let's focus on maximizing k.
    
    The known kissing number in 11D is 432.
    The Leech lattice construction gives 432.
    
    Let's try a greedy approach with a large candidate pool.
    """
    candidates = []
    
    # Weight 2 vectors: all ±1 in two positions
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                candidates.append(tuple(v))
    
    # Weight 3 vectors: all ±1 in three positions
    for i, j, k in combinations(range(D), 3):
        for si, sj, sk in product((1, -1), repeat=3):
            v = [0] * D
            v[i] = si
            v[j] = sj
            v[k] = sk
            candidates.append(tuple(v))
    
    # Weight 4 vectors: all ±1 in four positions
    for i, j, k, l in combinations(range(D), 4):
        for si, sj, sk, sl in product((1, -1), repeat=4):
            v = [0] * D
            v[i] = si
            v[j] = sj
            v[k] = sk
            v[l] = sl
            candidates.append(tuple(v))
    
    return candidates


def _check_compatible(v1, v2, max_norm_sq):
    """Check if v1 and v2 are compatible given max_norm_sq."""
    dot = sum(a * b for a, b in zip(v1, v2))
    n1_sq = sum(a * a for a in v1)
    n2_sq = sum(b * b for b in v2)
    dist_sq = n1_sq + n2_sq - 2 * dot
    return dist_sq >= max_norm_sq


def _max_norm_sq(vectors):
    """Compute maximum norm squared among vectors."""
    return max(sum(x * x for x in v) for v in vectors)


def construct():
    start_time = time.time()
    
    # Start with the known good set of weight-2 vectors
    base_vectors = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                base_vectors.append(tuple(v))
    
    best_set = list(base_vectors)
    best_size = len(best_set)
    
    # Try to add weight-3 and weight-4 vectors
    candidates = _generate_candidates()
    
    # Precompute norms
    norms_sq = {v: sum(x * x for x in v) for v in candidates}
    
    # Greedy addition
    for v in candidates:
        if time.time() - start_time > TIME_LIMIT:
            break
        
        if v in best_set:
            continue
        
        # Check if v can be added
        can_add = True
        for u in best_set:
            if not _check_compatible(v, u, max(norms_sq[v], _max_norm_sq([u]))):
                # Need to check against the max norm of the whole set
                pass
        
        # More efficient: check against current max norm
        current_max = _max_norm_sq(best_set)
        new_max = max(current_max, norms_sq[v])
        
        can_add = True
        for u in best_set:
            if not _check_compatible(v, u, new_max):
                can_add = False
                break
        
        # Also check all pairs in best_set still compatible with new_max
        if can_add:
            for i in range(len(best_set)):
                for j in range(i + 1, len(best_set)):
                    if not _check_compatible(best_set[i], best_set[j], new_max):
                        can_add = False
                        break
                if not can_add:
                    break
        
        if can_add:
            best_set.append(v)
            best_size += 1
    
    # Convert to numpy array
    result = np.array(best_set, dtype=np.int64)
    return result
# EVOLVE-BLOCK-END
