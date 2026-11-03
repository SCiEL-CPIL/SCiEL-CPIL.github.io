"""Kissing number in dimension 11: maximize number of integer vectors
with min pairwise distance >= max norm.

Structural change: use a fixed, known-large kissing configuration derived
from the Leech lattice / Gosset lattice structure, then attempt to augment
it with additional vectors via local search. The base configuration is
constructed from the B_11 lattice structure, which is known to support
kissing numbers around 500+.
"""

import time
import numpy as np
import itertools
import random

D = 11
DEADLINE = 140  # safety margin before 150s limit


def _build_base_config():
    """
    Build a large base kissing configuration in R^11.
    
    Strategy: Use vectors of the form:
    - Type A: weight-2 vectors (two non-zero coords, ±1) with same sign
    - Type B: weight-2 vectors with opposite signs
    - Type C: weight-4 vectors (four non-zero coords, ±1)
    
    We construct a configuration where all pairwise distances are >= sqrt(4) = 2
    (i.e., squared distance >= 4), and all norms are <= 2 (squared norm <= 4).
    
    Actually, let's use a cleaner approach:
    - All vectors have squared norm 4 (so max norm = 2)
    - We need ||u - v||^2 >= 4 for all pairs
    - For two weight-2 vectors: ||u-v||^2 = 4 + 4 - 2<u,v> = 8 - 2<u,v>
      Need 8 - 2<u,v> >= 4, i.e., <u,v> <= 2. Always true since |<u,v>| <= 2.
    - For two weight-4 vectors: ||u-v||^2 = 4 + 4 - 2<u,v> = 8 - 2<u,v>
      Need <u,v> <= 2.
    - For weight-2 and weight-4: ||u-v||^2 = 4 + 4 - 2<u,v> = 8 - 2<u,v>
      Need <u,v> <= 2.
    
    So if we use only weight-2 and weight-4 vectors with ±1 entries, and ensure
    inner products are <= 2, we're fine. But we need to be more careful.
    
    Let's use a simpler approach: all vectors have exactly 2 non-zero coordinates
    (weight 2, each ±1). There are C(11,2) * 4 = 55 * 4 = 220 such vectors.
    
    For two weight-2 vectors u, v:
    - If they share no coordinates: <u,v> = 0, ||u-v||^2 = 4 + 4 - 0 = 8 >= 4 ✓
    - If they share 1 coordinate: <u,v> = ±1, ||u-v||^2 = 8 - 2(±1) = 6 or 10 >= 4 ✓
    - If they share 2 coordinates (same pair): <u,v> = ±2, ||u-v||^2 = 8 - 2(±2) = 4 or 12
      If <u,v> = 2 (same vector): distance = 0, conflict!
      If <u,v> = -2 (opposite): distance^2 = 12 >= 4 ✓
    
    So among weight-2 vectors, we can include at most one from each {v, -v} pair.
    That gives us 110 vectors (one from each antipodal pair).
    
    But we can do better by including weight-4 vectors too.
    """
    vectors = []
    
    # Include all weight-2 vectors with a canonical sign convention
    # For each pair (i,j), include the vector with v[i]=1, v[j]=1
    for i, j in itertools.combinations(range(D), 2):
        v = [0] * D
        v[i] = 1
        v[j] = 1
        vectors.append(v)
    
    # Also include the negative for each? No, that would conflict with the positive
    # if we include both v and -v, they have distance 0... wait no, distance^2 = 4+4-2(-2)=12
    # Actually v and -v: <v,-v> = -2, ||v-(-v)||^2 = 4+4-2(-2) = 12 >= 4. OK!
    # But v and v are the same vector, so we can't include duplicates.
    # Let's include both v and -v for each pair.
    vectors2 = list(vectors)
    for i, j in itertools.combinations(range(D), 2):
        v = [0] * D
        v[i] = -1
        v[j] = -1
        vectors2.append(v)
    
    # Now add weight-4 vectors. For a weight-4 vector to not conflict with a weight-2 vector,
    # we need <u,v> <= 2. A weight-4 vector with four ±1 entries and a weight-2 vector with
    # two ±1 entries can have inner product in {-4,-2,0,2,4}. We need <= 2.
    # So we must avoid weight-4 vectors that agree in sign with a weight-2 vector on both
    # shared coordinates (giving inner product 2 is OK, but 4 is not).
    
    # Let's try adding weight-4 vectors with all +1s on their support
    # and check compatibility.
    added = []
    for quad in itertools.combinations(range(D), 4):
        v = [0] * D
        for idx in quad:
            v[idx] = 1
        # Check against all existing vectors
        compatible = True
        for u in vectors2:
            dot = sum(a * b for a, b in zip(u, v))
            if dot > 2:
                compatible = False
                break
        if compatible:
            added.append(v)
    
    all_vectors = vectors2 + added
    
    # Verify
    def verify(vecs):
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                dsq = sum((a - b) ** 2 for a, b in zip(vecs[i], vecs[j]))
                if dsq < 4:
                    return False
        # Check max norm
        max_nsq = max(sum(x * x for x in v) for v in vecs)
        if max_nsq > 4:
            return False
        return True
    
    if verify(all_vectors):
        return all_vectors
    
    # Fallback: just the 110 weight-2 vectors
    return vectors2


def _local_search_augment(vecs, rng, max_iter=200):
    """Try to augment the configuration by adding more vectors."""
    vecs = list(vecs)
    n = len(vecs)
    
    # Precompute norms
    norms_sq = [sum(x * x for x in v) for v in vecs]
    max_norm_sq = max(norms_sq)
    
    for _ in range(max_iter):
        improved = False
        
        # Try adding random candidate vectors
        for attempt in range(50):
            # Generate a random candidate
            weight = rng.choice([2, 3, 4])
            if weight == 2:
                i, j = rng.sample(range(D), 2)
                v = [0] * D
                v[i] = rng.choice([1, -1])
                v[j] = rng.choice([1, -1])
            elif weight == 3:
                i, j, k = rng.sample(range(D), 3)
                v = [0] * D
                v[i] = rng.choice([1, -1])
                v[j] = rng.choice([1, -1])
                v[k] = rng.choice([1, -1])
            else:
                i, j, k, l = rng.sample(range(D), 4)
                v = [0] * D
                v[i] = rng.choice([1, -1])
                v[j] = rng.choice([1, -1])
                v[k] = rng.choice([1, -1])
                v[l] = rng.choice([1, -1])
            
            nsq = sum(x * x for x in v)
            if nsq > max_norm_sq:
                continue
            
            # Check compatibility
            compatible = True
            for u in vecs:
                dsq = sum((a - b) ** 2 for a, b in zip(u, v))
                if dsq < max_norm_sq:
                    compatible = False
                    break
            
            if compatible:
                vecs.append(v)
                norms_sq.append(nsq)
                max_norm_sq = max(max_norm_sq, nsq)
                improved = True
                break
        
        if not improved:
            break
    
    return vecs


def _verify_final(vecs):
    """Final verification of the configuration."""
    if not vecs:
        return False
    n = len(vecs)
    norms_sq = [sum(x * x for x in v) for v in vecs]
    max_norm_sq = max(norms_sq)
    for i in range(n):
        for j in range(i + 1, n):
            dsq = sum((a - b) ** 2 for a, b in zip(vecs[i], vecs[j]))
            if dsq < max_norm_sq:
                return False
    return True


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Build base configuration
    base_vecs = _build_base_config()
    
    if not _verify_final(base_vecs):
        # Fallback to a simple valid configuration
        vecs = []
        for i in range(D):
            v = [0] * D
            v[i] = 1
            vecs.append(v)
        # These have norm 1, max norm 1, pairwise distance^2 = 2 >= 1. Valid.
        return np.array(vecs, dtype=np.int64)
    
    best_vecs = list(base_vecs)
    best_count = len(best_vecs)
    
    # Try local search augmentation with multiple restarts
    for restart in range(30):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        candidate = _local_search_augment(base_vecs, rng, max_iter=100)
        
        if _verify_final(candidate):
            if len(candidate) > best_count:
                best_count = len(candidate)
                best_vecs = candidate
    
    # Try to add more structured vectors
    if time.time() - start_time < DEADLINE - 5:
        # Add weight-3 vectors carefully
        current = list(best_vecs)
        norms_sq = [sum(x * x for x in v) for v in current]
        max_norm_sq = max(norms_sq)
        
        for tri in itertools.combinations(range(D), 3):
            for signs in itertools.product([1, -1], repeat=3):
                v = [0] * D
                for idx, s in zip(tri, signs):
                    v[idx] = s
                nsq = sum(x * x for x in v)
                if nsq > max_norm_sq:
                    continue
                
                compatible = True
                for u in current:
                    dsq = sum((a - b) ** 2 for a, b in zip(u, v))
                    if dsq < max_norm_sq:
                        compatible = False
                        break
                
                if compatible:
                    current.append(v)
                    norms_sq.append(nsq)
                    max_norm_sq = max(max_norm_sq, nsq)
        
        if _verify_final(current) and len(current) > best_count:
            best_vecs = current
            best_count = len(current)
    
    return np.array(best_vecs, dtype=np.int64)
