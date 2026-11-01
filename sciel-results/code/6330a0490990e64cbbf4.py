"""Construct nonzero integer vectors in R^11 for kissing-number search."""
import time
import numpy as np
from itertools import combinations

D = 11

# Known kissing number in dimension 11 is 504.
# The Leech lattice projection to R^11 gives 504 vectors of norm 4 with
# minimum distance 4 (so after normalizing to norm 2, the configuration is
# a kissing configuration of 504 unit spheres).

def construct():
    deadline = time.time() + 140  # 140s safety margin out of 150s

    # We build vectors from the Leech lattice structure projected to 11D.
    # The 504 minimal vectors of the Leech lattice (norm 4) can be represented
    # via coordinates over {0, ±1, ±2} with specific patterns.
    #
    # A simpler approach: use the known structure of the 504 kissing vectors
    # in dimension 11. They come from the D_11 lattice's 2*11*10 = 220 vectors
    # of type (±2, 0,...,0, 0,...,0) and (0,...,±1, 0,...,±1, 0,...) plus
    # additional vectors from the E_8 + D_3 decomposition or from the
    # Leech lattice construction.
    #
    # Actually, let's use a direct construction. The kissing number in d=11
    # is 504. The vectors can be obtained from the following construction:
    # Start with the 480 minimal vectors of the E8 lattice (in 8D), combine
    # with D3 lattice vectors.
    #
    # Let me use a more direct approach: generate candidate vectors and
    # verify the constraint.
    #
    # Key insight: The 504 vectors in dimension 11 can be constructed as:
    # Type 1: Vectors with two ±1 entries and rest 0 (from D_11): 2*11*10 = 220
    # Type 2: Vectors with four entries from {±1, ±1/2} patterns...
    #
    # Actually, the standard construction uses the following:
    # The 504 = 220 + 284 vectors. But let me use a verified construction.
    #
    # From Conway and Sloane: The 504 minimal vectors of the Leech lattice
    # projected to 11D. But the Leech lattice is 24D.
    #
    # Let me use the direct D_11 + E_8 hybrid approach:
    # In R^11, take the D_11 lattice vectors of type (±1, 0, ..., ±1, 0, ...) 
    # which gives 220 vectors, then add 284 more from a different construction.
    #
    # Actually, the simplest verified approach: use the 504 vectors from the
    # following explicit construction based on the D_12 lattice or from
    # coordinate patterns.
    #
    # Let me just generate all candidate vectors of small integer coordinates
    # and find the maximum valid set.

    # For efficiency, let's use known structure. The 504 kissing vectors in
    # dimension 11 have the following coordinate patterns (all with norm^2 = 4):
    # - 220 vectors of form (±1, 0, ..., ±1, 0, ...) with exactly two non-zero ±1 entries
    # - 284 vectors from more complex patterns
    
    # Let me try a greedy construction with the 220 D_11 vectors first,
    # then try to add more.

    # Generate the 220 D_11 type vectors: two entries are ±1, rest are 0
    vectors = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
    
    # Now try to add more vectors. The remaining 284 come from vectors with
    # four non-zero entries. In the Leech lattice projection, these have
    # coordinates like (±1, ±1, ±1, ±1, 0, ...) but that gives norm^2=4 with
    # four ±1's. Let's check: 4 vectors with ±1 each = norm^2 = 4.
    # But we need min distance >= max norm. If max norm is 2 (norm^2=4),
    # then min distance must be >= 2, i.e., min distance^2 >= 4.
    
    # For two vectors u, v with entries in {0, ±1}, the squared distance is
    # sum((u_i - v_i)^2). Each difference is in {-2, -1, 0, 1, 2}, so each
    # term is 0, 1, or 4.
    
    # Two vectors with two ±1 entries: if they share one position, distance^2
    # can be as small as... let's think. u has 1s at positions i,j; v has 1s at
    # positions i,k. Then at position i: both have same sign (say 1), diff=0.
    # At position j: u has 1, v has 0, diff^2=1. At position k: u has 0, v has 1,
    # diff^2=1. Total = 2. But we need distance^2 >= 4!
    
    # So the 220 D_11 vectors with two ±1 entries do NOT all satisfy the
    # minimum distance constraint with each other! We need to select a subset.
    
    # Let me reconsider. The norm of each vector is sqrt(2) for two-unit vectors.
    # So max norm = sqrt(2), and we need min distance >= sqrt(2), i.e., min dist^2 >= 2.
    # Wait, re-reading: "minimum pairwise distance is at least the maximum vector norm"
    # If all vectors have norm sqrt(2), then max norm = sqrt(2), and we need
    # min pairwise distance >= sqrt(2).
    
    # For two vectors u, v each with two ±1 entries:
    # If they're at the same two positions with same signs: distance = 0 (identical)
    # If they share one position: dist^2 = 1+1 = 2 >= 2. OK!
    # If they're at different positions: dist^2 = 1+1+1+1 = 4 >= 2. OK!
    # If opposite signs at shared position: e.g., u=(1,1,0...), v=(-1,0,1,0...)
    #   At pos 0: 1-(-1)=2, diff^2=4. At pos 1: 1-0=1, diff^2=1. At pos 2: 0-1=-1, diff^2=1.
    #   Total = 6 >= 2. OK!
    
    # So actually all 220 vectors with two ±1 entries satisfy min dist^2 >= 2 
    # among themselves (as long as no two are identical). Since we generate each
    # unique (i,j,si,sj) once, we have 220 distinct vectors, all with norm^2=2,
    # and min pairwise distance^2 >= 2. 
    
    # Now can we add more? Vectors with norm^2 = 2 could also be of type
    # (±1, ±1, 0, ...) which is what we have. Or type with one entry ±sqrt(2)
    # but that's not integer. So integer vectors with norm^2=2 are exactly
    # the two-±1 type. We have all 220.
    
    # What about vectors with larger norm? If we add a vector with norm^2 > 2,
    # then max norm increases, and we need all pairwise distances >= new max norm.
    
    # The optimal kissing configuration in d=11 has 504 vectors all of equal norm.
    # They all have norm^2 = 4 (norm = 2), and min distance = 2 (dist^2 = 4).
    
    # So let's work with norm^2 = 4 vectors. Integer vectors with norm^2=4:
    # Type A: one entry ±2 (22 vectors)
    # Type B: two entries ±1, ±1 (220 vectors) -- wait, that's norm^2 = 1+1 = 2, not 4!
    # Type C: four entries ±1 (norm^2 = 4): C(11,4) * 2^4 = 330 * 16 = 5280 candidates
    # Type D: one entry ±2 and... no, ±2 alone is norm^2=4.
    # Type E: two entries, one ±2 and one... 4 + x^2 = 4 means x=0. So just ±2.
    
    # So norm^2=4 integer vectors are:
    # - 22 of type (±2, 0, ..., 0)
    # - 5280 of type (±1, ±1, ±1, ±1, 0, ...) with exactly four ±1 entries
    
    # Total candidates: 5302. We need to find the largest subset with
    # min pairwise distance^2 >= 4.
    
    # This is a large combinatorial problem. Let me use a greedy approach
    # with the 22 Type A vectors first, then add Type C vectors greedily.
    
    # Actually, the known 504-vector construction:
    # 22 Type A vectors: (±2, 0, ..., 0)
    # 482 Type C vectors selected from the 5280.
    
    # But 22 + 482 = 504. Let me verify this makes sense.
    
    # Greedy: start with all 22 Type A vectors, then add Type C vectors
    # that are at distance >= 2 from all existing vectors.
    
    vectors = []
    # Add Type A: (±2, 0, ..., 0)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    
    # Now greedily add Type C vectors (four ±1 entries)
    # For speed, precompute which Type C vectors are compatible with Type A
    # A Type C vector v has four ±1 entries. Distance to Type A vector (2s, 0,...,0)
    # at position i: if v[i] != 0, dist^2 includes (v[i] - 2s)^2.
    # If v[i] = s (same sign as 2s's sign * 1... wait, 2s is ±2, v[i] is ±1)
    # If s=1: (v[i] - 2)^2. If v[i]=1: (1-2)^2=1. If v[i]=-1: (-1-2)^2=9.
    # If s=-1: (v[i]+2)^2. If v[i]=-1: (-1+2)^2=1. If v[i]=1: (1+2)^2=9.
    # If v[i]=0: (0-2s)^2 = 4.
    # 
    # For Type C vector v and Type A vector e_i * 2s:
    # dist^2 = sum over all positions. At position i: as computed above.
    # At other positions j where v[j]!=0: (v[j]-0)^2 = 1 each.
    # At positions where both are 0: 0.
    #
    # So dist^2 = (contribution at i) + (number of other non-zero positions of v)
    # = (contribution at i) + (4 - (1 if v[i]!=0 else 0))
    #
    # If v[i] != 0: dist^2 = (v[i]-2s)^2 + 3
    #   If v[i] = s: (s - 2s)^2 + 3 = s^2 + 3 = 4. OK (>=4)
    #   If v[i] = -s: (-s - 2s)^2 + 3 = 9s^2 + 3 = 12. OK
    # If v[i] == 0: dist^2 = 4 + 4 = 8. OK
    #
    # So ALL Type C vectors are at distance >= 2 from all Type A vectors! Great.
    
    # Now we need Type C vectors that are mutually at distance >= 2 (dist^2 >= 4).
    # Two Type C vectors u, v each with four ±1 entries:
    # dist^2 = sum((u_i - v_i)^2)
    # Each term: 0 if same, 1 if differ by 1 in |value|, 4 if opposite signs.
    # Actually: u_i, v_i in {-1, 0, 1}. (u_i-v_i)^2 in {0, 1, 4}.
    # 
    # If u and v have the same support (same 4 positions) and same signs: dist^2 = 0
    # If same support, one sign flip: dist^2 = 4 at that position. Total >= 4. OK!
    # If same support, two sign flips: dist^2 = 8. OK.
    # If different supports: depends on overlap.
    # If they share k positions: at shared positions, if same sign: 0, if diff: 4.
    #   At non-shared positions: 1 each (since one has ±1, other has 0).
    #   dist^2 = 4*(# shared with diff sign) + (4-k) + (4-k) = 4*d + 8 - 2k
    #   where d = # shared positions with different signs.
    #   Minimum when d=0: 8-2k. For k=4: 0 (identical). For k=3: 2 < 4. NOT OK!
    
    # So two Type C vectors sharing 3 positions with same signs have dist^2 = 2 < 4.
    # We need to avoid this.
    
    # Greedy selection from Type C:
    # Generate all 5280 Type C vectors, then greedily add those compatible.
    
    # For speed with 5280 candidates, use vectorized operations.
    
    # Generate all Type C vectors
    # Choose 4 positions from 11: C(11,4) = 330
    # For each, 2^4 = 16 sign patterns
    # Total: 5280
    
    if time.time() > deadline - 30:
        return vectors  # Return just the 22 Type A vectors if running low on time
    
    type_c = []
    for combo in combinations(range(D), 4):
        for signs in range(16):
            v = [0] * D
            for k, pos in enumerate(combo):
                v[pos] = 1 if (signs >> k) & 1 else -1
            type_c.append(v)
    
    type_c = np.array(type_c, dtype=np.int64)
    
    # Greedily add Type C vectors
    # Current vectors: 22 Type A
    # For each Type C vector, check distance to all current vectors
    
    # Since all Type C are compatible with Type A, we just need mutual compatibility
    # among selected Type C vectors.
    
    # For efficiency: two Type C vectors are compatible if dist^2 >= 4.
    # Precompute incompatibility: dist^2 < 4 means dist^2 in {0, 1, 2, 3}.
    # Since entries are integers, dist^2 is integer. dist^2 < 4 means dist^2 in {0,1,2,3}.
    
    # Let's use a different approach: represent each Type C vector by its support
    # (4-element subset) and signs. Two vectors are incompatible if they share
    # >= 3 positions with the same signs.
    
    # Actually, let me just do the greedy with numpy for speed.
    
    selected = []
    n = len(type_c)
    
    # For speed, process in chunks
    # dist^2 between type_c[i] and type_c[j] = ||type_c[i]||^2 + ||type_c[j]||^2 - 2*type_c[i]·type_c[j]
    # = 4 + 4 - 2*(type_c[i]·type_c[j]) = 8 - 2*(dot)
    # dist^2 >= 4 iff 8 - 2*dot >= 4 iff dot <= 2
    
    # So two Type C vectors are compatible iff their dot product <= 2.
    # Dot product of two Type C vectors: sum of products of corresponding entries.
    # Each entry product is in {-1, 0, 1}. Dot = (# same sign at shared) - (# opp sign at shared)
    # Max dot = 4 (identical), min dot = -4.
    
    # Compatible iff dot <= 2, i.e., NOT (dot >= 3), i.e., not (dot = 3 or 4).
    # dot = 4: identical vectors
    # dot = 3: share 4 positions, 3 same signs, 1 different? No: dot = 4-2k where k is
    #   # of sign differences at shared positions. dot=3 is not possible since
    #   dot has same parity as... actually dot = sum of 4 terms each in {-1,0,1}
    #   with exactly 4 non-zero terms total across both. Wait no, each vector has
    #   exactly 4 non-zero entries.
    
    # Let me just compute: for vectors with supports S_u, S_v:
    # dot = sum over i in S_u ∩ S_v of (u_i * v_i)
    # Each u_i * v_i is ±1.
    # So dot = (|S_u ∩ S_v| - 2 * (# of positions in intersection where signs differ))
    # 
    # dot = 4: |S_u ∩ S_v| = 4, all same signs (identical)
    # dot = 3: impossible (would need |intersection| = 3 with... 3-2k=3 means k=0,
    #   but then |intersection| must be 3, giving dot=3. Wait: if |S_u ∩ S_v|=3 and
    #   all same signs: dot = 3. Yes! That's possible.)
    # dot = 2: |S_u ∩ S_v| = 2 all same, or |S_u ∩ S_v| = 4 with 1 diff, or
    #   |S_u ∩ S_v| = 3 with... 3-2k=2 impossible (odd). So: |intersection|=2 same signs,
    #   or |intersection|=4 with exactly 1 sign difference.
    
    # Incompatible (dot > 2) means dot in {3, 4}:
    # dot=4: identical
    # dot=3: share exactly 3 positions, all same signs
    
    # So the constraint is: no two selected vectors can be identical, and no two
    # can share exactly 3 positions with the same signs.
    
    # Greedy: iterate through Type C vectors, add if compatible with all selected.
    
    # For speed with up to 5280 candidates and potentially thousands selected,
    # use a fast check.
    
    selected_indices = []
    
    # Convert to a set for fast lookup of identical vectors
    selected_set = set()
    
    # For the "share 3 with same signs" check, we can precompute for each vector
    # its "3-subsets with signs" and check overlaps.
    
    # Actually, let's just do the dot product check with numpy for batches.
    
    # Process all 5280 at once in batches for speed
    batch_size = 5280
    all_type_c = type_c  # shape (5280, 11)
    
    # We'll do it iteratively but vectorized where possible
    # For each candidate, compute dot product with all selected, check max <= 2
    
    # Start with empty selected
    current = np.empty((0, D), dtype=np.int64)
    
    # To speed up, note that the constraint is local (depends on 4 coordinates),
    # so we can use a smarter data structure. But let's try the direct approach
    # with numpy and see if it's fast enough.
    
    # 5280 candidates, each checked against growing selected set.
    # Worst case: 5280 * 5000 dot products of length 11 = ~290M operations.
    # With numpy vectorization per candidate, this should be a few seconds.
    
    for idx in range(n):
        if time.time() > deadline - 10:
            break
        candidate = all_type_c[idx]
        # Check against all currently selected
        if len(current) > 0:
            dots = current @ candidate  # shape (len(current),)
            if dots.max() > 2:
                continue  # incompatible
        # Also check if identical to any (dot=4 is caught by >2)
        current = np.vstack([current, candidate.reshape(1, D)]) if len(current) == 0 else np.concatenate([current, candidate.reshape(1, D)])
    
    # Combine Type A and selected Type C
    final_vectors = np.vstack([vectors, current]) if len(current) > 0 else vectors
    
    return final_vectors
