"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Precompute all 4-element subsets of [1..16]
_ALL_SETS = list(combinations(range(1, 17), 4))
N_SETS = len(_ALL_SETS)  # 1820

# Convert each set to a bitmask for fast intersection checks
# Elements are 1..16, so we use bits 0..15
_MASKS = [0] * N_SETS
for i, s in enumerate(_ALL_SETS):
    m = 0
    for x in s:
        m |= (1 << (x - 1))
    _MASKS[i] = m

# Precompute pairwise intersection sizes
# We need to check: for three distinct sets A, B, C, all three pairwise intersections must NOT be equal.
# A "sunflower" of 3 petals with equal pairwise intersection means |A∩B| = |A∩C| = |B∩C|.
# The task says "No three distinct sets may have all three pairwise intersections equal, including an empty common core."
# This means: for any 3 distinct sets, it is NOT allowed that |A∩B| == |A∩C| == |B∩C|.
# Wait, re-reading: "No three distinct sets may have all three pairwise intersections equal"
# This means |A∩B| = |A∩C| = |B∩C| is forbidden.
# The "including an empty common core" part clarifies that even if the common intersection is empty,
# if all three pairwise intersections have the same size, it's still a sunflower.

# Precompute intersection sizes between all pairs
# inter[i][j] = |S_i ∩ S_j|
# This is a 1820x1820 matrix, which is ~3.3M entries. Fine for memory.
_inter = [[0] * N_SETS for _ in range(N_SETS)]
for i in range(N_SETS):
    mi = _MASKS[i]
    for j in range(i + 1, N_SETS):
        mj = _MASKS[j]
        # Count bits in mi & mj
        common = mi & mj
        # Count set bits
        cnt = 0
        c = common
        while c:
            c &= c - 1
            cnt += 1
        _inter[i][j] = cnt
        _inter[j][i] = cnt

# Precompute for each pair (i,j) whether they can be part of a sunflower with a third set k
# A triple (i,j,k) is a sunflower if _inter[i][j] == _inter[i][k] == _inter[j][k]
# For a greedy/branch-and-bound approach, we need to check compatibility.
# Two sets i and j are "compatible" in the sense that they don't immediately form a sunflower
# with any already-chosen set. But the constraint is on triples.

# Let's define: a family F is valid if for all distinct i,j,k in F, not (inter[i][j]==inter[i][k]==inter[j][k])

# For a faster check: when adding a new set k to existing family F, we need to verify
# that for all pairs (i,j) in F, inter[i][j] != inter[i][k] or inter[i][j] != inter[j][k]
# i.e., NOT (inter[i][j] == inter[i][k] == inter[j][k])

# Precompute for each k, and for each pair (i,j), whether adding k creates a sunflower with i,j.
# This is too large. Instead, let's use a different approach.

# For each set k, precompute a "conflict profile" with respect to pairs.
# Actually, let's just do the check on the fly but with bitmasks for speed.

def _is_sunflower(i, j, k):
    """Check if sets i, j, k form a sunflower (all three pairwise intersections equal)."""
    return _inter[i][j] == _inter[i][k] == _inter[j][k]

def _can_add(family, k):
    """Check if set k can be added to family without creating a sunflower triple."""
    for idx_a in range(len(family)):
        a = family[idx_a]
        for idx_b in range(idx_a + 1, len(family)):
            b = family[idx_b]
            if _inter[a][b] == _inter[a][k] == _inter[b][k]:
                return False
    return True

def _greedy_randomized(deadline):
    """Greedy randomized construction with local improvement."""
    best_family = []
    best_size = 0
    
    while time.time() < deadline:
        # Random permutation of all sets
        order = list(range(N_SETS))
        random.shuffle(order)
        
        family = []
        for idx in order:
            if time.time() >= deadline:
                break
            if _can_add(family, idx):
                family.append(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
    
    return [list(_ALL_SETS[i]) for i in best_family]

def _local_improvement(family, deadline):
    """Try to improve the family by swapping or adding sets."""
    family = list(family)
    current_size = len(family)
    
    while time.time() < deadline:
        improved = False
        # Try adding any set not in family
        in_family = set(family)
        for k in range(N_SETS):
            if time.time() >= deadline:
                break
            if k not in in_family and _can_add(family, k):
                family.append(k)
                in_family.add(k)
                improved = True
                current_size += 1
                break
        
        if not improved:
            # Try swapping: remove one set, add two?
            # For now, just try removing one and seeing if we can add a better one
            for i in range(len(family)):
                if time.time() >= deadline:
                    break
                orig = family[i]
                family[i] = -1  # temporarily remove
                in_family.discard(orig)
                
                # Try to add as many as possible
                added = 0
                for k in range(N_SETS):
                    if time.time() >= deadline:
                        break
                    if k != orig and k not in in_family and _can_add(family, k):
                        family[i] = k
                        in_family.add(k)
                        added = 1
                        break
                
                if added == 0:
                    # Restore original
                    family[i] = orig
                    in_family.add(orig)
                else:
                    # We replaced one with one, size same. Try to add more
                    for k in range(N_SETS):
                        if time.time() >= deadline:
                            break
                        if k not in in_family and _can_add(family, k):
                            family.append(k)
                            in_family.add(k)
                            improved = True
                            current_size += 1
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return [list(_ALL_SETS[i]) for i in family]

def construct():
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    # Phase 1: Multiple randomized greedy restarts
    best = []
    best_size = 0
    
    num_restarts = 0
    while time.time() < deadline - 10:  # Leave 10s for local improvement
        family_idx = _greedy_randomized(deadline - 10)
        num_restarts += 1
        if len(family_idx) > best_size:
            best_size = len(family_idx)
            best = family_idx
    
    # Phase 2: Local improvement on the best found
    if time.time() < deadline - 2:
        improved = _local_improvement(best, deadline - 2)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    # Return sorted lists
    result = [sorted(s) for s in best]
    return result
# EVOLVE-BLOCK-END
