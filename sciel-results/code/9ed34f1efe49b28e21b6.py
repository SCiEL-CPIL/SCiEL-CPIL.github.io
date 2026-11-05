"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Precompute all 4-subsets of [1, 16] as bitmasks
_ALL_SUBSETS = list(combinations(range(1, 17), 4))
_TOTAL_SETS = len(_ALL_SUBSETS)
_MASKS = [0] * _TOTAL_SETS
for _i, _s in enumerate(_ALL_SUBSETS):
    _m = 0
    for _bit in _s:
        _m |= (1 << (_bit - 1))
    _MASKS[_i] = _m

# Precompute pairwise intersections
_INTER = [[0] * _TOTAL_SETS for _ in range(_TOTAL_SETS)]
for _i in range(_TOTAL_SETS):
    _mi = _MASKS[_i]
    for _j in range(_i + 1, _TOTAL_SETS):
        _v = _mi & _MASKS[_j]
        _INTER[_i][_j] = _v
        _INTER[_j][_i] = _v


def _check_conflict(candidate_idx, family):
    """Check if adding candidate_idx to family creates a sunflower.
    Uses bucket-based approach: group family members by intersection with candidate,
    then only check pairs within buckets of size >= 2."""
    m_cand = _MASKS[candidate_idx]
    
    # Group family members by their intersection with the candidate
    buckets = {}
    for fam_idx in family:
        inter_val = m_cand & _MASKS[fam_idx]
        if inter_val not in buckets:
            buckets[inter_val] = []
        buckets[inter_val].append(fam_idx)
    
    # For each bucket with size >= 2, check if any pair has intersection equal to bucket key
    for inter_val, members in buckets.items():
        if len(members) < 2:
            continue
        # Check all pairs in this bucket
        n_members = len(members)
        for a in range(n_members):
            for b in range(a + 1, n_members):
                if _INTER[members[a]][members[b]] == inter_val:
                    return True  # Conflict found
    return False


def _greedy_pass(order, deadline):
    """Run a single greedy pass with the given order of set indices."""
    family = []
    
    for idx in order:
        if time.time() > deadline - 2.0:
            break
        if not _check_conflict(idx, family):
            family.append(idx)
    
    return family


def _local_search_polish(family, deadline):
    """Try to add sets not in the family to improve the solution."""
    in_family = set(family)
    improved = True
    
    while improved and time.time() < deadline - 3.0:
        improved = False
        for idx in range(_TOTAL_SETS):
            if time.time() > deadline - 3.0:
                break
            if idx in in_family:
                continue
            
            if not _check_conflict(idx, family):
                family.append(idx)
                in_family.add(idx)
                improved = True
    
    return family


def construct():
    deadline = time.time() + 80.0  # Safety margin before 90s kill
    
    best_family = []
    best_size = 0
    
    random.seed(42)
    
    # Phase 1: Multiple random greedy restarts
    num_restarts = 500
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        order = list(range(_TOTAL_SETS))
        random.shuffle(order)
        
        family = _greedy_pass(order, deadline)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    # Phase 2: Local search polish on the best family
    if best_size > 0 and time.time() < deadline - 5.0:
        best_family = _local_search_polish(best_family, deadline)
        best_size = len(best_family)
    
    # Phase 3: More greedy restarts with remaining time, then polish
    while time.time() < deadline - 10.0:
        order = list(range(_TOTAL_SETS))
        random.shuffle(order)
        
        family = _greedy_pass(order, deadline)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            # Polish the new best
            if time.time() < deadline - 5.0:
                best_family = _local_search_polish(best_family, deadline)
                best_size = len(best_family)
    
    # Final polish if time allows
    if time.time() < deadline - 2.0:
        best_family = _local_search_polish(best_family, deadline)
        best_size = len(best_family)
    
    return [sorted(_ALL_SUBSETS[idx]) for idx in best_family]
# EVOLVE-BLOCK-END
