"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_DEADLINE = time.time() + 85.0

N = 12
M = 40
TOTAL_MASKS = 1 << N

# Precompute comparability: for each mask, a bitset (as Python int) of all masks that are comparable
# _comp[m] is a bitset where bit k is set if mask m and mask k are comparable (one is strict subset of other)
_comp = [0] * TOTAL_MASKS

for a in range(TOTAL_MASKS):
    bits = 0
    # For each b, check if a and b are comparable
    for b in range(TOTAL_MASKS):
        if a != b and ((a & b) == a or (a & b) == b):
            bits |= (1 << b)
    _comp[a] = bits

def _is_comparable(a, b):
    """Check if a and b are comparable (one is strict subset of the other)."""
    return (_comp[a] >> b) & 1

def _score_family(family):
    """Count unordered pairs (i, j) with i < j such that one is a strict subset of the other."""
    count = 0
    m = len(family)
    for i in range(m):
        ai = family[i]
        for j in range(i + 1, m):
            aj = family[j]
            if _is_comparable(ai, aj):
                count += 1
    return count

def _build_chain_family():
    """Build a family based on chains to maximize subset relations."""
    family = []
    used = set()
    
    # Add the main chain: {}, {0}, {0,1}, {0,1,2}, ..., {0,1,...,11}
    for i in range(N + 1):
        mask = (1 << i) - 1
        family.append(mask)
        used.add(mask)
    
    # Add more sets greedily: prioritize masks that are comparable with many existing family members
    remaining = [m for m in range(TOTAL_MASKS) if m not in used]
    random.shuffle(remaining)
    
    for mask in remaining:
        if len(family) >= M:
            break
        family.append(mask)
        used.add(mask)
    
    return family[:M]

def _delta_score(family, i, new_mask):
    """Compute the change in score when replacing family[i] with new_mask."""
    old_mask = family[i]
    if old_mask == new_mask:
        return 0
    
    m = len(family)
    delta = 0
    for j in range(m):
        if j == i:
            continue
        other = family[j]
        old_rel = 1 if _is_comparable(old_mask, other) else 0
        new_rel = 1 if _is_comparable(new_mask, other) else 0
        delta += new_rel - old_rel
    
    return delta

def _hill_climb_family(family):
    """Hill climbing: repeatedly try to improve by swapping one element."""
    current_family = family[:]
    current_score = _score_family(current_family)
    m = len(current_family)
    all_masks = list(range(TOTAL_MASKS))
    used_set = set(current_family)
    
    improved = True
    while improved and time.time() < _DEADLINE - 10.0:
        improved = False
        for i in range(m):
            if time.time() > _DEADLINE - 10.0:
                break
            old_mask = current_family[i]
            used_set.discard(old_mask)
            
            best_new_mask = None
            best_delta = 0
            
            for new_mask in all_masks:
                if new_mask in used_set:
                    continue
                
                delta = _delta_score(current_family, i, new_mask)
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            
            if best_new_mask is not None:
                used_set.add(best_new_mask)
                current_family[i] = best_new_mask
                current_score += best_delta
                improved = True
            else:
                used_set.add(old_mask)
    
    return current_family, current_score

def _local_search(family, max_iters=200000):
    """Local search via random swaps."""
    current_family = family[:]
    current_score = _score_family(current_family)
    best_family = current_family[:]
    best_score = current_score
    m = len(current_family)
    all_masks = list(range(TOTAL_MASKS))
    used_set = set(current_family)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break
        i = random.randrange(m)
        old_mask = current_family[i]
        
        # Try a few random new masks
        for _ in range(3):
            new_mask = random.choice(all_masks)
            if new_mask == old_mask or new_mask in used_set:
                continue
            
            used_set.discard(old_mask)
            used_set.add(new_mask)
            
            delta = _delta_score(current_family, i, new_mask)
            if delta >= 0:
                current_family[i] = new_mask
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_family = current_family[:]
            else:
                # Revert
                used_set.discard(new_mask)
                used_set.add(old_mask)
                break
    
    return best_family, best_score

def construct():
    """Construct a family of 40 distinct subsets of {0..11} maximizing strict subset pairs."""
    # Start with a chain-based initial family
    family = _build_chain_family()
    
    # Phase 1: Hill climbing with delta computation
    family, score = _hill_climb_family(family)
    
    # Phase 2: Random local search if time permits
    if time.time() < _DEADLINE - 15.0:
        family2, score2 = _local_search(family, max_iters=200000)
        if score2 > score:
            family = family2
            score = score2
    
    # Final polish: one more hill climb pass
    if time.time() < _DEADLINE - 5.0:
        family, score = _hill_climb_family(family)
    
    # Convert to list of lists of ints
    result = []
    for mask in family:
        subset = []
        for b in range(N):
            if (mask >> b) & 1:
                subset.append(b)
        result.append(subset)
    
    # Ensure types are correct
    for i in range(len(result)):
        result[i] = [int(x) for x in result[i]]
    
    return result

# EVOLVE-BLOCK-END
