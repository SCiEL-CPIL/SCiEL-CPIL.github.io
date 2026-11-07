"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

N = 12
M = 40
TOTAL_MASKS = 1 << N
FULL_MASK = TOTAL_MASKS - 1

# Precompute comparability: for each mask, store a bitset of which masks it's comparable to
# Using Python ints as bitsets for fast AND operations
_comp_bitsets = [0] * TOTAL_MASKS

def _build_comp_bitsets():
    """Precompute comparability bitsets for all masks."""
    for a in range(TOTAL_MASKS):
        bits = 0
        for b in range(TOTAL_MASKS):
            if a != b and ((a & b) == a or (a & b) == b):
                bits |= (1 << b)
        _comp_bitsets[a] = bits

_build_comp_bitsets()

def _score_family(family):
    """Count unordered pairs (i, j) with i < j such that one is a strict subset of the other."""
    count = 0
    m = len(family)
    for i in range(m):
        ai = family[i]
        for j in range(i + 1, m):
            aj = family[j]
            if _comp_bitsets[ai] >> aj & 1:
                count += 1
    return count

def _build_chain_family():
    """Build a strong initial family based on a chain structure."""
    chain = []
    for size in range(N + 1):
        mask = (1 << size) - 1
        chain.append(mask)
    
    family = chain[:]
    used = set(chain)
    
    # Greedily add masks that maximize comparability with existing family
    while len(family) < M:
        best_mask = -1
        best_gain = -1
        for mask in range(TOTAL_MASKS):
            if mask in used:
                continue
            gain = 0
            for f in family:
                if _comp_bitsets[mask] >> f & 1:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
                if gain == len(family):
                    break
        if best_mask >= 0:
            family.append(best_mask)
            used.add(best_mask)
        else:
            for mask in range(TOTAL_MASKS):
                if mask not in used:
                    family.append(mask)
                    used.add(mask)
                    break
            else:
                break
    
    return family

def _greedy_construct():
    """Pure greedy construction: add masks one by one maximizing total comparable pairs."""
    family = []
    used = set()
    
    # Start with a few seed masks
    seeds = [0, FULL_MASK, (1 << 6) - 1, ~((1 << 6) - 1) & FULL_MASK]
    for s in seeds:
        if s not in used:
            family.append(s)
            used.add(s)
    
    while len(family) < M:
        best_mask = -1
        best_gain = -1
        for mask in range(TOTAL_MASKS):
            if mask in used:
                continue
            gain = 0
            for f in family:
                if _comp_bitsets[mask] >> f & 1:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
                if gain == len(family):
                    break
        if best_mask >= 0:
            family.append(best_mask)
            used.add(best_mask)
        else:
            for mask in range(TOTAL_MASKS):
                if mask not in used:
                    family.append(mask)
                    used.add(mask)
                    break
            else:
                break
    
    return family

def _fast_local_search(family, max_time=None):
    """Fast local search using bitset-based delta computation."""
    if max_time is None:
        max_time = _DEADLINE - 5.0
    
    current_family = family[:]
    current_score = _score_family(current_family)
    best_family = current_family[:]
    best_score = current_score
    m = len(current_family)
    
    # Precompute for each family member: which other members are comparable
    # comp_with[i] = bitset of indices j != i where family[i] and family[j] are comparable
    def compute_comp_with(fam):
        cw = []
        for i in range(len(fam)):
            bits = 0
            for j in range(len(fam)):
                if i != j and (_comp_bitsets[fam[i]] >> fam[j] & 1):
                    bits |= (1 << j)
            cw.append(bits)
        return cw
    
    comp_with = compute_comp_with(current_family)
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < max_time:
        i = random.randrange(m)
        new_mask = random.randrange(TOTAL_MASKS)
        if new_mask == current_family[i]:
            continue
        
        # Check distinctness
        is_dup = False
        for k in range(m):
            if k != i and current_family[k] == new_mask:
                is_dup = True
                break
        if is_dup:
            continue
        
        old_mask = current_family[i]
        
        # Compute delta: new comparable pairs - old comparable pairs
        # Old: comp_with[i] has bits for all j where old_mask is comparable to family[j]
        # New: for each j != i, check if new_mask is comparable to family[j]
        old_comparable_count = bin(comp_with[i]).count('1')
        
        new_comparable_count = 0
        for j in range(m):
            if j == i:
                continue
            if _comp_bitsets[new_mask] >> current_family[j] & 1:
                new_comparable_count += 1
        
        delta = new_comparable_count - old_comparable_count
        
        if delta > 0:
            current_family[i] = new_mask
            current_score += delta
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_family = current_family[:]
                comp_with = compute_comp_with(current_family)
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return best_family, best_score

def _systematic_improve(family, max_time=None):
    """Systematic improvement: for each position, find the best replacement."""
    if max_time is None:
        max_time = _DEADLINE - 3.0
    
    current_family = family[:]
    current_score = _score_family(current_family)
    m = len(current_family)
    
    improved = True
    while improved and time.time() < max_time:
        improved = False
        for i in range(m):
            if time.time() > max_time:
                break
            old_mask = current_family[i]
            
            # Compute current comparable count for position i
            old_comparable_count = 0
            for j in range(m):
                if j != i and (_comp_bitsets[old_mask] >> current_family[j] & 1):
                    old_comparable_count += 1
            
            best_new = old_mask
            best_gain = 0
            
            for new_mask in range(TOTAL_MASKS):
                if new_mask == old_mask:
                    continue
                # Check distinctness
                is_dup = False
                for k in range(m):
                    if k != i and current_family[k] == new_mask:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                new_comparable_count = 0
                for j in range(m):
                    if j != i and (_comp_bitsets[new_mask] >> current_family[j] & 1):
                        new_comparable_count += 1
                
                gain = new_comparable_count - old_comparable_count
                if gain > best_gain:
                    best_gain = gain
                    best_new = new_mask
            
            if best_new != old_mask and best_gain > 0:
                current_family[i] = best_new
                current_score += best_gain
                improved = True
    
    return current_family, current_score

def construct():
    """Construct a family of 40 distinct subsets of {0..11} maximizing strict subset pairs."""
    # Phase 1: Try multiple initial constructions
    candidates = []
    
    # Chain-based
    fam1 = _build_chain_family()
    candidates.append(fam1)
    
    # Greedy
    fam2 = _greedy_construct()
    candidates.append(fam2)
    
    # Pick best initial
    best_family = max(candidates, key=lambda f: _score_family(f))
    best_score = _score_family(best_family)
    
    # Phase 2: Fast local search
    t2_end = min(_DEADLINE - 25.0, _START_TIME + 20.0)
    if time.time() < t2_end:
        f2, s2 = _fast_local_search(best_family, max_time=t2_end)
        if s2 > best_score:
            best_score = s2
            best_family = f2[:]
    
    # Phase 3: Systematic improvement
    t3_end = min(_DEADLINE - 10.0, _START_TIME + 40.0)
    if time.time() < t3_end:
        f3, s3 = _systematic_improve(best_family, max_time=t3_end)
        if s3 > best_score:
            best_score = s3
            best_family = f3[:]
    
    # Phase 4: More local search
    t4_end = min(_DEADLINE - 5.0, _START_TIME + 55.0)
    if time.time() < t4_end:
        f4, s4 = _fast_local_search(best_family, max_time=t4_end)
        if s4 > best_score:
            best_score = s4
            best_family = f4[:]
    
    # Phase 5: Final systematic polish
    t5_end = _DEADLINE - 2.0
    if time.time() < t5_end:
        f5, s5 = _systematic_improve(best_family, max_time=t5_end)
        if s5 > best_score:
            best_score = s5
            best_family = f5[:]
    
    # Convert to list of lists of ints
    result = []
    for mask in best_family:
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
