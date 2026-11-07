"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = time.time() + 82.0
N = 12
M = 40
TOTAL_MASKS = 1 << N

# Precompute comparability matrix: comp[a][b] = 1 if a and b are comparable (one strict subset of other)
_comp_matrix = np.zeros((TOTAL_MASKS, TOTAL_MASKS), dtype=np.int8)
for a in range(TOTAL_MASKS):
    for b in range(a + 1, TOTAL_MASKS):
        if a == b:
            continue
        if (a & b) == a or (a & b) == b:
            _comp_matrix[a][b] = 1
            _comp_matrix[b][a] = 1

def _score_family(family):
    """Count unordered comparable pairs using precomputed matrix."""
    m = len(family)
    count = 0
    for i in range(m):
        for j in range(i + 1, m):
            count += _comp_matrix[family[i]][family[j]]
    return count

def _build_chain_family():
    """Build a family seeded with a maximal chain, then fill remaining slots greedily."""
    # Build a chain: start from 0, add one bit at a time in a fixed order
    # This gives a chain of length 13 (0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095)
    chain = []
    mask = 0
    for b in range(N):
        chain.append(mask)
        mask |= (1 << b)
    chain.append(mask)  # full set
    # chain has 13 elements: sizes 0,1,2,...,12
    
    family = chain[:]
    used = set(chain)
    
    # Greedily fill remaining slots
    all_masks = list(range(TOTAL_MASKS))
    random.shuffle(all_masks)
    
    while len(family) < M:
        best_mask = -1
        best_gain = -1
        for mask in all_masks:
            if mask in used:
                continue
            gain = 0
            for f in family:
                gain += _comp_matrix[mask][f]
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
                if gain == len(family):
                    break
        if best_mask >= 0:
            family.append(best_mask)
            used.add(best_mask)
        else:
            break
    
    return family

def _simulated_annealing(family, max_time):
    """Simulated annealing with efficient delta computation using precomputed matrix."""
    current = family[:]
    current_score = _score_family(current)
    best = current[:]
    best_score = current_score
    m = len(current)
    
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    start_time = time.time()
    no_improve = 0
    
    while time.time() < max_time and T > T_min:
        i = random.randrange(m)
        new_mask = random.randrange(TOTAL_MASKS)
        if new_mask == current[i]:
            continue
        
        # Check distinctness
        is_dup = False
        for k in range(m):
            if k != i and current[k] == new_mask:
                is_dup = True
                break
        if is_dup:
            continue
        
        old_mask = current[i]
        # Compute delta using precomputed matrix
        delta = 0
        for j in range(m):
            if j == i:
                continue
            other = current[j]
            old_rel = _comp_matrix[old_mask][other]
            new_rel = _comp_matrix[new_mask][other]
            if new_rel != old_rel:
                delta += (1 if new_rel else 0) - (1 if old_rel else 0)
        
        new_score = current_score + delta
        
        if new_score > current_score:
            current[i] = new_mask
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best = current[:]
        else:
            # Accept with probability exp(delta/T)
            if random.random() < np.exp(delta / T):
                current[i] = new_mask
                current_score = new_score
            no_improve += 1
        
        T *= cooling
    
    return best, best_score

def _hill_climb_family(family, max_time):
    """Hill climbing: try all replacements for each position until no improvement."""
    current = family[:]
    current_score = _score_family(current)
    m = len(current)
    all_masks = list(range(TOTAL_MASKS))
    
    improved = True
    while improved and time.time() < max_time:
        improved = False
        for i in range(m):
            if time.time() > max_time:
                break
            old_mask = current[i]
            for new_mask in all_masks:
                if new_mask == old_mask:
                    continue
                # Check distinctness
                is_dup = False
                for k in range(m):
                    if k != i and current[k] == new_mask:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                delta = 0
                for j in range(m):
                    if j == i:
                        continue
                    other = current[j]
                    old_rel = _comp_matrix[old_mask][other]
                    new_rel = _comp_matrix[new_mask][other]
                    if new_rel != old_rel:
                        delta += (1 if new_rel else 0) - (1 if old_rel else 0)
                
                if delta > 0:
                    current[i] = new_mask
                    current_score += delta
                    improved = True
                    break
    
    return current, current_score

def construct():
    """Construct a family of 40 distinct subsets of {0..11} maximizing strict subset pairs."""
    best_family = None
    best_score = -1
    
    # Phase 1: Chain-seeded initialization + greedy fill
    family1 = _build_chain_family()
    s1 = _score_family(family1)
    best_family = family1[:]
    best_score = s1
    
    # Phase 2: Simulated annealing from best
    if time.time() < _DEADLINE - 10.0:
        t_end = min(_DEADLINE - 10.0, time.time() + 20.0)
        f2, s2 = _simulated_annealing(best_family, max_time=t_end)
        if s2 > best_score:
            best_score = s2
            best_family = f2[:]
    
    # Phase 3: Hill climbing
    if time.time() < _DEADLINE - 5.0:
        f3, s3 = _hill_climb_family(best_family, max_time=_DEADLINE - 5.0)
        if s3 > best_score:
            best_score = s3
            best_family = f3
    
    # Phase 4: More SA restarts if time permits
    remaining = _DEADLINE - time.time()
    if remaining > 10.0:
        num_restarts = int(remaining / 15.0)
        for r in range(num_restarts):
            if time.time() > _DEADLINE - 5.0:
                break
            # Perturb best family
            perturbed = best_family[:]
            for _ in range(random.randint(3, 6)):
                idx = random.randrange(M)
                new_mask = random.randrange(TOTAL_MASKS)
                attempts = 0
                while new_mask in perturbed and attempts < 30:
                    new_mask = random.randrange(TOTAL_MASKS)
                    attempts += 1
                if new_mask not in perturbed:
                    perturbed[idx] = new_mask
            
            t_end = min(_DEADLINE - 5.0, time.time() + max(2.0, (remaining / num_restarts) * 0.7))
            f4, s4 = _simulated_annealing(perturbed, max_time=t_end)
            if s4 > best_score:
                best_score = s4
                best_family = f4[:]
                
                # Quick hill climb after improvement
                if time.time() < _DEADLINE - 3.0:
                    f5, s5 = _hill_climb_family(best_family, max_time=_DEADLINE - 3.0)
                    if s5 > best_score:
                        best_score = s5
                        best_family = f5
    
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
