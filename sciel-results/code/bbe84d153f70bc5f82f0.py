"""Structural rewrite: precompute comparability, use clique-growth local search."""
import time
import random

N = 12
M = 40
TOTAL_MASKS = 1 << N

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

# Precompute popcount for all masks
_POPCOUNT = [0] * TOTAL_MASKS
for _m in range(1, TOTAL_MASKS):
    _POPCOUNT[_m] = _POPCOUNT[_m >> 1] + (_m & 1)

def _comparable(a, b):
    """Check if a and b are comparable (one is strict subset of the other)."""
    if a == b:
        return False
    return (a & b) == a or (a & b) == b

def _build_comparability_table():
    """Build a list of lists: comp_table[m] = list of all masks n != m that are comparable to m."""
    comp_table = [[] for _ in range(TOTAL_MASKS)]
    for a in range(TOTAL_MASKS):
        for b in range(a + 1, TOTAL_MASKS):
            if (a & b) == a or (a & b) == b:
                comp_table[a].append(b)
                comp_table[b].append(a)
    return comp_table

def _score_family(family, comp_table):
    """Count unordered pairs (i, j) with i < j such that one is a strict subset of the other."""
    count = 0
    m = len(family)
    for i in range(m):
        for j in range(i + 1, m):
            if _comparable(family[i], family[j]):
                count += 1
    return count

def _contributions(family, comp_table):
    """For each position i, count how many other family members are comparable to family[i]."""
    n = len(family)
    contrib = [0] * n
    fam_set = set(family)
    for i in range(n):
        for j in range(n):
            if i != j and _comparable(family[i], family[j]):
                contrib[i] += 1
    return contrib

def _greedy_build(comp_table):
    """Construct a family greedily by adding the mask most comparable to existing ones."""
    family = []
    remaining = set(range(TOTAL_MASKS))
    
    # Start with empty set for a natural anchor
    family.append(0)
    remaining.remove(0)
    
    for _ in range(M - 1):
        best_mask = -1
        best_score = -1
        for m in remaining:
            comp_count = 0
            for f in family:
                if _comparable(m, f):
                    comp_count += 1
            if comp_count > best_score:
                best_score = comp_count
                best_mask = m
                if best_score == len(family):
                    break
        if best_mask != -1:
            family.append(best_mask)
            remaining.remove(best_mask)
    
    return family

def _local_search_clique_growth(family, comp_table):
    """
    Clique-growth local search: target the least-connected member,
    swap it for the mask with maximum comparability to the rest.
    """
    current_family = family[:]
    current_score = _score_family(current_family, comp_table)
    n = len(current_family)
    
    while time.time() < _DEADLINE - 3.0:
        # Compute contributions
        contrib = _contributions(current_family, comp_table)
        
        # Find the least connected member
        min_contrib_idx = min(range(n), key=lambda i: contrib[i])
        min_contrib_val = contrib[min_contrib_idx]
        
        fam_set = set(current_family)
        
        # Find best replacement for position min_contrib_idx
        best_new_mask = -1
        best_delta = 0
        
        for m in range(TOTAL_MASKS):
            if m in fam_set:
                continue
            # New contribution if we put m at position min_contrib_idx
            new_contrib = 0
            for j in range(n):
                if j != min_contrib_idx and _comparable(m, current_family[j]):
                    new_contrib += 1
            delta = new_contrib - min_contrib_val
            if delta > best_delta:
                best_delta = delta
                best_new_mask = m
        
        if best_new_mask != -1:
            current_family[min_contrib_idx] = best_new_mask
            current_score += best_delta
        else:
            break
    
    return current_family, current_score

def _full_polish(family, comp_table):
    """Full 4096-candidate local search: try swapping every position with every candidate."""
    current_family = family[:]
    current_score = _score_family(current_family, comp_table)
    n = len(current_family)
    
    improved = True
    while improved and time.time() < _DEADLINE - 2.0:
        improved = False
        contrib = _contributions(current_family, comp_table)
        fam_set = set(current_family)
        
        for i in range(n):
            if time.time() > _DEADLINE - 2.0:
                break
            
            best_new_mask = -1
            best_delta = 0
            
            for m in range(TOTAL_MASKS):
                if m in fam_set:
                    continue
                new_contrib = 0
                for j in range(n):
                    if j != i and _comparable(m, current_family[j]):
                        new_contrib += 1
                delta = new_contrib - contrib[i]
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = m
            
            if best_new_mask != -1:
                current_family[i] = best_new_mask
                current_score += best_delta
                improved = True
                break  # Restart with updated family
    
    return current_family, current_score

def _random_restarts(family, comp_table, num_restarts=5):
    """Try a few random perturbations followed by local search."""
    best_family = family[:]
    best_score = _score_family(best_family, comp_table)
    
    for _ in range(num_restarts):
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Perturb: randomly swap 3-5 elements
        perturbed = best_family[:]
        n = len(perturbed)
        for _ in range(random.randint(3, 5)):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        
        # Local search on perturbed
        perturbed, score = _local_search_clique_growth(perturbed, comp_table)
        
        if score > best_score:
            best_family = perturbed
            best_score = score
    
    return best_family, best_score

def construct():
    """Construct a family of 40 distinct subsets of {0..11} maximizing strict subset pairs."""
    comp_table = _build_comparability_table()
    
    # Phase 1: Greedy construction
    family = _greedy_build(comp_table)
    
    # Phase 2: Clique-growth local search
    family, score = _local_search_clique_growth(family, comp_table)
    
    # Phase 3: Full polish
    family, score = _full_polish(family, comp_table)
    
    # Phase 4: Random restarts
    family, score = _random_restarts(family, comp_table, num_restarts=3)
    
    # Final full polish
    family, score = _full_polish(family, comp_table)
    
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
