"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = time.time() + 85.0

N = 12
M = 40
TOTAL_MASKS = 1 << N

def _comparable(a, b):
    """Check if a and b are comparable (one is strict subset of other)."""
    if a == b:
        return False
    return (a & b) == a or (a & b) == b

def _score_family(family):
    """Count unordered pairs (i, j) with i < j such that one is a strict subset of the other."""
    count = 0
    m = len(family)
    for i in range(m):
        for j in range(i + 1, m):
            if _comparable(family[i], family[j]):
                count += 1
    return count

def _greedy_build():
    """Construct a family greedily by adding the mask that is most comparable to existing ones."""
    family = []
    remaining = set(range(TOTAL_MASKS))
    
    # Precompute popcount for tie-breaking? 
    # Greedy: pick mask with max comparability to current family.
    for _ in range(M):
        best_mask = -1
        best_score = -1
        # To make it deterministic and potentially better, we can iterate in a specific order
        # or just pick the best.
        # Optimization: We can stop early if we find a perfect score? No.
        for m in remaining:
            # Calculate comparability with current family
            comp_count = 0
            for f in family:
                if _comparable(m, f):
                    comp_count += 1
            # Tie-breaker: smaller popcount? Or just first found?
            # Let's use smaller popcount to encourage diversity in sizes if scores are equal?
            # Actually, for a chain, we want consecutive sizes.
            # Let's just pick the one with max comp_count.
            if comp_count > best_score:
                best_score = comp_count
                best_mask = m
                if best_score == len(family):
                    # Can't get better than this
                    break
        
        if best_mask != -1:
            family.append(best_mask)
            remaining.remove(best_mask)
        else:
            # Should not happen unless remaining is empty
            break
            
    return family

def _local_search(family):
    """Improve family by swapping elements."""
    current_family = family[:]
    current_score = _score_family(current_family)
    
    # Precompute comparability matrix for current family? 
    # Or just recompute deltas.
    
    improved = True
    while improved and time.time() < _DEADLINE - 5.0:
        improved = False
        # Calculate current contribution of each element
        # contrib[i] = number of j != i such that comparable(family[i], family[j])
        contrib = [0] * M
        for i in range(M):
            for j in range(M):
                if i != j and _comparable(current_family[i], current_family[j]):
                    contrib[i] += 1
        
        for i in range(M):
            if time.time() > _DEADLINE - 5.0:
                return current_family, current_score
            
            # Try swapping current_family[i] with any m not in current_family
            # We can iterate through all masks and check if it's in family
            # To speed up, keep a set of current family
            fam_set = set(current_family)
            
            best_new_mask = -1
            best_delta = 0
            
            for m in range(TOTAL_MASKS):
                if m in fam_set:
                    continue
                
                # Calculate new contribution of m to family without i
                # new_contrib_m = sum(1 for j in range(M) if j != i and _comparable(m, current_family[j]))
                new_contrib_m = 0
                for j in range(M):
                    if j != i:
                        if _comparable(m, current_family[j]):
                            new_contrib_m += 1
                
                delta = new_contrib_m - contrib[i]
                
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = m
            
            if best_new_mask != -1:
                # Perform swap
                old_mask = current_family[i]
                current_family[i] = best_new_mask
                current_score += best_delta
                improved = True
                # Update contrib? 
                # It's safer to recompute contrib at the start of the next outer loop iteration
                # Or just break out and recompute.
                break # Restart i loop with updated family
    
    return current_family, current_score

def construct():
    """Construct a family of 40 distinct subsets of {0..11} maximizing strict subset pairs."""
    
    # Phase 1: Greedy construction
    family = _greedy_build()
    
    # Phase 2: Local search polish
    family, score = _local_search(family)
    
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
