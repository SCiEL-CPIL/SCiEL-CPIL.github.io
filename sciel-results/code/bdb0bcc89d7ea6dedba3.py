# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a maximum-size subset of Z/205Z with no two elements differing by a
    nonzero quadratic residue modulo 205.

    Uses exact branch-and-bound (maximum independent set) on the conflict graph
    where edges connect pairs whose difference is a forbidden (nonzero square)
    mod 205. Bitset-based conflict pruning and degree-based vertex ordering
    are used to make the search efficient.
    """
    m = 205

    # Compute forbidden differences: nonzero quadratic residues mod 205
    forbidden = set()
    for x in range(1, m):
        r = (x * x) % m
        if r != 0:
            forbidden.add(r)

    # Build conflict graph: conflict[i][j] = True if i and j cannot both be in the set
    # i.e., (i - j) % m is in forbidden OR (j - i) % m is in forbidden
    # Since forbidden is symmetric (if d is a square, -d might not be, but we check both directions)
    # Actually, we need: for pair (a,b), neither (a-b)%m nor (b-a)%m should be in forbidden.
    # So conflict if (a-b)%m in forbidden or (b-a)%m in forbidden.
    
    # Use bitsets for efficiency: conflict_mask[i] is a Python int (bitset) of vertices conflicting with i
    conflict_mask = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # Precompute degree for each vertex
    degree = [bin(conflict_mask[i]).count('1') for i in range(m)]

    # Best solution found so far
    best_size = 0
    best_set = [0]  # trivial solution: {0}

    # Branch and bound for maximum independent set
    # We use a recursive approach with bitsets for the candidate set
    
    def branch_and_bound(candidates, current_set, current_size):
        nonlocal best_size, best_set
        
        if current_size + len(candidates) <= best_size:
            return
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        # Ordering: pick vertex with highest degree in the induced subgraph on candidates
        # For efficiency, we try vertices in order of decreasing degree within candidates
        # But for simplicity and to keep it tractable, we use a heuristic ordering
        
        # Convert candidates to a list for ordering
        cand_list = [i for i in range(m) if (candidates >> i) & 1]
        
        if not cand_list:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        # Order by degree in the candidate subgraph (descending)
        def deg_in_candidates(v):
            return bin(candidates & conflict_mask[v]).count('1')
        
        cand_list.sort(key=deg_in_candidates, reverse=True)
        
        for v in cand_list:
            if not ((candidates >> v) & 1):
                continue
            
            # Include v
            new_set = current_set + [v]
            # New candidates: those in candidates that don't conflict with v
            new_candidates = candidates & ~conflict_mask[v] & ~(1 << v)
            branch_and_bound(new_candidates, new_set, current_size + 1)
            
            # Exclude v
            candidates = candidates & ~(1 << v)
            branch_and_bound(candidates, current_set, current_size)
    
    # Start with all vertices as candidates
    all_candidates = (1 << m) - 1
    branch_and_bound(all_candidates, [], 0)
    
    return np.array(best_set, dtype=np.int64)


# EVOLVE-BLOCK-END
