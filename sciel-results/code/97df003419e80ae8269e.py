import numpy as np
import itertools
import time

D = 11

def _generate_candidates(max_norm_sq):
    """
    Generate all integer vectors in R^D with squared norm <= max_norm_sq.
    """
    vectors = []
    # We need to generate all integer vectors v such that sum(v_i^2) <= max_norm_sq
    # Since D=11 and max_norm_sq is small (e.g., 4), we can iterate over possible components.
    # Max component value is sqrt(max_norm_sq).
    
    max_val = int(np.sqrt(max_norm_sq))
    
    # Recursive generation or itertools.product
    # Range for each component: -max_val to max_val
    # This is (2*max_val+1)^D which is huge if max_val is large.
    # But we filter by norm.
    
    # Better: generate by weight (number of non-zero components) and values.
    # For max_norm_sq=4:
    # Possible squared norms: 1, 2, 3, 4.
    # 
    # Norm 1: (1, 0, ..., 0) permutations.
    # Norm 2: (1, 1, 0, ...) or (sqrt(2)... no, integers). (1,1) or (sqrt(2) no).
    #          So only (1,1) type? No, (1,1) has norm sq 2.
    #          Are there other integer vectors with norm sq 2? No.
    # Norm 3: (1, 1, 1, ...).
    # Norm 4: (2, 0, ...) or (1, 1, 1, 1, ...).
    
    # General approach:
    # Iterate over all combinations of indices and values.
    # Since max_norm_sq is small, the number of non-zero components is small.
    
    vectors_set = set()
    
    # Helper to add a vector
    def add_vec(vals):
        # vals is a list of (index, value) pairs
        v = [0] * D
        for i, val in vals:
            v[i] = val
        if v != [0]*D:
            vectors_set.add(tuple(v))
            
    # Norm 1
    for i in range(D):
        add_vec([(i, 1)])
        add_vec([(i, -1)])
        
    # Norm 2
    # Only possible with two 1s (or -1s)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                add_vec([(i, si), (j, sj)])
                
    # Norm 3
    # Only possible with three 1s
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    add_vec([(i, si), (j, sj), (k, sk)])
                    
    # Norm 4
    # Type A: One 2 (or -2)
    for i in range(D):
        add_vec([(i, 2)])
        add_vec([(i, -2)])
        
    # Type B: Four 1s
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        add_vec([(i, si), (j, sj), (k, sk), (l, sl)])
                        
    # If max_norm_sq > 4, we would need more cases.
    # For now, assume max_norm_sq is 4.
    # If we want to be general:
    if max_norm_sq > 4:
        # This implementation is hardcoded for 4.
        # For a more robust solution, we'd need a general generator.
        # But 4 is a good candidate for D=11.
        pass

    return np.array(list(vectors_set), dtype=np.int64)

def _greedy_mis(conflict, n, deadline, rng_seed=0):
    """
    Greedy Maximum Independent Set with random restarts.
    """
    best_set = np.zeros(n, dtype=bool)
    best_size = 0
    start_time = time.time()
    
    # Pre-compute degrees for initial ordering
    degrees = conflict.sum(axis=1)
    
    for trial in range(500):
        if time.time() > deadline:
            break
            
        if trial == 0:
            # First try: degree ordering
            order = np.argsort(degrees)
        else:
            rng = np.random.default_rng(rng_seed + trial)
            # Random permutation, but bias towards low degree?
            # Simple random permutation
            order = rng.permutation(n)
            
        selected = np.zeros(n, dtype=bool)
        removed = np.zeros(n, dtype=bool)
        
        for idx in order:
            if removed[idx]:
                continue
            selected[idx] = True
            # Mark neighbors
            removed[conflict[idx]] = True
            
        size = selected.sum()
        if size > best_size:
            best_size = size
            best_set = selected.copy()
            
    return best_set, best_size

def _local_search_mis(conflict, n, initial_set, deadline):
    """
    Local search: try to add vertices, removing conflicts if it increases size.
    """
    current_set = initial_set.copy()
    current_size = current_set.sum()
    start_time = time.time()
    improved = True
    
    # Precompute list of vertices not in set
    # We will iterate through all vertices and try to add them
    
    while improved and time.time() < deadline:
        improved = False
        # Shuffle order of candidates to avoid bias
        rng = np.random.default_rng(int(time.time() * 1000))
        candidates = rng.permutation(n)
        
        for v in candidates:
            if current_set[v]:
                continue
                
            # Check if v can be added
            # Conflicts with current set:
            neighbors_in_set = np.where(conflict[v] & current_set)[0]
            
            if len(neighbors_in_set) == 0:
                # Can add directly
                current_set[v] = True
                current_size += 1
                improved = True
            else:
                # Check if removing neighbors and adding v increases size
                # New size = current_size - len(neighbors_in_set) + 1
                if current_size - len(neighbors_in_set) + 1 > current_size:
                    # This implies -len(neighbors) + 1 > 0 => len(neighbors) < 1
                    # Which is the case above.
                    # So this branch is never taken if we require strict increase?
                    # Wait. If we remove 1 neighbor and add 1, size is same.
                    # We want to improve size.
                    # So this simple add-remove doesn't help if it doesn't increase size.
                    # We need a more complex move: Add v, remove K neighbors, add L others?
                    # That's complex.
                    pass
                    
    return current_set, current_size

def construct():
    deadline = time.time() + 140.0 # 10s margin
    
    # Try M=4
    candidates = _generate_candidates(4)
    n = len(candidates)
    
    if n == 0:
        return np.array([], dtype=np.int64)
        
    # Compute conflict matrix
    # Condition: ||u - v||^2 < M_target
    # M_target = 4
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u.v
    # We want to find u, v such that ||u - v||^2 >= 4 is VALID.
    # Conflict if ||u - v||^2 < 4.
    
    # Compute squared norms
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Dot product matrix
    dot = candidates @ candidates.T
    
    # Dist sq matrix
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot
    
    # Conflict if dist_sq < 4
    conflict = (dist_sq < 4)
    np.fill_diagonal(conflict, False)
    
    # Run Greedy MIS
    best_set, best_size = _greedy_mis(conflict, n, deadline - 5.0)
    
    # Run Local Search on the best set
    best_set, best_size = _local_search_mis(conflict, n, best_set, deadline - 1.0)
    
    result = candidates[best_set]
    
    # Verify
    if len(result) == 0:
        return result
        
    # Check validity
    # Max norm sq in result
    max_norm_sq = np.max(np.sum(result * result, axis=1))
    # Min dist sq
    if len(result) > 1:
        # Efficient check?
        # For small k, O(k^2) is fine.
        d = len(result)
        min_dist_sq = np.inf
        for i in range(d):
            for j in range(i+1, d):
                dsq = np.sum((result[i] - result[j])**2)
                if dsq < min_dist_sq:
                    min_dist_sq = dsq
        if min_dist_sq < max_norm_sq:
            # Invalid, return empty or fallback?
            # The problem requires validity.
            # If greedy produced invalid set, we should filter?
            # But greedy ensures no conflicts in the graph.
            # The graph was built with M=4.
            # If the set contains vectors with norm < 4, the actual M might be smaller?
            # No, M is max norm in set.
            # If max norm is 2, condition is dist >= 2.
            # Our graph enforced dist >= 4.
            # So dist >= 4 implies dist >= 2.
            # So it is valid.
            pass
            
    return result

# Fallback if M=4 fails or is slow
def construct_fallback():
    # Use seed logic
    pass

# Main entry
def construct():
    try:
        return _construct_impl()
    except Exception as e:
        # Fallback to seed-like behavior
        import numpy as np
        import itertools
        D = 11
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

def _construct_impl():
    deadline = time.time() + 140.0
    
    candidates = _generate_candidates(4)
    n = len(candidates)
    
    if n == 0:
        return np.array([], dtype=np.int64)
        
    norms_sq = np.sum(candidates * candidates, axis=1)
    dot = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot
    conflict = (dist_sq < 4)
    np.fill_diagonal(conflict, False)
    
    best_set, best_size = _greedy_mis(conflict, n, deadline - 5.0)
    best_set, best_size = _local_search_mis(conflict, n, best_set, deadline - 1.0)
    
    result = candidates[best_set]
    return result

# Ensure the function name is construct
# The prompt asks to keep function names.
# I defined construct() and _construct_impl().
# I should just define construct() properly.

import numpy as np
import itertools
import time

D = 11

def _generate_candidates(max_norm_sq=4):
    vectors_set = set()
    def add_vec(vals):
        v = [0] * D
        for i, val in vals:
            v[i] = val
        if v != [0]*D:
            vectors_set.add(tuple(v))
            
    for i in range(D):
        add_vec([(i, 1)])
        add_vec([(i, -1)])
        
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                add_vec([(i, si), (j, sj)])
                
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    add_vec([(i, si), (j, sj), (k, sk)])
                    
    for i in range(D):
        add_vec([(i, 2)])
        add_vec([(i, -2)])
        
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        add_vec([(i, si), (j, sj), (k, sk), (l, sl)])

    return np.array(list(vectors_set), dtype=np.int64)

def _greedy_mis(conflict, n, deadline, rng_seed=0):
    best_set = np.zeros(n, dtype=bool)
    best_size = 0
    start_time = time.time()
    degrees = conflict.sum(axis=1)
    
    for trial in range(1000):
        if time.time() > deadline:
            break
            
        if trial == 0:
            order = np.argsort(degrees)
        else:
            rng = np.random.default_rng(rng_seed + trial)
            order = rng.permutation(n)
            
        selected = np.zeros(n, dtype=bool)
        removed = np.zeros(n, dtype=bool)
        
        for idx in order:
            if removed[idx]:
                continue
            selected[idx] = True
            removed[conflict[idx]] = True
            
        size = selected.sum()
        if size > best_size:
            best_size = size
            best_set = selected.copy()
            
    return best_set, best_size

def _local_search_mis(conflict, n, initial_set, deadline):
    current_set = initial_set.copy()
    current_size = current_set.sum()
    start_time = time.time()
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        rng = np.random.default_rng(int(time.time() * 1000) % 10000)
        candidates = rng.permutation(n)
        
        for v in candidates:
            if current_set[v]:
                continue
            neighbors_in_set = np.where(conflict[v] & current_set)[0]
            if len(neighbors_in_set) == 0:
                current_set[v] = True
                current_size += 1
                improved = True
            elif len(neighbors_in_set) == 1:
                # Swapping 1 for 1 doesn't increase size.
                # But maybe it allows adding more later?
                # Skip for now.
                pass
                
    return current_set, current_size

def construct():
    deadline = time.time() + 140.0
    
    candidates = _generate_candidates(4)
    n = len(candidates)
    
    if n == 0:
        return np.array([], dtype=np.int64)
        
    norms_sq = np.sum(candidates * candidates, axis=1)
    dot = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot
    conflict = (dist_sq < 4)
    np.fill_diagonal(conflict, False)
    
    best_set, best_size = _greedy_mis(conflict, n, deadline - 5.0)
    best_set, best_size = _local_search_mis(conflict, n, best_set, deadline - 1.0)
    
    result = candidates[best_set]
    return result
