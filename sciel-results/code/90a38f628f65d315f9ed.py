"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Generate candidate vectors with small integer coordinates
    # Use coordinates in {-1, 0, 1} first, then expand if time permits
    candidates = []
    # All nonzero vectors with entries in {-1, 0, 1}
    # That's 3^11 - 1 = 177146 vectors, too many. Use a subset.
    # Use vectors with exactly 2 or 3 nonzero entries
    from itertools import combinations
    
    # Vectors with exactly 2 nonzero entries (±1)
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                candidates.append(np.array(v, dtype=np.int64))
    
    # Vectors with exactly 3 nonzero entries (±1)
    for i, j, k in combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(np.array(v, dtype=np.int64))
    
    # Also include vectors with a single entry of magnitude 2
    for i in range(D):
        for s in (2, -2):
            v = [0] * D
            v[i] = s
            candidates.append(np.array(v, dtype=np.int64))
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute norms
    norms = np.sqrt(np.sum(candidates ** 2, axis=1))
    
    # Greedy initial construction: sort by norm, add if compatible
    # Constraint: for all pairs in set, ||vi - vj|| >= max_norm_in_set
    # Start with the smallest norm vectors
    
    order = np.argsort(norms)
    selected = []
    selected_norms = []
    max_norm = 0
    
    for idx in order:
        if time.time() > deadline - 30:
            break
        v = candidates[idx]
        nrm = norms[idx]
        if nrm <= max_norm + 1e-9:
            # Check distance to all selected
            if selected:
                dists = np.sqrt(np.sum((np.array(selected) - v) ** 2, axis=1))
                if np.min(dists) < max_norm - 1e-9:
                    continue
            selected.append(v)
            selected_norms.append(nrm)
            max_norm = nrm
        else:
            # Adding this would increase max_norm, need all existing to have distance >= nrm
            if selected:
                dists = np.sqrt(np.sum((np.array(selected) - v) ** 2, axis=1))
                if np.min(dists) < nrm - 1e-9:
                    continue
            selected.append(v)
            selected_norms.append(nrm)
            max_norm = nrm
    
    best_set = list(selected)
    best_count = len(best_set)
    
    # Local search / SA to improve
    # Build conflict structure
    def check_compatible(vecs, max_n):
        """Check if all pairwise distances >= max_n"""
        if len(vecs) <= 1:
            return True
        arr = np.array(vecs)
        # Compute all pairwise distances
        # For efficiency, check incrementally
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                d = np.sqrt(np.sum((arr[i] - arr[j]) ** 2))
                if d < max_n - 1e-9:
                    return False
        return True
    
    def get_max_norm(vecs):
        if not vecs:
            return 0
        return max(np.sqrt(np.sum(v**2)) for v in vecs)
    
    # SA: try to add random vectors, or swap
    current_set = list(best_set)
    current_count = len(current_set)
    
    # Precompute distances from each candidate to each current vector for efficiency
    # This is expensive, so do it lazily
    
    temperature = 1.0
    cooling = 0.999
    
    sa_start = time.time()
    sa_budget = min(60, deadline - sa_start - 20)
    
    while time.time() < sa_start + sa_budget:
        # Try to add a random candidate
        if random.random() < 0.7:
            idx = random.randint(0, n_cand - 1)
            v = candidates[idx]
            nrm = norms[idx]
            
            # Check if adding v to current_set is valid
            new_max = max(get_max_norm(current_set), nrm) if current_set else nrm
            if current_set:
                arr = np.array(current_set)
                dists = np.sqrt(np.sum((arr - v) ** 2, axis=1))
                if np.min(dists) < new_max - 1e-9:
                    # Try swapping: remove a vector that conflicts with v
                    # Find vectors in current_set that are too close to v
                    too_close = [i for i, d in enumerate(dists) if d < new_max - 1e-9]
                    if too_close:
                        # Try removing one of the too_close vectors and adding v
                        for tc in random.sample(too_close, min(len(too_close), 3)):
                            new_set = current_set[:tc] + current_set[tc+1:] + [v]
                            new_max2 = get_max_norm(new_set)
                            if check_compatible(new_set, new_max2):
                                if len(new_set) >= len(current_set):
                                    current_set = new_set
                                    current_count = len(current_set)
                                    if current_count > best_count:
                                        best_set = list(current_set)
                                        best_count = current_count
                        continue
                    continue
                else:
                    current_set.append(v)
                    current_count += 1
                    if current_count > best_count:
                        best_set = list(current_set)
                        best_count = current_count
            else:
                current_set.append(v)
                current_count += 1
                if current_count > best_count:
                    best_set = list(current_set)
                    best_count = current_count
        else:
            # Try swapping a random vector for a random candidate
            if current_count > 1:
                out_idx = random.randint(0, current_count - 1)
                in_idx = random.randint(0, n_cand - 1)
                v_in = candidates[in_idx]
                
                new_set = current_set[:out_idx] + current_set[out_idx+1:] + [v_in]
                new_max = get_max_norm(new_set)
                if check_compatible(new_set, new_max):
                    current_set = new_set
                    current_count = len(current_set)
                    if current_count > best_count:
                        best_set = list(current_set)
                        best_count = current_count
        
        temperature *= cooling
    
    # Final polish: try to add more vectors greedily
    final_set = list(best_set)
    final_max = get_max_norm(final_set)
    
    for idx in order:
        if time.time() > deadline - 5:
            break
        v = candidates[idx]
        nrm = norms[idx]
        new_max = max(final_max, nrm)
        if final_set:
            arr = np.array(final_set)
            dists = np.sqrt(np.sum((arr - v) ** 2, axis=1))
            if np.min(dists) >= new_max - 1e-9:
                final_set.append(v)
                final_max = new_max
    
    if len(final_set) > best_count:
        best_set = final_set
        best_count = len(final_set)
    
    if time.time() > deadline - 2:
        return np.array(best_set, dtype=np.int64)
    
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
