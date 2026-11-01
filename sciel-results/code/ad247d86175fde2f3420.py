# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import scipy.sparse

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Uses Multi-Start Greedy with Sparse Conflict Matrix.
    Includes norm^2=1, 2, 3, 4 candidates.
    """
    D = 11
    deadline = time.time() + 110.0 # Safety margin
    
    # Generate candidate vectors
    candidates = []
    
    # Type 0: one ±1 (norm^2 = 1)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Type 1: two ±1's (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Type 2: three ±1's (norm^2 = 3)
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    # Type 3: one ±2 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
            
    # Type 4: two ±2's (norm^2 = 8) - WAIT. 
    # If we include norm^2=8, max norm becomes sqrt(8) ~ 2.82.
    # Dist must be >= 2.82, i.e. dist^2 >= 8.
    # This makes constraints MUCH tighter.
    # Current max norm^2 in code was 4.
    # Let's stick to max norm^2 = 4 to allow denser packing.
    # So NO norm^2=8 vectors.
    
    # Type 5: four ±1's (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute squared norms
    norms_sq = np.sum(candidates**2, axis=1)
    max_norm_sq = np.max(norms_sq) # Should be 4
    
    # Compute pairwise dot products
    dots = candidates @ candidates.T
    
    # Compute squared distances
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Conflict matrix: conflict[i,j] = True if dist_sq[i,j] < max_norm_sq
    conflict = (dist_sq < max_norm_sq)
    np.fill_diagonal(conflict, False)
    
    # Convert to sparse for faster neighbor retrieval
    conflict_sparse = scipy.sparse.csr_matrix(conflict)
    
    best_size = 0
    best_selection = []
    
    np.random.seed(42)
    
    def kempe_chain_improve(selected_list, conflict_sparse, candidates, n, max_iter=50):
        """
        Apply Kempe-chain local search to improve an independent set.
        Try multiple partition strategies and swap elements to increase set size.
        """
        if len(selected_list) == 0:
            return selected_list
        
        selected_set = set(selected_list)
        current_size = len(selected_set)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
                
            improved = False
            
            # Strategy 1: Sign-based partition (based on first non-zero component sign)
            partition_a = []
            partition_b = []
            for idx in selected_set:
                v = candidates[idx]
                # Find first non-zero element
                nz = np.nonzero(v)[0]
                if len(nz) > 0 and v[nz[0]] > 0:
                    partition_a.append(idx)
                else:
                    partition_b.append(idx)
            
            # Strategy 2: Random partition
            shuffled = list(selected_set)
            np.random.shuffle(shuffled)
            mid = len(shuffled) // 2
            partition_a = shuffled[:mid]
            partition_b = shuffled[mid:]
            
            # Strategy 3: Percentile-based on norm
            norms = [np.sum(candidates[i]**2) for i in selected_set]
            sorted_by_norm = sorted(selected_set, key=lambda x: np.sum(candidates[x]**2))
            mid = len(sorted_by_norm) // 2
            partition_a = sorted_by_norm[:mid]
            partition_b = sorted_by_norm[mid:]
            
            for part_a, part_b in [(partition_a, partition_b)]:
                if len(part_a) == 0 or len(part_b) == 0:
                    continue
                    
                # Build conflict graph within selected set
                # For each node in part_a, check if it conflicts with nodes in part_b
                # We want to find swaps that increase the set size
                
                # Try removing one from part_a and adding one from part_b (or vice versa)
                # But we need to ensure the result is still independent
                
                # More effective: try to find a node in part_b that can replace a node in part_a
                # such that we can then add more nodes
                
                # Simpler approach: try all single swaps
                for i in part_a:
                    if time.time() > deadline:
                        break
                    
                    # Remove i
                    new_set = set(selected_set)
                    new_set.discard(i)
                    
                    # Try adding each node from part_b
                    for j in part_b:
                        if j in new_set:
                            continue
                        
                        # Check if j conflicts with any node in new_set
                        conflicts = False
                        for k in new_set:
                            if conflict_sparse[i, j].toarray()[0, 0] == 1:  # i and j conflict
                                # But i is removed, so check j against new_set
                                pass
                            if conflict_sparse[j, k].toarray()[0, 0] == 1:
                                conflicts = True
                                break
                        
                        if not conflicts:
                            # Try to add more nodes from part_a that were blocked by i
                            for m in part_a:
                                if m == i or m in new_set:
                                    continue
                                
                                # Check if m conflicts with any node in new_set ∪ {j}
                                m_conflicts = False
                                for k in new_set:
                                    if conflict_sparse[m, k].toarray()[0, 0] == 1:
                                        m_conflicts = True
                                        break
                                if not m_conflicts and not conflict_sparse[m, j].toarray()[0, 0]:
                                    new_set.add(m)
                            
                            if len(new_set) > current_size:
                                selected_set = new_set
                                current_size = len(selected_set)
                                improved = True
                                break
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return list(selected_set)
    
    # Multi-Start Greedy
    for restart in range(500):
        if time.time() > deadline:
            break
            
        active = np.ones(n, dtype=bool)
        # Initial degrees
        degrees = conflict_sparse.sum(axis=1).A1.astype(np.int32)
        
        selected = []
        
        # Create a list of active indices
        active_indices = np.arange(n)
        
        for _ in range(n):
            if not np.any(active):
                break
                
            # Find active indices
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            # Get degrees of active nodes
            active_degrees = degrees[active_indices]
            
            # Find min degree
            min_deg = np.min(active_degrees)
            
            # Find all active nodes with min degree
            min_mask = (active_degrees == min_deg)
            min_candidates = active_indices[min_mask]
            
            # Randomly pick one
            chosen = np.random.choice(min_candidates)
            
            selected.append(chosen)
            
            # Update active and degrees
            neighbors = np.array(conflict_sparse[chosen].indices)
            
            # Mark chosen and neighbors as inactive
            to_remove = np.concatenate(([chosen], neighbors))
            
            for u in to_remove:
                if not active[u]:
                    continue
                active[u] = False
                neigh_u = np.array(conflict_sparse[u].indices)
                degrees[neigh_u] -= 1
                
        # Apply Kempe-chain improvement
        improved_selected = kempe_chain_improve(selected, conflict_sparse, candidates, n, max_iter=20)
        
        if len(improved_selected) > best_size:
            best_size = len(improved_selected)
            best_selection = improved_selected.copy()
            
    result = candidates[best_selection]
    return result
# EVOLVE-BLOCK-END
