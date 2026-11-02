"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.linalg import eigs


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s limit

    # Generate all weight-4 vectors with ±1 entries
    # C(11,4) = 330 positions, 2^4 = 16 sign combos -> 5280 vectors
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)

    # Precompute dot product matrix
    dots = candidates @ candidates.T

    # Conflict: dot > 2 (for norm²=4 vectors, need dot ≤ 2)
    incompat = (dots > 2).astype(np.int8)
    np.fill_diagonal(incompat, 0)

    degrees = incompat.sum(axis=1)

    best_set = set()
    best_size = 0

    # Strategy 1: Spectral clustering approach
    # Use the smallest non-trivial eigenvalues of the Laplacian to find clusters
    if time.time() < deadline - 60.0:
        # Compute Laplacian: L = D - A where A is adjacency (incompat)
        L = np.diag(degrees.astype(np.float64)) - incompat.astype(np.float64)
        
        # Get first few eigenvectors (smallest eigenvalues)
        try:
            # Use sparse eigenvalue solver for efficiency
            eigvals, eigvecs = eigs(L, k=6, which='SM', return_eigenvectors=True)
            # Sort by eigenvalue
            idx = np.argsort(eigvals.real)
            eigvals = eigvals[idx].real
            eigvecs = eigvecs[:, idx]
            
            # Use the Fiedler vectors to assign cluster labels
            # Cluster by sign of eigenvector values
            for vec_idx in range(1, min(4, eigvecs.shape[1])):
                vec = eigvecs[:, vec_idx]
                # Try selecting nodes where vec > 0 and vec < 0
                for threshold in [0.0, np.median(vec), np.percentile(vec, 30), np.percentile(vec, 70)]:
                    if time.time() > deadline - 50.0:
                        break
                    
                    # Select nodes above threshold
                    mask = vec > threshold
                    cluster_nodes = np.where(mask)[0]
                    
                    if len(cluster_nodes) < 10:
                        continue
                    
                    # Greedy within cluster
                    selected = []
                    selected_set = set()
                    # Order by degree within cluster
                    sub_incompat = incompat[np.ix_(cluster_nodes, cluster_nodes)]
                    sub_degrees = sub_incompat.sum(axis=1)
                    order = cluster_nodes[np.argsort(sub_degrees)]
                    
                    for i in order:
                        if i in selected_set:
                            continue
                        if not selected:
                            selected.append(i)
                            selected_set.add(i)
                            continue
                        if np.all(incompat[i, np.array(selected)] == 0):
                            selected.append(i)
                            selected_set.add(i)
                    
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_set = set(selected)
                        
        except Exception:
            pass

    # Strategy 2: Greedy by minimum degree with multiple orderings
    if time.time() < deadline - 30.0:
        order = np.argsort(degrees)
        selected = []
        selected_set = set()
        for i in order:
            if i in selected_set:
                continue
            if not selected:
                selected.append(i)
                selected_set.add(i)
                continue
            if np.all(incompat[i, np.array(selected)] == 0):
                selected.append(i)
                selected_set.add(i)

        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)

    # Strategy 3: Random restarts with biased sampling
    rng = np.random.default_rng(123)
    
    # Precompute weights for sampling (prefer low-degree nodes)
    weights = 1.0 / (degrees + 1)
    weights = weights / weights.sum()
    
    n_restarts = 0
    max_restarts = 300
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20.0:
            break
        n_restarts += 1
        
        order = rng.choice(n_candidates, size=n_candidates, replace=False, p=weights)
        
        selected = []
        selected_set = set()
        for i in order:
            if i in selected_set:
                continue
            if not selected:
                selected.append(i)
                selected_set.add(i)
                continue
            if np.all(incompat[i, np.array(selected)] == 0):
                selected.append(i)
                selected_set.add(i)

        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            if best_size >= 900:
                break

    # Strategy 4: Aggressive local search with swaps
    if time.time() < deadline - 10.0 and best_set:
        current_set = list(best_set)
        current_set_arr = np.array(current_set)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Compute conflicts for all non-selected nodes
            if len(current_set_arr) > 0:
                conflicts = incompat[:, current_set_arr].sum(axis=1)
            else:
                conflicts = np.zeros(n_candidates, dtype=np.int64)
            
            # Mask out already selected nodes
            in_set_mask = np.zeros(n_candidates, dtype=bool)
            in_set_mask[current_set_arr] = True
            
            # Try to add nodes with 0 conflicts
            zero_conflict = np.where((conflicts == 0) & (~in_set_mask))[0]
            if len(zero_conflict) > 0:
                # Add all zero-conflict nodes
                for node in zero_conflict:
                    current_set.append(node)
                    current_set_arr = np.array(current_set)
                    in_set_mask[node] = True
                    conflicts = incompat[:, current_set_arr].sum(axis=1)
                    in_set_mask = np.zeros(n_candidates, dtype=bool)
                    in_set_mask[current_set_arr] = True
                improved = True
                continue
            
            # Try swaps: remove one node, add two that become compatible
            # For each node in set, try removing it and adding the best candidate
            for remove_idx in range(len(current_set)):
                if time.time() > deadline - 5.0:
                    break
                
                removed_node = current_set[remove_idx]
                temp_set = current_set[:remove_idx] + current_set[remove_idx+1:]
                
                if len(temp_set) == 0:
                    continue
                    
                temp_set_arr = np.array(temp_set)
                temp_conflicts = incompat[:, temp_set_arr].sum(axis=1)
                
                # Find candidates with minimal conflicts
                temp_in_set = np.zeros(n_candidates, dtype=bool)
                temp_in_set[temp_set_arr] = True
                temp_conflicts[temp_in_set] = n_candidates  # Exclude already selected
                
                # Sort by conflict count
                sorted_candidates = np.argsort(temp_conflicts)
                
                added_count = 0
                for candidate in sorted_candidates[:20]:
                    if temp_conflicts[candidate] == 0:
                        temp_set.append(candidate)
                        temp_set_arr = np.array(temp_set)
                        temp_conflicts = incompat[:, temp_set_arr].sum(axis=1)
                        temp_in_set = np.zeros(n_candidates, dtype=bool)
                        temp_in_set[temp_set_arr] = True
                        temp_conflicts[temp_in_set] = n_candidates
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if len(temp_set) > len(current_set):
                    current_set = temp_set
                    current_set_arr = np.array(current_set)
                    improved = True
                    break

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)

    # Strategy 5: Simulated annealing style - accept worse solutions temporarily
    if time.time() < deadline - 8.0 and best_set:
        current_set = list(best_set)
        current_arr = np.array(current_set)
        current_size = len(current_set)
        
        rng2 = np.random.default_rng(999)
        
        for iteration in range(500):
            if time.time() > deadline - 5.0:
                break
            
            # Randomly pick a node to remove and a node to add
            remove_idx = rng2.integers(0, len(current_set))
            removed = current_set[remove_idx]
            
            # Find a candidate to add
            available = [i for i in range(n_candidates) if i not in set(current_set)]
            if not available:
                break
                
            # Weight by low conflict
            if len(current_set) > 1:
                cur_arr = np.array(current_set)
                cand_conflicts = incompat[np.array(available), cur_arr].sum(axis=1)
                w = 1.0 / (cand_conflicts + 1)
                w = w / w.sum()
                add_idx = rng2.choice(len(available), p=w)
                add_node = available[add_idx]
            else:
                add_node = available[rng2.integers(0, len(available))]
            
            new_set = current_set[:remove_idx] + current_set[remove_idx+1:] + [add_node]
            new_arr = np.array(new_set)
            
            # Check if valid
            if len(new_arr) > 1:
                new_dots = new_arr @ new_arr.T
                new_max_dot = np.max(new_dots) - new_dots[np.arange(len(new_arr)), np.arange(len(new_arr))]
                if new_max_dot <= 2:
                    current_set = new_set
                    current_arr = new_arr
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)

    # Final verification and fallback
    if not best_set:
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    result = candidates[list(best_set)]
    
    # Verify
    if len(result) > 1:
        dists = np.sqrt(np.sum((result[:, None, :] - result[None, :, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        max_norm = np.max(np.linalg.norm(result, axis=1))
        
        if min_dist < max_norm - 1e-9:
            vectors = []
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        vectors.append(v)
            return np.array(vectors, dtype=np.int64)
    
    return result
# EVOLVE-BLOCK-END
