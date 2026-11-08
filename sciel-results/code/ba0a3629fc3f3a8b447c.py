"""Kissing number in dimension 11 - weight-4 vectors with greedy MIS + local search."""
import numpy as np
import itertools
import time

D = 11
DEADLINE = 140.0  # safety margin before 150s kill


def _generate_weight4_vectors():
    """Generate all nonzero integer vectors of weight 4 (exactly 4 non-zero entries of ±1)."""
    vectors = []
    # Choose 4 positions out of 11
    for positions in itertools.combinations(range(D), 4):
        # 2^4 = 16 sign patterns
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            vectors.append(v)
    return np.array(vectors)  # shape (13312, 11)


def _compute_conflict_matrix(A):
    """
    For weight-4 vectors with norm²=4, two vectors a,b are compatible iff
    ||a-b||² >= 4, i.e., 8 - 2*a·b >= 4, i.e., a·b <= 2.
    Conflict (incompatible) iff a·b > 2, i.e., a·b >= 3.
    For weight-4 ±1 vectors, a·b can be -4,-2,0,2,4.
    So conflict iff a·b == 4 (identical) or... wait, a·b=4 means same vector.
    Actually for distinct vectors, a·b can be -4,-2,0,2.
    a·b=2 means they share 3 positions with same sign and 1 different, or 2 same 2 diff... 
    Let me recompute: a·b = sum over 4 positions of a_i*b_i.
    If they share k positions with same sign and (4-k) with different sign (among shared positions),
    but they might not share all 4 positions.
    
    Actually, let's just compute dot products directly.
    Conflict iff a·b > 2. For weight-4 ±1 vectors, possible dot products:
    - If disjoint supports: 0
    - If share 1 position: ±1
    - If share 2 positions: -2, 0, 2
    - If share 3 positions: -3, -1, 1, 3
    - If share 4 positions: -4, -2, 0, 2, 4
    
    So conflict (a·b > 2) only when a·b = 3 or 4.
    a·b=4: identical vectors (same support, same signs)
    a·b=3: share 3 positions with same sign, 1 position with different sign
           (supports overlap in 3, and on those 3 they agree, on the 4th they differ)
    """
    n = len(A)
    # Compute all pairwise dot products
    dots = A @ A.T  # (n, n)
    # Conflict if dot > 2
    conflict = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    return conflict


def _greedy_mis(conflict, rng):
    """Greedy maximum independent set: repeatedly pick node with fewest conflicts among remaining."""
    n = len(conflict)
    remaining = set(range(n))
    selected = []
    
    while remaining:
        # Find node in remaining with minimum degree within remaining
        best_node = None
        best_degree = float('inf')
        for node in remaining:
            deg = conflict[node].sum()  # total conflicts, but we need within remaining
            # More accurate: count conflicts within remaining
            deg_in_rem = 0
            for other in remaining:
                if other != node and conflict[node, other]:
                    deg_in_rem += 1
            if deg_in_rem < best_degree:
                best_degree = deg_in_rem
                best_node = node
        
        selected.append(best_node)
        remaining.discard(best_node)
        # Remove all neighbors of best_node from remaining
        neighbors = np.where(conflict[best_node] == 1)[0]
        remaining -= set(neighbors.tolist())
    
    return selected


def _greedy_mis_fast(conflict, rng):
    """Faster greedy MIS using numpy."""
    n = len(conflict)
    remaining = np.ones(n, dtype=bool)
    selected = []
    
    while remaining.any():
        # Count conflicts within remaining for each remaining node
        rem_indices = np.where(remaining)[0]
        if len(rem_indices) == 0:
            break
        sub_conflict = conflict[np.ix_(rem_indices, rem_indices)]
        degrees = sub_conflict.sum(axis=1)
        best_idx = np.argmin(degrees)
        best_node = rem_indices[best_idx]
        
        selected.append(best_node)
        remaining[best_node] = False
        # Remove neighbors
        neighbors = np.where(conflict[best_node] == 1)[0]
        remaining[neighbors] = False
    
    return selected


def _local_improve(selected, conflict, max_iters=100):
    """Try to add more nodes or swap to improve the independent set."""
    selected_set = set(selected)
    n = len(conflict)
    improved = True
    iterations = 0
    
    while improved and iterations < max_iters:
        improved = False
        iterations += 1
        
        # Try adding any unselected node that has no conflict with selected
        unselected = [i for i in range(n) if i not in selected_set]
        for node in unselected:
            if not conflict[node, list(selected_set)].any():
                selected_set.add(node)
                improved = True
                break
        
        # Try swapping: remove one selected, add two unselected
        if not improved:
            unselected = [i for i in range(n) if i not in selected_set]
            for sel in list(selected_set):
                # Remove sel temporarily
                new_set = selected_set - {sel}
                # Find unselected nodes compatible with new_set
                candidates = [u for u in unselected if not conflict[u, list(new_set)].any()]
                if len(candidates) >= 2:
                    # Try to find two candidates that are also compatible with each other
                    for i in range(len(candidates)):
                        for j in range(i+1, len(candidates)):
                            if not conflict[candidates[i], candidates[j]]:
                                selected_set = new_set | {candidates[i], candidates[j]}
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
    
    return list(selected_set)


def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Generate all weight-4 vectors
    A = _generate_weight4_vectors()
    n = len(A)
    
    # Compute conflict matrix
    conflict = _compute_conflict_matrix(A)
    
    # Greedy MIS
    selected = _greedy_mis_fast(conflict, rng)
    
    # Local improvement
    selected = _local_improve(selected, conflict, max_iters=50)
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 5:
        return A[selected].copy()
    
    # Try multiple random restarts of greedy with different tie-breaking
    best_selected = selected
    best_size = len(selected)
    
    while time.time() - start_time < DEADLINE - 10:
        # Randomized greedy: shuffle tie-breaking
        remaining = np.ones(n, dtype=bool)
        selected_r = []
        
        while remaining.any():
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            sub_conflict = conflict[np.ix_(rem_indices, rem_indices)]
            degrees = sub_conflict.sum(axis=1)
            min_deg = degrees.min()
            min_nodes = rem_indices[degrees == min_deg]
            if len(min_nodes) > 1:
                best_node = min_nodes[rng.randint(len(min_nodes))]
            else:
                best_node = min_nodes[0]
            
            selected_r.append(best_node)
            remaining[best_node] = False
            neighbors = np.where(conflict[best_node] == 1)[0]
            remaining[neighbors] = False
        
        # Local improve
        selected_r = _local_improve(selected_r, conflict, max_iters=30)
        
        if len(selected_r) > best_size:
            best_size = len(selected_r)
            best_selected = selected_r
        
        if time.time() - start_time > DEADLINE - 5:
            break
    
    return A[best_selected].copy()


if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} vectors")
    # Verify
    if len(result) > 1:
        dots = result @ result.T
        norms_sq = np.diag(dots)
        off_diag = dots - np.diag(np.diag(dots))
        min_dist_sq = (norms_sq[:, None] + norms_sq[None, :] - 2 * dots).min()
        max_norm_sq = norms_sq.max()
        print(f"Min pairwise dist²: {min_dist_sq}, Max norm²: {max_norm_sq}")
        print(f"Valid: {min_dist_sq >= max_norm_sq}")
