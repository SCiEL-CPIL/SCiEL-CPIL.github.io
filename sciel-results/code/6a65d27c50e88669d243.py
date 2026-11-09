"""
Maximum Independent Set via priority scoring.
Strategy:
1. Fast static scoring: combine degree, neighbor-degree sum, coreness, and spectral gap features.
2. Multi-start local search on the vertex ordering (swap-based) to refine the greedy outcome.
3. Optional local search on the independent set itself for final polishing.
"""
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def _greedy_decode(adjacency, order):
    """
    Run the fixed greedy decoder given a specific vertex order (permutation).
    Returns: (count, selected_mask)
    """
    n = adjacency.shape[0]
    selected = np.zeros(n, dtype=bool)
    available = np.ones(n, dtype=bool)
    count = 0
    
    for v in order:
        if not available[v]:
            continue
        selected[v] = True
        count += 1
        available[adjacency[v]] = False
        
    return count, selected

def _evaluate_scores(adjacency, scores):
    """
    Evaluate a set of scores by sorting them and running the greedy decoder.
    Returns: count
    """
    order = np.argsort(-scores, kind='stable')
    count, _ = _greedy_decode(adjacency, order)
    return count

def _compute_core_numbers(adjacency):
    """
    Compute the core number for each vertex in the graph.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(int)
    
    deg = degrees.copy()
    core = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    
    for _ in range(n):
        if not np.any(remaining):
            break
        masked_degrees = np.where(remaining, deg, np.inf)
        v = np.argmin(masked_degrees)
        core[v] = deg[v]
        remaining[v] = False
        neighbors = np.where(adjacency[v])[0]
        active_neighbors = neighbors[remaining[neighbors]]
        if len(active_neighbors) > 0:
            deg[active_neighbors] -= 1
            
    return core

def _compute_spectral_features(adjacency):
    """
    Compute spectral features: use the Fiedler vector (eigenvector of the second smallest
    eigenvalue of the Laplacian) to identify a spectral cut. Vertices on one side of the cut
    are prioritized to minimize crossing edges.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)
    # Laplacian L = D - A
    L = np.diag(degrees) - adjacency.astype(float)
    
    try:
        # Convert to sparse for efficiency
        L_sparse = sp.csr_matrix(L)
        
        # Compute smallest eigenvalues using shift-invert mode
        k = min(3, n - 1)
        if k < 2:
            return np.zeros(n)
            
        eigvals, eigvecs = spla.eigsh(L_sparse, k=k, sigma=-1e-6, which='LM')
        
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Fiedler vector is the eigenvector for lambda_2 (index 1, since lambda_1 ≈ 0)
        if len(eigvals) >= 2:
            fiedler = eigvecs[:, 1]
        else:
            fiedler = eigvecs[:, -1]
            
        # Normalize Fiedler vector to have unit norm
        norm_f = np.linalg.norm(fiedler)
        if norm_f > 1e-10:
            fiedler = fiedler / norm_f
            
        return fiedler
    except Exception:
        # Fallback: return zeros
        return np.zeros(n)

def _baseline_scores(adjacency):
    """
    Compute a combined baseline score using degree, neighbor-degree sum, coreness, and spectral features.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)
    neighbor_deg_sum = adjacency @ degrees
    coreness = _compute_core_numbers(adjacency).astype(float)
    spectral = _compute_spectral_features(adjacency)
    
    # Normalize each feature
    def norm(x):
        return x / (x.max() + 1e-9)
        
    deg_norm = norm(degrees)
    nds_norm = norm(neighbor_deg_sum)
    core_norm = norm(coreness)
    spec_norm = norm(np.abs(spectral))
    
    # We want low degree, low neighbor-degree sum, low coreness, and low spectral value (one side of cut)
    # to have HIGH priority. So we negate them.
    # Weighted combination:
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 0.3
    scores = -(alpha * deg_norm + beta * nds_norm + gamma * core_norm + delta * spec_norm)
    
    return scores

def _local_search_order(adjacency, initial_order, deadline):
    """
    Improve the order via local search (swaps).
    """
    n = adjacency.shape[0]
    current_order = initial_order.copy()
    current_count, _ = _greedy_decode(adjacency, current_order)
    
    best_order = current_order.copy()
    best_count = current_count
    
    step = 0
    no_improve_count = 0
    max_no_improve = 200
    T = 1.0
    cooling_rate = 0.99
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
            
        # Choose a move: swap two random positions
        i, j = np.random.choice(n, 2, replace=False)
        
        # Swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        new_count, _ = _greedy_decode(adjacency, current_order)
        
        delta = new_count - current_count
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_count = new_count
            if new_count > best_count:
                best_count = new_count
                best_order = current_order.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            current_order[i], current_order[j] = current_order[j], current_order[i]
            no_improve_count += 1
            
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
        step += 1
        
    return best_order, best_count

def _local_search_set(adjacency, initial_set, deadline):
    """
    Local search on the independent set itself. Try to add vertices that don't conflict,
    and try swaps (remove one, add two) if possible.
    Optimized for speed: uses vectorized operations and strict time checking.
    """
    n = adjacency.shape[0]
    current_set = initial_set.copy()
    current_size = int(np.sum(current_set))
    
    best_set = current_set.copy()
    best_size = current_size
    
    def get_blocked_mask(s):
        if not np.any(s):
            return np.zeros(n, dtype=bool)
        neighbor_counts = adjacency @ s
        return s | (neighbor_counts > 0)
        
    blocked = get_blocked_mask(current_set)
    
    max_steps = 5000
    
    for step in range(max_steps):
        if time.time() >= deadline:
            break
            
        candidates = np.where(~blocked)[0]
        
        if len(candidates) == 0:
            # Try to remove a vertex to unblock others
            set_vertices = np.where(current_set)[0]
            if len(set_vertices) == 0:
                break
            v_to_remove = set_vertices[np.random.randint(len(set_vertices))]
            current_set[v_to_remove] = False
            current_size -= 1
            blocked = get_blocked_mask(current_set)
            continue
        
        # Try to add a random candidate
        v = candidates[np.random.randint(len(candidates))]
        current_set[v] = True
        current_size += 1
        blocked = get_blocked_mask(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set.copy()
            
        # Occasionally try a swap: remove a vertex, see if we can add two
        if np.random.random() < 0.1:
            set_vertices = np.where(current_set)[0]
            if len(set_vertices) > 1:
                v_remove = set_vertices[np.random.randint(len(set_vertices))]
                current_set[v_remove] = False
                current_size -= 1
                blocked = get_blocked_mask(current_set)
                candidates_after = np.where(~blocked)[0]
                if len(candidates_after) >= 2:
                    # Add two random candidates
                    v1, v2 = np.random.choice(candidates_after, 2, replace=False)
                    current_set[v1] = True
                    current_set[v2] = True
                    current_size += 2
                    blocked = get_blocked_mask(current_set)
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set.copy()
                        
    return best_set, best_size

def priority(adjacency):
    """
    Computes priority scores for vertices to maximize the size of the 
    independent set found by the greedy decoder.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin from 120s limit
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    # 1. Baseline scores
    try:
        best_scores = _baseline_scores(adjacency)
    except Exception:
        # Fallback to simple degree-based scoring if spectral fails or is too slow
        degrees = adjacency.sum(axis=1).astype(float)
        best_scores = -degrees / (degrees.max() + 1e-9)
        
    best_count = _evaluate_scores(adjacency, best_scores)
    
    # 2. Local search on the vertex ordering
    if time.time() < deadline - 10.0:
        initial_order = np.argsort(-best_scores, kind='stable')
        # Allocate half the remaining time to order search
        order_deadline = min(deadline - 5.0, start_time + 40.0)
        refined_order, refined_count = _local_search_order(adjacency, initial_order, order_deadline)
        
        if refined_count > best_count:
            best_count = refined_count
            # Reconstruct scores from the order?
            # Actually, the greedy decoder uses scores. We need to return scores.
            # The order is a permutation. We can assign scores based on the rank in this order.
            # But the greedy decoder sorts by score.
            # If we assign score = -rank, the greedy decoder will pick in the order of the permutation.
            # Let's create a score vector where the vertex at index i in refined_order gets a score such that it is picked i-th.
            # Simpler: just return the order as scores? No, scores are continuous.
            # Let's assign scores such that argsort(-scores) == refined_order.
            # score[v] = n - index_of_v_in_refined_order
            rank = np.zeros(n, dtype=float)
            for i, v in enumerate(refined_order):
                rank[v] = n - i
            best_scores = rank
            
    # 3. Local search on the independent set
    if time.time() < deadline - 2.0:
        order = np.argsort(-best_scores, kind='stable')
        _, init_set = _greedy_decode(adjacency, order)
        
        # Limit local search time to remaining time minus safety margin
        ls_deadline = deadline - 1.0
        ls_set, ls_count = _local_search_set(adjacency, init_set, ls_deadline)
        
        if ls_count > best_count:
            # Convert set to scores: vertices in the set get high scores
            # We need to return scores such that the greedy decoder picks this set.
            # If we set score=1 for set members and 0 for others, the greedy decoder
            # will pick all set members first (since they are independent), then
            # pick other non-conflicting vertices if any remain.
            # This might result in a set LARGER than ls_set if ls_set was not maximal.
            # But the score is based on the *final* greedy output.
            # If the final output is larger, that's good!
            best_scores = np.zeros(n)
            best_scores[ls_set] = 1.0
            best_count = ls_count # Note: best_count might be lower than actual greedy result
            
    return best_scores
