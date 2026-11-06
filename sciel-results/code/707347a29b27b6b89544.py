"""Improved program for Maximum Independent Set greedy scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_size(adjacency_bool, order):
    """Compute independent set size given a vertex order (indices sorted by priority)."""
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            available[adjacency_bool[v]] = False
    return count


def _evaluate_order(adjacency_bool, order):
    """Evaluate a priority vector by running the greedy decoder."""
    return _greedy_size(adjacency_bool, order)


def _spectral_priority(A, n):
    """
    Generate initial priority scores using spectral methods on the graph.
    We use the adjacency matrix to compute eigenvector-based scores.
    For MIS, we want vertices that are "central" in some sense but also
    have low degree. We can use the second eigenvector or a combination.
    
    Actually, for maximum independent set, a good heuristic is related to
    the eigenvector of the adjacency matrix. Vertices with high values in
    the principal eigenvector tend to be in dense regions. For MIS, we
    want to pick vertices from sparse regions, so we might want to use
    a different approach.
    
    Let's use a combination:
    - degree (lower is better for MIS greedy)
    - spectral score from the Laplacian or adjacency
    
    For ER graphs, the principal eigenvector of A is roughly uniform.
    Let's try using the Fiedler vector (second eigenvector) which captures
    community structure, but for ER graphs this is less clear.
    
    Alternative: use the eigenvector of (D - A) which is the Laplacian.
    The null space of L gives connected components. For a connected graph,
    the second eigenvector gives a partition.
    
    Let's try a simpler spectral approach: compute the eigenvector
    corresponding to the smallest non-zero eigenvalue of the Laplacian,
    and use that as a score. Vertices with extreme values might be good
    candidates for MIS.
    
    Actually, let's think differently. The greedy algorithm picks vertices
    in order. A good ordering for MIS is one that picks vertices that
    "cover" many edges early (so their neighbors are removed) but also
    are not too connected to each other.
    
    For spectral methods, we can use the fact that the eigenvector of the
    adjacency matrix gives a "centrality" measure. For MIS, we want to
    pick vertices that are peripheral (low centrality) but have high
    "independence" (low degree).
    
    Let's compute the principal eigenvector of A and use it as a score.
    We'll combine it with degree.
    """
    # Compute eigenvalues and eigenvectors of the adjacency matrix
    # For n=200, this is fast.
    try:
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # Sort by magnitude (real part for symmetric matrix)
        idx = np.argsort(-np.abs(eigenvalues))
        # Principal eigenvector
        principal = np.abs(eigenvectors[:, idx[0]])
    except:
        principal = np.ones(n) / n
    
    # Normalize principal to [0, 1]
    p_min = principal.min()
    p_max = principal.max()
    if p_max > p_min:
        principal_norm = (principal - p_min) / (p_max - p_min)
    else:
        principal_norm = np.zeros(n)
    
    return principal_norm


def _laplacian_fiedler(A, n):
    """
    Compute the Fiedler vector (second eigenvector of the Laplacian).
    This captures the main partition of the graph.
    """
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    
    try:
        eigenvalues, eigenvectors = np.linalg.eig(L)
        # Sort by eigenvalue (real part)
        idx = np.argsort(eigenvalues.real)
        # Second eigenvector (Fiedler vector)
        fiedler = np.abs(eigenvectors[:, idx[1]])
    except:
        fiedler = np.ones(n) / n
    
    # Normalize to [0, 1]
    f_min = fiedler.min()
    f_max = fiedler.max()
    if f_max > f_min:
        fiedler_norm = (fiedler - f_min) / (f_max - f_min)
    else:
        fiedler_norm = np.zeros(n)
    
    return fiedler_norm


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    A = adjacency.astype(bool)
    
    # Precompute degrees
    deg = A.sum(axis=1).astype(float)
    
    # Compute neighborhood edge counts for each vertex (clustering proxy)
    C = np.zeros(n)
    for v in range(n):
        if deg[v] < 2:
            C[v] = 0.0
        else:
            nbrs = np.where(A[v])[0]
            if len(nbrs) > 0:
                sub = A[np.ix_(nbrs, nbrs)]
                C[v] = sub.sum() / 2.0
    
    # Spectral scores
    spectral = _spectral_priority(A, n)
    fiedler = _laplacian_fiedler(A, n)
    
    # Heuristic 1: Ascending Degree
    order1 = np.argsort(deg, kind='stable')
    
    # Heuristic 2: Ascending Degree, then Ascending Clustering
    score_h2 = deg + 0.1 * C
    order2 = np.argsort(score_h2, kind='stable')
    
    # Heuristic 3: Random
    rng = np.random.default_rng(42)
    order3 = rng.permutation(n)
    
    # Heuristic 4: Degree + Noise
    noise = rng.normal(0, 0.5, size=n)
    score_h4 = deg + noise
    order4 = np.argsort(score_h4, kind='stable')
    
    # Heuristic 5: Spectral-based - low spectral score first (peripheral vertices)
    score_h5 = deg * 0.5 + spectral * 0.5
    order5 = np.argsort(score_h5, kind='stable')
    
    # Heuristic 6: Fiedler-based - vertices with extreme Fiedler values
    score_h6 = deg * 0.5 + fiedler * 0.5
    order6 = np.argsort(score_h6, kind='stable')
    
    # Heuristic 7: Combined - degree, clustering, spectral
    score_h7 = deg * 0.4 + C * 0.1 + spectral * 0.3 + fiedler * 0.2
    order7 = np.argsort(score_h7, kind='stable')
    
    candidates = [order1, order2, order3, order4, order5, order6, order7]
    
    best_size = -1
    best_order = None
    
    # Evaluate initial candidates
    for order in candidates:
        if time.time() > deadline:
            break
        size = _evaluate_order(A, order)
        if size > best_size:
            best_size = size
            best_order = order.copy()
            
    # Local Search: Multi-start Hill Climbing
    num_restarts = 25
    max_iters_per_restart = 1500  # 25 * 1500 = 37,500 evals
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize current order for this restart
        if r == 0:
            current_order = best_order.copy()
        elif r % 3 == 1:
            # Perturb the best order found so far
            current_order = best_order.copy()
            for _ in range(10):
                i, j = rng.integers(0, n, size=2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        elif r % 3 == 2:
            # Start from a random heuristic
            current_order = candidates[r % len(candidates)].copy()
            for _ in range(5):
                i, j = rng.integers(0, n, size=2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        else:
            # Start from a new random order
            current_order = rng.permutation(n)
                
        current_size = _evaluate_order(A, current_order)
        
        # If this restart starts better than global best, update
        if current_size > best_size:
            best_size = current_size
            best_order = current_order.copy()
            
        no_improve_count = 0
        max_no_improve = 80
        
        for i in range(max_iters_per_restart):
            if time.time() > deadline - 5:
                break
                
            if no_improve_count > max_no_improve:
                break
                
            # Move: Pick a random vertex and move it to a random position
            v_idx = rng.integers(0, n)
            target_pos = rng.integers(0, n)
            
            if v_idx == target_pos:
                continue
                
            v = current_order[v_idx]
            
            new_order = current_order.copy()
            new_order = np.delete(new_order, v_idx)
            if v_idx < target_pos:
                insert_idx = target_pos - 1
            else:
                insert_idx = target_pos
            new_order = np.insert(new_order, insert_idx, v)
            
            new_size = _evaluate_order(A, new_order)
            
            if new_size > current_size:
                current_order = new_order
                current_size = new_size
                no_improve_count = 0
                if current_size > best_size:
                    best_size = current_size
                    best_order = current_order.copy()
            else:
                no_improve_count += 1
                
    # Convert best_order to priority vector
    # Higher priority means selected earlier.
    # So vertex at index 0 in order gets highest priority.
    # priority[v] = n - index(v)
    priority_vec = np.zeros(n)
    for idx, v in enumerate(best_order):
        priority_vec[v] = n - idx
        
    return priority_vec.astype(float)
# EVOLVE-BLOCK-END
