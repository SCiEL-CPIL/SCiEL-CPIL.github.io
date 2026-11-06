"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_size(adjacency, order):
    """Compute the size of the independent set produced by greedy decoding
    with the given vertex order (indices in descending priority)."""
    n = len(order)
    selected = np.zeros(n, dtype=bool)
    for v in order:
        if not selected[v]:
            selected[v] = True
            # Mark all neighbors as blocked
            if adjacency[v].any():
                selected[adjacency[v]] = True
    return selected.sum()


def _evaluate_order(adjacency, scores):
    """Evaluate the greedy independent set size for given scores."""
    order = np.argsort(-scores)
    return _greedy_size(adjacency, order)


def _spectral_priority(adjacency):
    """
    Use power iteration on the adjacency matrix to get the principal eigenvector.
    Vertices with smaller principal eigenvector values are in sparser regions
    and should be prioritized.
    """
    n = adjacency.shape[0]
    # Use a few power iterations to approximate the principal eigenvector
    x = np.ones(n) / n
    A = adjacency.astype(float)
    for _ in range(50):
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            break
        x_new /= norm
        if np.allclose(x, x_new, atol=1e-10):
            break
        x = x_new
    # Lower values in the principal eigenvector indicate sparser regions
    return -x


def _laplacian_fiedler(adjacency):
    """
    Compute the Fiedler vector (second smallest eigenvalue eigenvector) of the
    graph Laplacian. This provides a partitioning signal.
    """
    n = adjacency.shape[0]
    A = adjacency.astype(float)
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Power iteration on L for the second smallest eigenvalue is tricky.
    # Instead, use inverse iteration or just use a simpler approach.
    # For now, let's use the fact that the Fiedler vector is orthogonal to the
    # constant vector. We can use a few steps of inverse power iteration.
    
    # Actually, let's use a different approach: compute the eigenvector
    # corresponding to the second smallest eigenvalue using a modified
    # power iteration that deflates the constant vector.
    
    # Simpler: just use the degree-based approach combined with spectral
    # Let's try a different spectral approach: use the eigenvector of A
    # but with a shift to get interior eigenvalues.
    
    # For simplicity and speed, let's just use the principal eigenvector
    # and combine with degree.
    return None


def _marginal_loss_scores(adjacency, base_scores):
    """
    Refine scores by considering the marginal loss of selecting each vertex.
    A vertex that blocks many high-priority neighbors should be penalized.
    """
    n = adjacency.shape[0]
    A = adjacency.astype(float)
    scores = base_scores.copy()
    
    # Compute the sum of scores of neighbors for each vertex
    # This represents the "cost" of blocking those neighbors
    neighbor_score_sum = A @ scores
    
    # Adjust scores: reduce score for vertices that block high-scoring neighbors
    # But also consider degree: low degree vertices are generally better
    degree = adjacency.sum(axis=1).astype(float)
    
    # Combine: base spectral score minus a penalty for blocking high-scoring neighbors
    # Normalize the penalty
    max_penalty = neighbor_score_sum.max() if neighbor_score_sum.max() > 0 else 1.0
    penalty = neighbor_score_sum / max_penalty
    
    # Final score: spectral score with a degree-aware penalty
    refined = scores - 0.3 * penalty + 0.1 * (-degree)
    
    return refined


def _local_search(adjacency, scores, max_iters=100, time_limit=None, start_time=None):
    """
    Perform local search on the vertex ordering to improve the independent set size.
    Uses 2-opt style swaps and insertions.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 150
    
    n = adjacency.shape[0]
    order = np.argsort(-scores)
    best_size = _greedy_size(adjacency, order)
    best_order = order.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit - 5:
            break
        
        improved = False
        
        # Try random swaps
        for _ in range(50):
            if time.time() - start_time > time_limit - 5:
                break
            
            i, j = np.random.randint(0, n, 2)
            if i == j:
                continue
            
            new_order = best_order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
            size = _greedy_size(adjacency, new_order)
            
            if size > best_size:
                best_size = size
                best_order = new_order
                improved = True
        
        # Try random insertions (move a vertex to a random position)
        for _ in range(50):
            if time.time() - start_time > time_limit - 5:
                break
            
            i, j = np.random.randint(0, n, 2)
            if i == j:
                continue
            
            new_order = list(best_order)
            v = new_order.pop(i)
            new_order.insert(j, v)
            new_order = np.array(new_order)
            size = _greedy_size(adjacency, new_order)
            
            if size > best_size:
                best_size = size
                best_order = new_order
                improved = True
        
        if not improved:
            # Perturb: try a larger random restart
            if np.random.random() < 0.1:
                # Random shuffle of a subset
                k = np.random.randint(n // 2, n)
                idx = np.random.choice(n, k, replace=False)
                best_order[idx] = np.random.permutation(best_order[idx])
    
    return best_order, best_size


def priority(adjacency):
    """
    Assign a score to each vertex such that a greedy decoder picking vertices
    in descending score order (skipping neighbors of already-picked vertices)
    yields a large independent set.
    """
    start_time = time.time()
    time_limit = 150
    n = adjacency.shape[0]
    
    # Step 1: Get initial spectral-based scores
    spectral_scores = _spectral_priority(adjacency)
    
    # Step 2: Combine with degree-based scores
    degree = adjacency.sum(axis=1).astype(float)
    max_degree = degree.max() if degree.max() > 0 else 1.0
    degree_scores = -degree / max_degree
    
    # Combine spectral and degree scores
    base_scores = 0.6 * spectral_scores + 0.4 * degree_scores
    
    # Step 3: Refine with marginal loss
    refined_scores = _marginal_loss_scores(adjacency, base_scores)
    
    # Step 4: Local search to improve the ordering
    if time.time() - start_time < time_limit - 10:
        best_order, best_size = _local_search(
            adjacency, refined_scores,
            max_iters=200,
            time_limit=time_limit,
            start_time=start_time
        )
        
        # Convert the best order back to scores
        scores = np.zeros(n)
        for rank, v in enumerate(best_order):
            scores[v] = n - rank
    else:
        scores = refined_scores
    
    # Safety check: return before timeout
    if time.time() - start_time > time_limit - 2:
        return refined_scores
    
    return scores
# EVOLVE-BLOCK-END
