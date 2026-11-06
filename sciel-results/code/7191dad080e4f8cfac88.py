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


def _greedy_size_fast(adjacency_bool, order):
    """Faster evaluation using precomputed neighbor lists."""
    n = len(order)
    available = [True] * n
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            for u in adjacency_bool[v]:
                available[u] = False
    return count


def _spectral_priority(A, n):
    """
    Generate initial priority scores using spectral methods on the graph.
    """
    try:
        eigenvalues, eigenvectors = np.linalg.eig(A)
        idx = np.argsort(-np.abs(eigenvalues))
        principal = np.abs(eigenvectors[:, idx[0]])
    except:
        principal = np.ones(n) / n
    
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
    """
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    
    try:
        eigenvalues, eigenvectors = np.linalg.eig(L)
        idx = np.argsort(eigenvalues.real)
        fiedler = np.abs(eigenvectors[:, idx[1]])
    except:
        fiedler = np.ones(n) / n
    
    f_min = fiedler.min()
    f_max = fiedler.max()
    if f_max > f_min:
        fiedler_norm = (fiedler - f_min) / (f_max - f_min)
    else:
        fiedler_norm = np.zeros(n)
    
    return fiedler_norm


def _dynamic_scores(A, deg, spectral, fiedler, W1, W2, b1, b2, n, temperature=1.0):
    """
    Compute dynamic scores using a two-layer 'attention-like' network.
    Input features: [degree, spectral, fiedler, 1]
    Output: score per vertex
    """
    # Build feature matrix: n x 4
    X = np.column_stack([deg / max(deg.max(), 1), spectral, fiedler, np.ones(n)])
    
    # First layer: ReLU activation
    Z1 = X @ W1 + b1  # n x hidden
    H1 = np.maximum(Z1, 0)
    
    # Second layer: linear output
    scores = H1 @ W2 + b2  # n x 1
    return scores.flatten()


def _straight_through_train(A, deg, spectral, fiedler, n, hidden_dim=8, 
                              max_epochs=50, lr=0.01, rng=None):
    """
    Train the dynamic scoring network using straight-through estimation.
    The loss is negative of the greedy independent set size.
    We use a soft surrogate: at each step, we compute scores, pick the 
    highest-scoring available vertex, and track which vertices were selected.
    
    For straight-through: we use the argmax to select, but backprop through
    a softmax approximation of the selection probability.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Initialize weights
    W1 = rng.normal(0, 0.1, size=(4, hidden_dim))
    b1 = np.zeros(hidden_dim)
    W2 = rng.normal(0, 0.1, size=(hidden_dim, 1))
    b2 = np.zeros(1)
    
    # Precompute features
    X = np.column_stack([deg / max(deg.max(), 1), spectral, fiedler, np.ones(n)])
    
    # Neighbor lists for fast evaluation
    neighbor_lists = [np.where(A[v])[0] for v in range(n)]
    
    best_size = -1
    best_params = (W1.copy(), W2.copy(), b1.copy(), b2.copy())
    
    for epoch in range(max_epochs):
        # Compute current scores
        Z1 = X @ W1 + b1
        H1 = np.maximum(Z1, 0)
        scores = (H1 @ W2 + b2).flatten()
        
        # Greedy selection with straight-through
        available = np.ones(n, dtype=bool)
        selected = []
        for step in range(n):
            if not available.any():
                break
            # Mask unavailable vertices
            masked_scores = np.where(available, scores, -np.inf)
            v = np.argmax(masked_scores)
            selected.append(v)
            available[v] = False
            for u in neighbor_lists[v]:
                available[u] = False
        
        # Compute size
        size = len(selected)
        
        if size > best_size:
            best_size = size
            best_params = (W1.copy(), W2.copy(), b1.copy(), b2.copy())
        
        # Straight-through gradient: 
        # We want to increase scores of selected vertices and decrease others.
        # Use a simple policy gradient approach:
        # For each selected vertex v at step t, its score should have been high.
        # Approximate gradient: dLoss/dscores[v] += 1 for selected, -1 for not selected
        # But we need to account for the sequential nature.
        
        # Simpler approach: treat this as a ranking problem.
        # The greedy size depends on which vertices are selected.
        # We can compute the marginal contribution of each vertex.
        
        # Alternative: use a softmax-based soft selection
        # P(v selected) ∝ exp(score[v] / temperature) among available vertices
        
        # For now, use a simple heuristic gradient:
        # Increase scores of vertices that were selected, decrease others
        grad_scores = np.zeros(n)
        for v in selected:
            grad_scores[v] = 1.0
        # Normalize
        grad_scores = grad_scores - grad_scores.mean()
        
        # Backprop through network
        # dLoss/dscores = grad_scores
        dW2 = H1.T @ grad_scores.reshape(-1, 1)
        db2 = np.sum(grad_scores).reshape(1)
        
        dH1 = grad_scores.reshape(-1, 1) @ W2.T
        dZ1 = dH1 * (Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)
        
        # Update weights
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    
    return best_params, best_size


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = adjacency.shape[0]
    A = adjacency.astype(bool)
    
    # Precompute neighbor lists for faster evaluation
    neighbor_lists = [np.where(A[v])[0].tolist() for v in range(n)]
    
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
    
    # Heuristic 8: Degree squared (penalize high-degree vertices more)
    score_h8 = deg ** 2
    order8 = np.argsort(score_h8, kind='stable')
    
    # Heuristic 9: Degree + clustering (higher clustering = worse for MIS)
    score_h9 = deg + 0.3 * C
    order9 = np.argsort(score_h9, kind='stable')
    
    # Heuristic 10: Inverse degree (prefer low-degree vertices strongly)
    score_h10 = 1.0 / (deg + 1.0)
    order10 = np.argsort(-score_h10, kind='stable')
    
    candidates = [order1, order2, order3, order4, order5, order6, order7, order8, order9, order10]
    
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
    
    # Try dynamic scoring with straight-through training
    if time.time() < deadline - 30:
        try:
            hidden_dim = 8
            max_epochs = 30
            lr = 0.01
            
            W1, W2, b1, b2 = _straight_through_train(
                A, deg, spectral, fiedler, n,
                hidden_dim=hidden_dim,
                max_epochs=max_epochs,
                lr=lr,
                rng=rng
            )
            
            # Get final scores
            final_scores = _dynamic_scores(A, deg, spectral, fiedler, W1, W2, b1, b2, n)
            dynamic_order = np.argsort(-final_scores, kind='stable')
            dynamic_size = _evaluate_order(A, dynamic_order)
            
            if dynamic_size > best_size:
                best_size = dynamic_size
                best_order = dynamic_order.copy()
        except Exception:
            pass
    
    # Local Search: Multi-start Hill Climbing with improved moves
    num_restarts = 30
    max_iters_per_restart = 2000  # 30 * 2000 = 60,000 evals
    
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        # Initialize current order for this restart
        if r == 0:
            current_order = best_order.copy()
        elif r % 4 == 1:
            # Perturb the best order found so far
            current_order = best_order.copy()
            for _ in range(8):
                i, j = rng.integers(0, n, size=2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        elif r % 4 == 2:
            # Start from a random heuristic
            current_order = candidates[r % len(candidates)].copy()
            for _ in range(5):
                i, j = rng.integers(0, n, size=2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        elif r % 4 == 3:
            # Start from a new random order
            current_order = rng.permutation(n)
        else:
            # Start from best order with larger perturbation
            current_order = best_order.copy()
            for _ in range(20):
                i, j = rng.integers(0, n, size=2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
                
        current_size = _evaluate_order(A, current_order)
        
        # If this restart starts better than global best, update
        if current_size > best_size:
            best_size = current_size
            best_order = current_order.copy()
            
        no_improve_count = 0
        max_no_improve = 60
        
        for i in range(max_iters_per_restart):
            if time.time() > deadline - 10:
                break
                
            if no_improve_count > max_no_improve:
                break
                
            # Choose move type: adjacent swap, random swap, or random insertion
            move_type = rng.integers(0, 3)
            
            if move_type == 0:
                # Adjacent swap: swap two neighboring positions
                v_idx = rng.integers(0, n - 1)
                new_order = current_order.copy()
                new_order[v_idx], new_order[v_idx + 1] = new_order[v_idx + 1], new_order[v_idx]
            elif move_type == 1:
                # Random swap: swap two random positions
                v_idx, target_pos = rng.integers(0, n, size=2)
                if v_idx == target_pos:
                    continue
                new_order = current_order.copy()
                new_order[v_idx], new_order[target_pos] = new_order[target_pos], new_order[v_idx]
            else:
                # Random insertion: remove a vertex and insert at random position
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
