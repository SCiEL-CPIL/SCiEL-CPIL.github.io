"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assign a priority score to each vertex for a greedy Maximum Independent Set decoder.
    
    Strategy: Direct combinatorial optimization of the selection order via
    simulated annealing on the discrete permutation, plus spectral initialization.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    adj_bool = adj.astype(bool)
    degree = adj_bool.sum(axis=1).astype(float)
    
    # Fast greedy simulation: given a priority array, select vertices in descending order
    def simulate_greedy(scores):
        selected = 0
        available = np.ones(n, dtype=bool)
        order = np.argsort(-scores, kind='stable')
        
        for v in order:
            if not available[v]:
                continue
            selected += 1
            available[v] = False
            nbrs = adj_bool[v]
            if nbrs.any():
                available[nbrs] = False
            if not available.any():
                break
        return selected
    
    # Spectral initialization: use eigenvector of adjacency matrix
    # Low eigenvalues correspond to smooth functions on the graph
    try:
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        A = csr_matrix(adj_bool)
        # Get smallest algebraic eigenvalue and its eigenvector
        k = min(3, n - 1)
        if k > 0:
            vals, vecs = eigsh(A, k=k, which='SA')
            # Use the eigenvector corresponding to smallest eigenvalue
            spectral_score = vecs[:, 0]
        else:
            spectral_score = np.zeros(n)
    except:
        spectral_score = np.zeros(n)
    
    # Normalize spectral score
    if spectral_score.std() > 1e-10:
        spectral_score = (spectral_score - spectral_score.mean()) / spectral_score.std()
    else:
        spectral_score = np.zeros(n)
    
    # Composite initial score
    base_score = -degree + 0.1 * (adj_bool @ degree) + spectral_score
    
    # Get initial greedy solution size
    current_scores = base_score + np.random.normal(0, 0.5, n)
    current_size = simulate_greedy(current_scores)
    best_scores = current_scores.copy()
    best_size = current_size
    
    # Simulated annealing on the discrete order
    # We maintain a permutation (order) and swap elements
    # Temperature schedule: geometric cooling
    T_start = 2.0
    T_end = 0.01
    n_iterations = 0
    
    # Get initial order
    order = np.argsort(-current_scores, kind='stable')
    
    # Precompute for fast evaluation: we need to re-simulate when order changes
    # For efficiency, we'll track which vertices are selected
    
    def simulate_from_order(ord_arr):
        selected = 0
        available = np.ones(n, dtype=bool)
        for v in ord_arr:
            if not available[v]:
                continue
            selected += 1
            available[v] = False
            nbrs = adj_bool[v]
            if nbrs.any():
                available[nbrs] = False
            if not available.any():
                break
        return selected
    
    # Annealing loop
    T = T_start
    cooling_rate = (T_end / T_start) ** (1.0 / max(1, int(50000)))
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick two random positions to swap
        i, j = np.random.randint(0, n, 2)
        if i == j:
            continue
        i, j = int(i), int(j)
        
        # Swap
        order[i], order[j] = order[j], order[i]
        
        new_size = simulate_from_order(order)
        delta = new_size - current_size
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                # Reconstruct scores from best order
                best_scores = np.zeros(n)
                for rank, v in enumerate(order):
                    best_scores[v] = n - rank
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Revert
            order[i], order[j] = order[j], order[i]
            no_improve += 1
        
        T = max(T_end, T * cooling_rate)
        n_iterations += 1
        
        # Periodically try larger moves
        if n_iterations % 1000 == 0:
            # Try a random restart with new heuristic
            trial_scores = -degree + 0.1 * (adj_bool @ degree) + np.random.normal(0, 1, n)
            trial_size = simulate_greedy(trial_scores)
            if trial_size > best_size:
                best_size = trial_size
                best_scores = trial_scores.copy()
                order = np.argsort(-trial_scores, kind='stable')
                current_size = trial_size
                no_improve = 0
    
    # Final: also try some simple heuristics
    simple_heuristics = [
        -degree + np.random.normal(0, 0.5, n),
        -np.sqrt(degree + 1) + np.random.normal(0, 0.5, n),
        -np.log1p(degree) + np.random.normal(0, 0.5, n),
        np.random.permutation(n).astype(float),
    ]
    
    for h in simple_heuristics:
        if time.time() > deadline:
            break
        trial_size = simulate_greedy(h)
        if trial_size > best_size:
            best_size = trial_size
            best_scores = h.copy()
    
    return best_scores
# EVOLVE-BLOCK-END
