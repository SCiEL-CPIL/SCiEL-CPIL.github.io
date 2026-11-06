"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assign a priority score to each vertex for a greedy Maximum Independent Set decoder.
    
    Strategy: Multi-stage approach with structural heuristics and local search.
    1. Compute rich structural features (degree, neighbor degree, clustering proxy).
    2. Try multiple heuristic score combinations.
    3. Local search: swap pairs to improve greedy outcome.
    4. Final polish with targeted perturbations.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    adj_bool = adj.astype(bool)
    degree = adj_bool.sum(axis=1).astype(float)
    
    # Precompute structural features
    # Mean neighbor degree (weighted by connectivity)
    mean_nbr_deg = np.zeros(n, dtype=float)
    for i in range(n):
        nbrs = adj_bool[i]
        if degree[i] > 0:
            mean_nbr_deg[i] = (adj_bool[i] @ degree) / degree[i]
    
    # Clustering proxy: fraction of edges among neighbors (approximate via degree)
    # Higher clustering means picking this vertex blocks fewer potential edges
    nbr_clustering = np.zeros(n, dtype=float)
    for i in range(n):
        nbrs = adj_bool[i]
        if degree[i] > 1:
            # Count edges among neighbors
            sub = adj_bool[np.where(nbrs)[0]][:, np.where(nbrs)[0]]
            nbr_clustering[i] = sub.sum() / (degree[i] * (degree[i] - 1))
    
    # Effective loss: how many vertices we lose when picking this one
    # Accounts for the fact that neighbors of neighbors may be redundant
    eff_loss = degree.copy()
    
    # Score: we want low degree, high mean neighbor degree (so blocking high-deg neighbors is good),
    # and high neighbor clustering (neighbors are already connected, so blocking them wastes less)
    
    def simulate_greedy_fast(scores):
        """Fast greedy simulation using vectorized operations where possible."""
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
    
    def simulate_from_order(ord_arr):
        """Simulate greedy from a fixed order."""
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
    
    # Generate multiple heuristic score vectors
    heuristics = []
    
    # H1: Pure low-degree
    heuristics.append(-degree.copy())
    
    # H2: Low degree + bonus for high mean neighbor degree
    heuristics.append(-degree + 0.3 * mean_nbr_deg)
    
    # H3: Low degree + clustering bonus
    heuristics.append(-degree + 0.5 * nbr_clustering)
    
    # H4: Combined structural
    heuristics.append(-degree + 0.2 * mean_nbr_deg + 0.3 * nbr_clustering)
    
    # H5: Log-transformed degree
    heuristics.append(-np.log1p(degree) + 0.2 * mean_nbr_deg)
    
    # H6: Square root degree
    heuristics.append(-np.sqrt(degree + 1) + 0.3 * mean_nbr_deg)
    
    # H7: Degree with quadratic penalty
    heuristics.append(-degree - 0.01 * degree**2 + 0.2 * mean_nbr_deg)
    
    # H8: Randomized versions of best heuristics
    rng = np.random.RandomState(42)
    for base in heuristics[:4]:
        heuristics.append(base + rng.normal(0, 0.3, n))
    
    # Evaluate all heuristics and find best
    best_scores = None
    best_size = 0
    
    for h in heuristics:
        size = simulate_greedy_fast(h)
        if size > best_size:
            best_size = size
            best_scores = h.copy()
    
    if best_scores is None:
        best_scores = -degree.copy()
        best_size = simulate_greedy_fast(best_scores)
    
    # Local search: try swapping pairs in the order
    order = np.argsort(-best_scores, kind='stable')
    current_size = best_size
    
    # Phase 1: Hill climbing with swaps
    improved = True
    max_sweeps = 10
    sweep = 0
    while improved and sweep < max_sweeps and time.time() < deadline:
        improved = False
        sweep += 1
        # Try random swaps
        n_tries = min(5000, max(1000, n * 20))
        for _ in range(n_tries):
            if time.time() > deadline:
                break
            i, j = np.random.randint(0, n, 2)
            if i == j:
                continue
            i, j = int(i), int(j)
            
            order[i], order[j] = order[j], order[i]
            new_size = simulate_from_order(order)
            
            if new_size > current_size:
                current_size = new_size
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    # Reconstruct scores from current order
                    best_scores = np.zeros(n)
                    for rank, v in enumerate(order):
                        best_scores[v] = n - rank
            else:
                # Revert
                order[i], order[j] = order[j], order[i]
    
    # Phase 2: Targeted local search around best solution
    if time.time() < deadline:
        # Try inserting high-priority vertices earlier
        for _ in range(2000):
            if time.time() > deadline:
                break
            # Pick a random vertex and try moving it up
            v = np.random.randint(0, n)
            pos = np.where(order == v)[0][0]
            if pos == 0:
                continue
            target_pos = max(0, pos - np.random.randint(1, max(2, pos // 2)))
            
            # Move v to target_pos
            v_elem = order[pos]
            order = np.delete(order, pos)
            order = np.insert(order, target_pos, v_elem)
            
            new_size = simulate_from_order(order)
            if new_size > current_size:
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_scores = np.zeros(n)
                    for rank, v in enumerate(order):
                        best_scores[v] = n - rank
            else:
                # Revert: put v back
                v_elem = order[target_pos]
                order = np.delete(order, target_pos)
                order = np.insert(order, pos, v_elem)
    
    # Phase 3: Final polish - try small perturbations of the best scores
    if time.time() < deadline:
        for trial in range(50):
            if time.time() > deadline:
                break
            # Small random perturbation
            perturbed = best_scores + np.random.normal(0, 0.1, n)
            size = simulate_greedy_fast(perturbed)
            if size > best_size:
                best_size = size
                best_scores = perturbed.copy()
    
    return best_scores
# EVOLVE-BLOCK-END
