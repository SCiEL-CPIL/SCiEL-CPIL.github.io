"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import eigh


def _spectral_init(a, b):
    """
    Spectral initialization: align high-flow facilities with low-distance locations.
    
    We compute the principal eigenvectors of the flow matrix A and distance matrix B.
    Facilities with high values in the principal eigenvector of A (high "flow centrality")
    should be matched with locations having low values in the principal eigenvector of B
    (low "distance centrality"), to minimize the sum of flow*distance products.
    """
    n = a.shape[0]
    
    # Symmetrize matrices for eigenvalue decomposition
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Get the principal (largest magnitude) eigenvectors
    # For A: we want facilities that are "central" in the flow network
    # For B: we want locations that are "peripheral" (low distance centrality)
    
    try:
        # Compute eigenvalues and eigenvectors of A_sym
        # We want the eigenvector corresponding to the largest eigenvalue
        eigvals_a, eigvecs_a = eigh(a_sym)
        # Largest eigenvalue is the last one in ascending order from eigh
        v_a = eigvecs_a[:, -1]
        
        # Compute eigenvalues and eigenvectors of B_sym
        eigvals_b, eigvecs_b = eigh(b_sym)
        # Smallest eigenvalue corresponds to the most "peripheral" structure
        # Actually, we want to align high flow with low distance.
        # The principal eigenvector of B gives distance centrality.
        # We want to assign high-flow facilities to low-distance-centrality locations.
        v_b = eigvecs_b[:, -1]  # Principal eigenvector of B
        
        # Rank facilities by their value in v_a (higher = more central in flow)
        # Rank locations by their value in v_b (lower = more peripheral in distance)
        facility_rank = np.argsort(-v_a)  # Descending order of flow centrality
        location_rank = np.argsort(v_b)   # Ascending order of distance centrality
        
        p = np.empty(n, dtype=np.int64)
        p[facility_rank] = location_rank
        
    except Exception:
        # Fallback to rank-matching if spectral fails
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
    
    return p


def _compute_cost(a, b, p):
    """Compute the QAP cost for a given permutation p."""
    n = len(p)
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Vectorized: create the permuted distance matrix
    b_perm = b[np.ix_(p, p)]
    return np.sum(a * b_perm)


def _neighborhood_2opt(a, b, p, max_swaps=None):
    """
    Generate and evaluate 2-swap (swap two facilities) neighbors.
    Returns the best neighbor permutation and its cost improvement.
    """
    n = len(p)
    current_cost = _compute_cost(a, b, p)
    
    best_cost = current_cost
    best_p = p.copy()
    
    if max_swaps is None:
        max_swaps = n * (n - 1) // 2
    
    # Evaluate all pairs of swaps
    # For efficiency, we can compute the delta cost for swapping positions i and j
    # But for simplicity and correctness, let's do a direct approach with some optimization
    
    # Precompute row sums and column sums for faster delta computation
    # Actually, let's just do a simple loop over a subset of swaps if n is large
    # For n=100, there are 4950 pairs, which is manageable
    
    p_list = p.tolist()
    
    for i in range(n):
        for j in range(i + 1, n):
            # Swap positions i and j
            p_list[i], p_list[j] = p_list[j], p_list[i]
            
            # Compute new cost
            new_cost = _compute_cost(a, b, np.array(p_list, dtype=np.int64))
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = np.array(p_list, dtype=np.int64)
            
            # Swap back
            p_list[i], p_list[j] = p_list[j], p_list[i]
    
    return best_p, best_cost


def _iterated_local_search(a, b, p_init, max_iterations=50, time_budget=100):
    """
    Iterated Local Search:
    1. Start from a good initial solution (spectral init)
    2. Perform local search (2-swap) to reach a local optimum
    3. Apply a perturbation
    4. Repeat local search
    5. Keep the best solution found
    """
    import time
    
    start_time = time.time()
    n = len(p_init)
    
    # Start from spectral initialization
    p_best = p_init.copy()
    cost_best = _compute_cost(a, b, p_best)
    
    p_current = p_best.copy()
    
    for iteration in range(max_iterations):
        # Check time budget
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Local search: 2-swap
        p_current, cost_current = _neighborhood_2opt(a, b, p_current)
        
        # Update best
        if cost_current < cost_best:
            cost_best = cost_current
            p_best = p_current.copy()
        
        # Perturbation: swap a random pair of elements
        # The strength of perturbation can increase with iterations
        num_swaps = min(1 + iteration // 10, 5)
        for _ in range(num_swaps):
            i, j = np.random.choice(n, size=2, replace=False)
            p_current[i], p_current[j] = p_current[j], p_current[i]
        
        # Check time budget again
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
    
    return p_best, cost_best


def solve(a, b):
    n = a.shape[0]
    
    # Spectral initialization
    p_spectral = _spectral_init(a, b)
    
    # Iterated Local Search
    p_best, cost_best = _iterated_local_search(a, b, p_spectral, max_iterations=30, time_budget=110)
    
    return p_best.tolist()
# EVOLVE-BLOCK-END
