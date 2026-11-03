"""Bootstrap portfolio: SEARCH approach with time-bounded local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.linalg


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping facilities at positions i and j in permutation p.
    Uses vectorized sum-minus-exclusion for speed.
    """
    x = p[i]
    y = p[j]
    
    # Direct pair contribution
    delta = a[i, j] * (b[y, x] - b[x, y]) + a[j, i] * (b[x, y] - b[y, x])
    
    # Vectorized sum over all k
    V1 = b[y, p]
    V2 = b[x, p]
    V3 = b[p, y]
    V4 = b[p, x]
    
    T1 = a[i, :] * (V1 - V2)
    T2 = a[:, i] * (V3 - V4)
    T3 = a[j, :] * (V2 - V1)
    T4 = a[:, j] * (V4 - V3)
    
    S = np.sum(T1 + T2 + T3 + T4)
    
    # Subtract terms for k=i and k=j
    # T(i)
    T_i = (a[i, i] * (b[y, x] - b[x, x]) + 
           a[i, i] * (b[x, y] - b[x, x]) + 
           a[j, i] * (b[x, x] - b[y, x]) + 
           a[i, j] * (b[x, x] - b[x, y]))
           
    # T(j)
    T_j = (a[i, j] * (b[y, y] - b[x, y]) + 
           a[j, i] * (b[y, y] - b[y, x]) + 
           a[j, j] * (b[x, y] - b[y, y]) + 
           a[j, j] * (b[y, x] - b[y, y]))
           
    delta += S - T_i - T_j
    
    return int(delta)


def _greedy_init(a, b, n):
    """Greedy initialization based on row sums."""
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _spectral_init(a, b, n, seed=42):
    """
    Spectral embedding initialization.
    
    Construct a linearization matrix M that captures the joint structure of a and b,
    then use leading eigenvectors to generate a high-quality initial permutation.
    
    The idea: QAP cost = sum_{i,j} a[i,j]*b[p[i],p[j]].
    We can approximate this by finding a permutation that aligns the "structure" of a with b.
    
    Approach:
    1. Compute row/col sums to get a "gravity" vector for each matrix
    2. Build a combined matrix that reflects interaction patterns
    3. Use eigen-decomposition to find a natural ordering
    """
    rng = np.random.RandomState(seed)
    
    # Step 1: Compute aggregate weights
    a_row_sums = a.sum(axis=1)  # sum over j for each i
    a_col_sums = a.sum(axis=0)  # sum over i for each j
    b_row_sums = b.sum(axis=1)
    b_col_sums = b.sum(axis=0)
    
    # Step 2: Build a linearization matrix
    # M[i,j] approximates how facility i should relate to location j
    # Use: M = a @ b + diag(a_row_sums) @ diag(b_row_sums) / n
    # This captures both direct interaction and aggregate compatibility
    
    # Normalize for numerical stability
    a_norm = a.astype(np.float64)
    b_norm = b.astype(np.float64)
    
    # Compute M = a @ b (captures pairwise interaction structure)
    M = a_norm @ b_norm
    
    # Add diagonal correction to account for self-interactions
    # This helps align the "center of mass" of each facility with locations
    a_mean = a_norm.mean(axis=1, keepdims=True)
    b_mean = b_norm.mean(axis=0, keepdims=True)
    M += np.outer(a_mean.flatten(), b_mean.flatten())
    
    # Step 3: Eigen-decomposition of symmetric part
    # Use the symmetric part for stable eigendecomposition
    M_sym = (M + M.T) / 2.0
    
    # Get leading eigenvectors
    try:
        eigvals, eigvecs = scipy.linalg.eigh(M_sym)
        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        eigvals = eigvals[idx]
        
        # Use top k eigenvectors to create an embedding
        k = min(8, n)
        X = eigvecs[:, :k]  # n x k embedding
        
        # Project to 1D for ordering
        # Use the first few eigenvectors weighted by their eigenvalues
        weights = eigvals[:k] ** 0.5
        X_weighted = X * weights[np.newaxis, :]
        
        # Create a 1D score for each facility
        scores = X_weighted @ np.ones(k)
        
        # Sort facilities by their spectral score
        facility_order = np.argsort(scores)
        
        # For locations, use a similar approach on b
        # But we need to match facility i to location p[i]
        # So we need to order locations too
        
        # Alternative: use the full M matrix directly
        # M[i,j] tells us how good it is to put facility i at location j
        # We want to find a permutation that maximizes sum M[i, p[i]]
        
        # Use greedy assignment based on M
        # Or use the eigenvector structure to create a natural ordering
        
        # Let's try: order facilities by their projection on first eigenvector
        # and order locations by their projection on first eigenvector of b^T b or similar
        
        # Actually, let's use a simpler approach:
        # Create a combined score for each (facility, location) pair
        # and use a greedy or auction-like assignment
        
        # For now, let's use the spectral ordering of facilities
        # and match them to locations ordered by b's row sums
        
        loc_order = np.argsort(b_row_sums)
        
        p = np.empty(n, dtype=np.int64)
        p[facility_order] = loc_order
        
        return p
        
    except Exception:
        # Fallback to greedy
        return _greedy_init(a, b, n)


def _random_init(n):
    """Random permutation initialization."""
    return np.random.permutation(n)


def _tabu_search(a, b, p, time_limit, tabu_tenure=30, max_iters=None, sample_size=1500):
    """Tabu search with 1-swap neighborhood."""
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    tabu_list = {}
    iteration = 0
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if max_iters is not None and iteration >= max_iters:
            break
        
        iteration += 1
        
        # Best improving 1-swap
        best_swap_i, best_swap_j = -1, -1
        best_delta = 0
        
        # Sample a subset of swaps for speed
        actual_sample_size = min(sample_size, n * n // 4)
        
        if actual_sample_size < n * n:
            # Generate random pairs using randint for speed
            i_arr = np.random.randint(0, n, actual_sample_size)
            j_arr = np.random.randint(0, n, actual_sample_size)
            
            # Filter i != j
            mask = i_arr != j_arr
            i_arr = i_arr[mask]
            j_arr = j_arr[mask]
            
            for k in range(len(i_arr)):
                i = int(i_arr[k])
                j = int(j_arr[k])
                
                key = (i, j) if i < j else (j, i)
                if key in tabu_list and tabu_list[key] > iteration:
                    continue
                
                delta = _swap_delta(a, b, p, i, j)
                
                if delta < best_delta:
                    best_delta = delta
                    best_swap_i, best_swap_j = i, j
        else:
            # Full neighborhood (unlikely for n=256)
            for i in range(n):
                for j in range(i+1, n):
                    key = (i, j)
                    if key in tabu_list and tabu_list[key] > iteration:
                        continue
                    delta = _swap_delta(a, b, p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_swap_i, best_swap_j = i, j
        
        if best_swap_i == -1:
            break
        
        # Perform the swap
        i, j = best_swap_i, best_swap_j
        key = (i, j) if i < j else (j, i)
        tabu_list[key] = iteration + tabu_tenure
        
        p[i], p[j] = p[j], p[i]
        current_cost += best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Clean up expired tabu entries occasionally
        if iteration % 1000 == 0:
            expired = [k for k, v in tabu_list.items() if v <= iteration]
            for k in expired:
                del tabu_list[k]
    
    return best_p, best_cost





def _solve_tai256c(a, b):
    n = len(a)
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Phase 1: Generate diverse initial solutions
    inits = []
    inits.append(_greedy_init(a, b, n))
    inits.append(_spectral_init(a, b, n, seed=42))
    inits.append(_spectral_init(a, b, n, seed=123))
    for _ in range(3):
        inits.append(_random_init(n))
    
    best_p = None
    best_cost = np.inf
    
    # Phase 2: Quick Tabu Search on each initialization to find a good basin
    # Allocate ~4s per init (5 inits * 4s = 20s)
    quick_time = 4.0
    
    for i, p_init in enumerate(inits):
        if time.time() >= deadline - 40:
            break
        
        p_tabu, cost_tabu = _tabu_search(a, b, p_init, time_limit=quick_time, sample_size=500)
        
        if cost_tabu < best_cost:
            best_cost = cost_tabu
            best_p = p_tabu.copy()
    
    # Phase 3: Long Tabu Search from the best quick solution
    if best_p is not None and time.time() < deadline - 5:
        polish_time = deadline - time.time() - 5
        p_polished, cost_polished = _tabu_search(a, b, best_p, time_limit=polish_time, sample_size=2500)
        
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished
    
    # Fallback if no time was left or error
    if best_p is None:
        p_greedy = _greedy_init(a, b, n)
        best_p, best_cost = _tabu_search(a, b, p_greedy, time_limit=min(10.0, deadline - time.time() - 5))
    
    return best_p.tolist()


def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
