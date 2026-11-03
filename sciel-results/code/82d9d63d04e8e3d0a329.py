"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    """
    Solve the Quadratic Assignment Problem for a n x n instance.
    Uses a fast vectorized 2-opt local search with Iterated Local Search (ILS).
    Assumes a and b are symmetric (standard for QAPLIB tai instances).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin within 120s limit
    
    n = len(a)
    
    # Convert to float64 for BLAS operations
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    # Symmetrize if necessary (tai256c is symmetric, but this ensures correctness for general case if slight asymmetry exists)
    # However, for speed, we assume symmetry as per task context. 
    # If a or b are not symmetric, the delta formula 4*(Vi(x)-Vi(y)) is incorrect.
    # For tai256c, a and b are symmetric.
    
    def compute_cost(p):
        """Compute the exact QAP cost for assignment p."""
        B_p = b[np.ix_(p, p)]
        return int(np.sum(a * B_p))
    
    def greedy_init():
        """
        Rank-matching initialization.
        """
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p
    
    def local_search(p, max_time=None):
        """
        Fast 2-opt local search using vectorized delta calculation.
        Returns (p, cost).
        """
        p = p.copy()
        cost = compute_cost(p)
        
        # Precompute B_p = b[np.ix_(p, p)]
        B_p = b[np.ix_(p, p)]
        
        if max_time is None:
            max_time = deadline
            
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Iterate through facilities in a fixed order (or random)
            # Fixed order is faster, but might be less robust. 
            # We'll use a random order per pass to avoid bias.
            indices = np.arange(n)
            np.random.shuffle(indices)
            
            for i in indices:
                if time.time() > max_time:
                    break
                
                # Compute V_i_all[j] = sum_k a[i,k] * b[p[k], p[j]]
                # This is a[i, :] @ B_p
                V_i_all = a[i, :] @ B_p
                
                # We want to find j != i that minimizes Delta(i, j)
                # Delta(i, j) = 4 * (V_i(p[i]) - V_i(p[j])) = 4 * (V_i_all[i] - V_i_all[j])
                # To minimize Delta, we need to maximize V_i_all[j].
                
                # Mask out i
                V_i_all[i] = -np.inf
                
                best_j = np.argmax(V_i_all)
                
                if best_j != i and V_i_all[best_j] > V_i_all[i] + 1e-9:
                    # Perform swap i, best_j
                    delta = 4.0 * (V_i_all[i] - V_i_all[best_j])
                    
                    # Update p
                    p[i], p[best_j] = p[best_j], p[i]
                    
                    # Update B_p
                    # Swap rows i and best_j
                    B_p[[i, best_j], :] = B_p[[best_j, i], :]
                    # Swap cols i and best_j
                    B_p[:, [i, best_j]] = B_p[:, [best_j, i]]
                    
                    cost += delta
                    improved = True
                    
                    # Optional: restart the loop if a large improvement was made
                    # For now, continue to next i to find more improvements in this pass
        
        return p, cost
    
    # ILS
    best_p = greedy_init()
    best_cost = compute_cost(best_p)
    
    # Initial local search
    best_p, best_cost = local_search(best_p)
    
    # Iterated Local Search
    while time.time() < deadline - 2.0:
        # Perturb
        p_current = best_p.copy()
        # Number of random swaps
        num_swaps = np.random.randint(2, 10)
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p_current[i], p_current[j] = p_current[j], p_current[i]
        
        # Local search
        p_new, cost_new = local_search(p_current)
        
        if cost_new < best_cost:
            best_p = p_new
            best_cost = cost_new
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
