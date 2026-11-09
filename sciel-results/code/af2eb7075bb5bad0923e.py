# EVOLVE-BLOCK-START
import numpy as np
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    D = entropy of (x-y) distribution, M = max(entropy of x, x+y, x+2y distributions)
    Returns a 10x10 probability distribution P[y,x] maximizing D/M.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Leave 10s safety margin
    
    n = 10
    
    # Precompute projection indices for fast evaluation
    # For each cell (y,x), precompute which bin it falls into for each projection
    diff_idx = np.zeros((n, n), dtype=np.int32)  # x-y + (n-1)
    x_idx = np.zeros((n, n), dtype=np.int32)     # x
    sum_idx = np.zeros((n, n), dtype=np.int32)   # x+y
    two_sum_idx = np.zeros((n, n), dtype=np.int32)  # x+2y
    
    for y in range(n):
        for x in range(n):
            diff_idx[y, x] = x - y + (n-1)
            x_idx[y, x] = x
            sum_idx[y, x] = x + y
            two_sum_idx[y, x] = x + 2*y
    
    n_diff = 2*n - 1
    n_x = n
    n_sum = 2*n - 1
    n_two_sum = 3*n - 1
    
    def compute_entropies_fast(P):
        """Fast computation of D and M using precomputed indices"""
        P_flat = P.flatten()
        
        # Use np.bincount for fast aggregation
        diff_counts = np.bincount(diff_idx.flatten(), weights=P_flat, minlength=n_diff)
        x_counts = np.bincount(x_idx.flatten(), weights=P_flat, minlength=n_x)
        sum_counts = np.bincount(sum_idx.flatten(), weights=P_flat, minlength=n_sum)
        two_sum_counts = np.bincount(two_sum_idx.flatten(), weights=P_flat, minlength=n_two_sum)
        
        def shannon_entropy(counts):
            counts = counts[counts > 1e-15]
            if len(counts) == 0:
                return 0.0
            return -np.sum(counts * np.log2(counts))
        
        D = shannon_entropy(diff_counts)
        H0 = shannon_entropy(x_counts)
        H1 = shannon_entropy(sum_counts)
        H2 = shannon_entropy(two_sum_counts)
        M = max(H0, H1, H2)
        
        return D, M
    
    def objective(P):
        D, M = compute_entropies_fast(P)
        if M < 1e-9:
            return 0.0
        return D / M
    
    # Start with uniform distribution
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P.copy()
    best_score = objective(P)
    
    np.random.seed(42)
    
    def normalize(P):
        P = np.maximum(P, 0)
        s = P.sum()
        if s < 1e-15:
            return np.ones((n, n)) / (n*n)
        return P / s
    
    def coordinate_descent(P, max_iter=500, step_size=0.05):
        """Coordinate descent optimization"""
        P = P.copy()
        current_score = objective(P)
        best_local_P = P.copy()
        best_local_score = current_score
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            for y in range(n):
                for x in range(n):
                    old_val = P[y, x]
                    
                    # Try increasing
                    P[y, x] = old_val + step_size
                    P_new = normalize(P)
                    score_new = objective(P_new)
                    if score_new > best_local_score + 1e-12:
                        best_local_score = score_new
                        best_local_P = P_new.copy()
                        improved = True
                    
                    # Try decreasing
                    P[y, x] = max(0, old_val - step_size)
                    P_new = normalize(P)
                    score_new = objective(P_new)
                    if score_new > best_local_score + 1e-12:
                        best_local_score = score_new
                        best_local_P = P_new.copy()
                        improved = True
                    
                    P[y, x] = old_val
            
            if not improved:
                break
        
        return best_local_P, best_local_score
    
    # Phase 1: Multiple random restarts
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            P_init = best_P.copy()
        else:
            # Random Dirichlet distribution
            P_rand = np.random.dirichlet(np.ones(n*n) * 0.5).reshape(n, n)
            P_init = P_rand
        
        P_local, score_local = coordinate_descent(P_init, max_iter=300, step_size=0.03)
        
        if score_local > best_score:
            best_score = score_local
            best_P = P_local.copy()
    
    # Phase 2: Fine-tuning
    if time.time() < deadline:
        P_fine, score_fine = coordinate_descent(best_P, max_iter=500, step_size=0.01)
        if score_fine > best_score:
            best_score = score_fine
            best_P = P_fine.copy()
    
    # Phase 3: Ultra-fine tuning
    if time.time() < deadline:
        P_ultra, score_ultra = coordinate_descent(best_P, max_iter=800, step_size=0.003)
        if score_ultra > best_score:
            best_score = score_ultra
            best_P = P_ultra.copy()
    
    # Final validation and normalization
    best_P = np.maximum(best_P, 0)
    s = best_P.sum()
    if s < 1e-15:
        best_P = np.ones((n, n)) / (n*n)
    else:
        best_P = best_P / s
    
    return best_P

# EVOLVE-BLOCK-END
