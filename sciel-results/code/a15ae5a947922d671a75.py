import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    deadline = time.time() + 115.0  # safety margin within 120s limit
    
    # Helper: compute cost of permutation
    def compute_cost(pos):
        return int(np.sum(a * b[np.ix_(pos, pos)]))
    
    # Helper: compute delta for swapping positions i and j
    def swap_delta(pos, i, j):
        x = pos[i]
        y = pos[j]
        diff1 = b[y, pos] - b[x, pos]
        diff2 = b[x, pos] - b[y, pos]
        diff3 = b[pos, y] - b[pos, x]
        diff4 = b[pos, x] - b[pos, y]
        return (a[i, :] @ diff1 + a[j, :] @ diff2 + a[:, i] @ diff3 + a[:, j] @ diff4)
    
    # Frank-Wolfe on Birkhoff polytope
    def frank_wolfe_init():
        """
        Use Frank-Wolfe method on the Birkhoff polytope to find a good
        doubly stochastic matrix, then extract a permutation.
        
        The QAP can be written as: min X^T A X B X^T where X is doubly stochastic.
        We use the linear assignment problem to project onto the Birkhoff polytope.
        """
        rng = np.random.default_rng(123)
        
        # Start with a random permutation as initial point
        init_perm = rng.permutation(n)
        X = np.zeros((n, n))
        X[init_perm, np.arange(n)] = 1.0
        
        # Frank-Wolfe iterations
        best_X = X.copy()
        best_cost = compute_cost(init_perm)
        
        for it in range(200):
            if time.time() > deadline - 60.0:
                break
            
            # Compute gradient: G = 2*(A @ X @ B + B @ X @ A)
            # Actually for QAP: f(X) = sum_{i,j,k,l} a[i,j]*b[k,l]*X[i,k]*X[j,l]
            # Gradient: G[i,k] = 2 * sum_{j,l} a[i,j]*b[k,l]*X[j,l] + 2 * sum_{j,l} a[j,i]*b[l,k]*X[j,l]
            # Simpler: G = 2*(A @ X @ B + B.T @ X.T @ A.T).T
            
            # Compute gradient efficiently
            AX = a @ X
            G = 2.0 * (AX @ b + b.T @ X.T @ a.T).T
            
            # Find linear minimizer: min_{Y in Birkhoff} <G, Y>
            # This is the linear assignment problem
            row_idx, col_idx = linear_sum_assignment(G)
            Y = np.zeros((n, n))
            Y[row_idx, col_idx] = 1.0
            
            # Step size: use backtracking line search
            # f(X) = <A, X @ B @ X.T> approximately
            # Use standard FW step size: gamma_t = 2/(t+2)
            gamma = 2.0 / (it + 2)
            
            X_new = (1 - gamma) * X + gamma * Y
            
            # Extract permutation from X_new
            row_idx_new, col_idx_new = linear_sum_assignment(-X_new)
            perm = col_idx_new
            
            cost = compute_cost(perm)
            if cost < best_cost:
                best_cost = cost
                best_X = X_new.copy()
            
            X = X_new
        
        # Final extraction
        row_idx, col_idx = linear_sum_assignment(-X)
        return col_idx, best_cost
    
    # Local search with 2-opt
    def local_search(pos, max_iter=5000):
        """Greedy 2-opt local search."""
        pos = pos.copy()
        current_cost = compute_cost(pos)
        rng = np.random.default_rng(456)
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Try random swaps
            for _ in range(100):
                i = rng.integers(n)
                j = rng.integers(n)
                if i == j:
                    continue
                
                delta = swap_delta(pos, i, j)
                if delta < 0:
                    pos[i], pos[j] = pos[j], pos[i]
                    current_cost += delta
                    improved = True
            
            if not improved:
                break
        
        return pos, current_cost
    
    # Phase 1: Frank-Wolfe initialization
    pos_fw, cost_fw = frank_wolfe_init()
    
    # Phase 2: Local search from FW solution
    pos_ls, cost_ls = local_search(pos_fw, max_iter=10000)
    
    best_pos = pos_ls
    best_cost = cost_ls
    
    # Phase 3: Iterated local search with perturbations
    rng = np.random.default_rng(789)
    
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb best solution
        pos = best_pos.copy()
        n_swaps = n // 8
        for _ in range(n_swaps):
            i, j = rng.choice(n, size=2, replace=False)
            pos[i], pos[j] = pos[j], pos[i]
        
        # Local search
        pos, cost = local_search(pos, max_iter=5000)
        
        if cost < best_cost:
            best_cost = cost
            best_pos = pos
    
    # Final polish
    best_pos, best_cost = local_search(best_pos, max_iter=20000)
    
    return best_pos.tolist()
