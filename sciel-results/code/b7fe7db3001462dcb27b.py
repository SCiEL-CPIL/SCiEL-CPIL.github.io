"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def solve(a, b):
    n = len(a)
    
    # Convert to float for numerical stability in Sinkhorn iterations
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    
    # Sinkhorn-Knopp algorithm with temperature annealing on the Birkhoff polytope
    # We minimize the quadratic objective over doubly stochastic matrices
    
    def sinkhorn(K, num_iters, eps=1.0):
        """
        Sinkhorn-Knopp algorithm to find the closest doubly stochastic matrix
        to the kernel K under the relative entropy.
        
        Parameters:
        -----------
        K : np.ndarray
            Positive matrix (kernel)
        num_iters : int
            Number of Sinkhorn iterations
        eps : float
            Entropic regularization parameter (smaller = closer to permutation)
        
        Returns:
        --------
        P : np.ndarray
            Doubly stochastic matrix
        """
        n = K.shape[0]
        # Initialize with positive matrix
        K_safe = np.maximum(K, 1e-10)
        
        # Normalize to have positive entries
        K_safe = K_safe / K_safe.max()
        
        # Initialize row and column scaling vectors
        u = np.ones(n)
        v = np.ones(n)
        
        for _ in range(num_iters):
            # Row scaling
            u = 1.0 / (eps * K_safe @ v)
            # Column scaling
            v = 1.0 / (eps * K_safe.T @ u)
        
        P = np.outer(u, v) * K_safe
        return P
    
    def birkhoff_cost(P):
        """
        Compute the quadratic assignment cost for a doubly stochastic matrix P.
        cost = sum_{i,j} a[i,j] * (P^T B P)[i,j]
        """
        # P^T B P
        PB = P.T @ B
        PBP = PB @ P
        cost = np.sum(A * PBP)
        return cost
    
    def gradient_birkhoff(P):
        """
        Compute the gradient of the Birkhoff cost with respect to P.
        grad = A^T P + P A^T (for symmetric A, B)
        More precisely: d/dP [tr(P^T A P B)] = A P B^T + A^T P B
        """
        # For cost = sum_{i,j} A[i,j] * (P^T B P)[i,j] = tr(P^T A P B)
        # Gradient = A P B^T + A^T P B
        # Since A and B are symmetric in QAP:
        grad = A @ P @ B + A.T @ P @ B.T
        return grad
    
    def extract_permutation(P):
        """
        Extract a permutation from a doubly stochastic matrix P
        using the Hungarian algorithm.
        """
        # Use negative cost for minimization
        cost_matrix = -P
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.zeros(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    
    def compute_cost(p):
        """Compute the exact QAP cost for a permutation."""
        total = 0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total
    
    # Temperature annealing schedule
    # Start with high temperature (smooth) and decrease to low temperature (sharp)
    initial_eps = 1.0
    final_eps = 0.01
    num_annealing_steps = 20
    
    # Create annealing schedule
    eps_schedule = np.geomspace(initial_eps, final_eps, num_annealing_steps)
    
    # Initialize kernel
    # Use a random positive initialization
    np.random.seed(42)
    K = np.random.exponential(1.0, size=(n, n))
    
    # Warm start: use a simple heuristic to get a good initial point
    # Match high-flow facilities to low-distance locations
    a_row_sums = A.sum(axis=1)
    b_row_sums = B.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    
    # Create initial doubly stochastic matrix from this permutation
    P = np.zeros((n, n))
    for i in range(n):
        P[i, p_init[i]] = 1.0
    
    # Run Sinkhorn with annealing
    start_time = time.time()
    time_budget = 110.0  # Leave 10 seconds for final local search
    
    current_P = P.copy()
    current_cost = birkhoff_cost(current_P)
    
    for step, eps in enumerate(eps_schedule):
        if time.time() - start_time > time_budget:
            break
        
        # Create kernel from current P with added noise for exploration
        # Mix current solution with random positive matrix
        noise = np.random.exponential(0.1, size=(n, n))
        K = current_P + noise
        
        # Run Sinkhorn iterations
        num_iters = max(50, 200 - step * 10)
        current_P = sinkhorn(K, num_iters, eps)
        
        # Compute cost
        new_cost = birkhoff_cost(current_P)
        
        # Accept if better or with some probability (simulated annealing)
        if new_cost < current_cost:
            current_cost = new_cost
        else:
            # Accept with probability exp(-delta / temperature)
            delta = new_cost - current_cost
            if delta > 0:
                accept_prob = np.exp(-delta / (eps * 1000))
                if np.random.random() < accept_prob:
                    current_cost = new_cost
                else:
                    # Revert
                    pass
    
    # Extract permutation from final doubly stochastic matrix
    p = extract_permutation(current_P)
    
    # Compute exact cost
    current_cost_exact = compute_cost(p)
    
    # Final local search with efficient swap evaluation
    # Precompute for efficient swap delta calculation
    # For swap of p[i] and p[j]:
    # delta = sum_k [a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) + 
    #                   a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]]) +
    #                   a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]]) +
    #                   a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])]
    #        + a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]]) +
    #          a[j,i]*(b[p[i],p[j]] - b[p[j],p[i]])
    
    def swap_delta(i, j, p):
        """Compute the change in cost when swapping p[i] and p[j]."""
        pi, pj = p[i], p[j]
        
        # Terms involving i and j
        delta = 0
        
        # For all k != i, j:
        # a[i,k] * (b[pj, p[k]] - b[pi, p[k]])
        # a[k,i] * (b[p[k], pj] - b[p[k], pi])
        # a[j,k] * (b[pi, p[k]] - b[pj, p[k]])
        # a[k,j] * (b[p[k], pi] - b[p[k], pj])
        
        # Vectorized computation
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        k_indices = np.where(mask)[0]
        
        if len(k_indices) > 0:
            pk = p[k_indices]
            
            # a[i, k] terms
            delta += np.sum(a[i, k_indices] * (b[pj, pk] - b[pi, pk]))
            # a[k, i] terms
            delta += np.sum(a[k_indices, i] * (b[pk, pj] - b[pk, pi]))
            # a[j, k] terms
            delta += np.sum(a[j, k_indices] * (b[pi, pk] - b[pj, pk]))
            # a[k, j] terms
            delta += np.sum(a[k_indices, j] * (b[pk, pi] - b[pk, pj]))
        
        # Direct i-j and j-i terms
        delta += a[i, j] * (b[pj, pi] - b[pi, pj])
        delta += a[j, i] * (b[pi, pj] - b[pj, pi])
        
        return delta
    
    # Local search passes
    improved = True
    max_passes = 5
    pass_count = 0
    remaining_time = time_budget - (time.time() - start_time)
    
    while improved and pass_count < max_passes and remaining_time > 5.0:
        improved = False
        pass_count += 1
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 2.0:
            break
            
        for i in range(n):
            if time.time() - start_time > time_budget:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > time_budget:
                    break
                d = swap_delta(i, j, p)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost_exact += d
                    improved = True
    
    return p.tolist()
# EVOLVE-BLOCK-END
