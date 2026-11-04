"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit (150s kill)
    
    def calc_cost(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))
    
    # Phase 1: Initial solution via linear assignment on row sums
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    cost_matrix = np.outer(a_row_sums, b_row_sums)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    p_init = np.empty(n, dtype=np.int64)
    p_init[row_ind] = col_ind
    p_init = p_init.copy()
    
    best_p = p_init.copy()
    best_cost = calc_cost(best_p)
    
    # Phase 2: Continuous relaxation via Sinkhorn-Knopp with annealing
    # Represent assignment as doubly stochastic matrix X
    # Objective: minimize trace(A X^T B X) = sum_{i,j,k,l} A[i,j] X[i,k] B[k,l] X[j,l]
    
    # Initialize X from initial permutation
    X = np.zeros((n, n), dtype=np.float64)
    X[np.arange(n), p_init] = 1.0
    
    # Sinkhorn-Knopp: alternate row and column normalization
    def sinkhorn(X, eps, max_iter=1000):
        """Normalize X to be doubly stochastic using Sinkhorn-Knopp."""
        X = X.copy()
        X = np.maximum(X, 1e-12)  # Avoid zeros
        for _ in range(max_iter):
            # Row normalization
            row_sums = X.sum(axis=1, keepdims=True)
            X = X / row_sums
            # Column normalization
            col_sums = X.sum(axis=0, keepdims=True)
            X = X / col_sums
            # Check convergence
            if np.max(np.abs(row_sums - 1.0)) < 1e-8:
                break
        return X
    
    # Gradient of the quadratic objective with respect to X
    # f(X) = trace(A X^T B X)
    # df/dX = 2 * A^T X B^T (for symmetric A, B this is 2 * A X B)
    def compute_gradient(X, A, B):
        return 2.0 * (A @ X @ B)
    
    # Log-sum-exp smoothed assignment energy for regularization
    def logsumexp_smoothed(X, eps):
        """Add entropy regularization to encourage permutation-like solutions."""
        # Entropy term: -sum(X * log(X))
        X_clipped = np.maximum(X, 1e-12)
        return -np.sum(X_clipped * np.log(X_clipped))
    
    # Annealing schedule
    num_phases = 50
    phase_duration = (deadline - time.time()) / num_phases
    if phase_duration < 0.1:
        phase_duration = 0.1
    
    current_X = X.copy()
    current_cost = best_cost
    
    for phase in range(num_phases):
        phase_start = time.time()
        if phase_start > deadline - 5.0:
            break
        
        # Temperature schedule: start high, decrease
        T = max(0.01, 1.0 - phase / num_phases)
        lr = 0.01 * T  # Learning rate annealing
        
        # Perform gradient descent steps within this phase
        max_steps = 200
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            # Compute gradient
            grad = compute_gradient(current_X, a, b)
            
            # Add entropy regularization gradient
            X_clipped = np.maximum(current_X, 1e-12)
            entropy_grad = -np.log(X_clipped) - 1.0
            grad = grad + 0.1 * entropy_grad
            
            # Gradient descent step
            current_X = current_X - lr * grad
            
            # Project back to Birkhoff polytope via Sinkhorn
            current_X = sinkhorn(current_X, T, max_iter=100)
            
            # Extract permutation from current_X (greedy assignment)
            # Use Hungarian algorithm on -X to find best matching
            perm_indices = linear_sum_assignment(-current_X)
            perm = np.empty(n, dtype=np.int64)
            perm[perm_indices[0]] = perm_indices[1]
            
            # Evaluate cost
            cost = calc_cost(perm)
            
            if cost < best_cost:
                best_cost = cost
                best_p = perm.copy()
            
            # Occasionally accept worse solutions for exploration (simulated annealing style)
            elif cost < current_cost and random.random() < np.exp(-(cost - current_cost) / max(T, 0.01)):
                current_cost = cost
                current_X = np.zeros((n, n), dtype=np.float64)
                current_X[np.arange(n), perm] = 1.0
                current_X = sinkhorn(current_X, T, max_iter=50)
        
        # Update current_X to be close to best found so far
        if phase % 5 == 0:
            current_X = np.zeros((n, n), dtype=np.float64)
            current_X[np.arange(n), best_p] = 1.0
            current_X = sinkhorn(current_X, T, max_iter=50)
            current_cost = best_cost
    
    # Phase 3: Local search polish on the best permutation found
    polish_deadline = deadline - 3.0
    
    def calc_delta(i, j, perm):
        u = perm[i]
        v = perm[j]
        
        # Diagonal part
        delta = (a[i, i] - a[j, j]) * (b[v, v] - b[u, u])
        
        # Off-diagonal part
        diff_A = a[i] - a[j]
        diff_B = b[v, perm] - b[u, perm]
        
        total_dot = np.dot(diff_A, diff_B)
        
        # Subtract terms for k=i and k=j
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        off_diag_sum = total_dot - term_i - term_j
        delta += 2 * off_diag_sum
        
        return int(delta)
    
    current_p = best_p.copy()
    current_cost = best_cost
    
    # Tabu search with short tenure
    tabu_list = {}
    tabu_tenure = 30
    iteration = 0
    
    while time.time() < polish_deadline:
        improved = False
        start_i = random.randint(0, n - 1)
        
        for k in range(n):
            i = (start_i + k) % n
            if time.time() > polish_deadline:
                break
            u = current_p[i]
            
            start_j = random.randint(0, n - 1)
            for m in range(n - 1):
                j = (start_j + m) % n
                if i == j:
                    continue
                if time.time() > polish_deadline:
                    break
                
                is_tabu = False
                if (i, j) in tabu_list and tabu_list[(i, j)] > iteration:
                    is_tabu = True
                if (j, i) in tabu_list and tabu_list[(j, i)] > iteration:
                    is_tabu = True
                
                delta = calc_delta(i, j, current_p)
                
                if delta < 0 or (is_tabu and current_cost + delta < best_cost):
                    v = current_p[j]
                    current_p[i], current_p[j] = v, u
                    current_cost += delta
                    
                    tabu_list[(i, j)] = iteration + tabu_tenure
                    tabu_list[(j, i)] = iteration + tabu_tenure
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = current_p.copy()
                    
                    improved = True
                    break
            
            if improved:
                break
        
        iteration += 1
        
        if not improved:
            # Local optimum reached, do a random kick
            if time.time() > polish_deadline - 5.0:
                break
            
            i_kick = random.randint(0, n - 1)
            j_kick = random.randint(0, n - 1)
            if i_kick != j_kick:
                delta = calc_delta(i_kick, j_kick, current_p)
                v = current_p[j_kick]
                current_p[i_kick], current_p[j_kick] = v, current_p[i_kick]
                current_cost += delta
                
                tabu_list[(i_kick, j_kick)] = iteration + tabu_tenure
                tabu_list[(j_kick, i_kick)] = iteration + tabu_tenure
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
    
    # Final safety check
    if sorted(best_p.tolist()) != list(range(n)):
        valid_p = np.arange(n, dtype=np.int64)
        return valid_p.tolist()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
