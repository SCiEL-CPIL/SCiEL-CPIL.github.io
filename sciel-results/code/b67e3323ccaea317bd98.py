import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 150s limit

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_swap_delta(p, i, j):
        """Compute cost delta for swapping positions i and j."""
        u, v = p[i], p[j]
        if u == v:
            return 0
        # Vectorized delta computation
        diff_a_row = a[i, :] - a[j, :]
        diff_a_col = a[:, i] - a[:, j]
        diff_b_row = b[v, :] - b[u, :]
        diff_b_col = b[:, v] - b[:, u]
        
        s1 = np.dot(diff_a_row, diff_b_row[p])
        s2 = np.dot(diff_a_col, diff_b_col[p])
        
        # Subtract self-interaction terms
        s1 -= (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        s1 -= (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        s2 -= (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
        s2 -= (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
        
        cross = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])
        
        return s1 + s2 + cross

    def local_search(p, cost, max_passes=20):
        """First-improvement 2-opt local search."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return p, cost
                    delta = fast_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        # Build linearized cost matrix
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2
        C = M1 + M2
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    # Initial solution: greedy by row sums
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()

    # Simulated Annealing with multiple neighborhoods
    rng = np.random.RandomState(42)
    
    # Temperature schedule
    T_start = max(1.0, abs(current_cost) * 0.01)
    T_end = max(0.1, T_start * 0.001)
    alpha = 0.995  # cooling rate
    
    T = T_start
    step_count = 0
    last_ls_step = 0
    
    while time.time() < deadline:
        # Choose move type: 60% swap, 20% insert, 20% block
        move_type = rng.random()
        
        if move_type < 0.6:
            # Swap two random positions
            i, j = rng.choice(n, size=2, replace=False)
            if i == j:
                continue
            delta = fast_swap_delta(p, i, j)
            
            # Acceptance criterion
            if delta < 0 or rng.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                step_count += 1
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        elif move_type < 0.8:
            # Insert: move element at position i to position j
            i, j = rng.choice(n, size=2, replace=False)
            if i == j:
                continue
            # Compute delta for moving p[i] to position j
            # This is more complex; use full recomputation for simplicity
            old_p = p.copy()
            # Remove from i and insert at j
            if i < j:
                p[i:j] = p[i+1:j+1]
                p[j] = old_p[i]
            else:
                p[i+1:j+1] = p[i:j]
                p[i] = old_p[i]
            
            new_cost = compute_cost(p)
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_cost = new_cost
                step_count += 1
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            else:
                p = old_p  # revert
        else:
            # Block move: reverse a segment
            i, j = sorted(rng.choice(n, size=2, replace=False))
            if i == j:
                continue
            old_p = p.copy()
            p[i:j+1] = p[i:j+1][::-1]
            
            new_cost = compute_cost(p)
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_cost = new_cost
                step_count += 1
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            else:
                p = old_p  # revert
        
        # Cool down
        T = max(T_end, T * alpha)
        
        # Periodic local search refinement
        if step_count - last_ls_step >= 500 and time.time() < deadline - 15:
            last_ls_step = step_count
            p, current_cost = local_search(p, current_cost, max_passes=5)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Occasional Hungarian re-optimization to escape local optima
        if step_count % 2000 == 0 and time.time() < deadline - 30:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            if h_cost < best_cost:
                best_cost = h_cost
                best_p = h_p.copy()
                p = h_p.copy()
                current_cost = h_cost
                p, current_cost = local_search(p, current_cost, max_passes=10)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()

    # Final local search polish
    if time.time() < deadline - 5:
        best_p, best_cost = local_search(best_p.copy(), best_cost, max_passes=20)

    return best_p.tolist()
