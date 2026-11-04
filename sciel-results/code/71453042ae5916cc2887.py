"""QAP solver: Simulated Annealing with large-step moves + Hungarian refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Spectral + Hungarian initial permutation.
    2. Simulated Annealing with large-step (non-adjacent) swaps.
    3. Periodic Hungarian linearization refinement.
    4. Final systematic neighborhood polish.
    """
    n = len(a)
    t_start = time.time()
    deadline = t_start + 145.0  # safety margin before 150s limit
    rng = np.random.RandomState(42)
    
    def compute_cost(p):
        """Compute exact QAP cost for permutation p."""
        total = 0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total
    
    def swap_delta(i, j, p):
        """Compute cost change for swapping facilities i and j in permutation p.
        
        Uses O(n) computation via precomputed row/col differences.
        """
        u = p[i]
        v = p[j]
        if u == v:
            return 0
        
        # Corner terms
        delta = (a[i, i] * (b[v, v] - b[u, u]) +
                 a[i, j] * (b[v, u] - b[u, v]) +
                 a[j, i] * (b[u, v] - b[v, u]) +
                 a[j, j] * (b[u, u] - b[v, v]))
        
        # Row i: sum_{k!=i,j} a[i,k] * (b[v,p[k]] - b[u,p[k]])
        br_diff = b[v] - b[u]
        bc_diff = b[:, v] - b[:, u]
        
        s_ri = np.dot(a[i], br_diff[p])
        delta += s_ri - a[i, i] * br_diff[u] - a[i, j] * br_diff[v]
        
        # Row j: sum_{k!=i,j} a[j,k] * (b[u,p[k]] - b[v,p[k]])
        s_rj = np.dot(a[j], br_diff[p])
        delta -= s_rj - a[j, i] * br_diff[u] - a[j, j] * br_diff[v]
        
        # Col i: sum_{k!=i,j} a[k,i] * (b[p[k],v] - b[p[k],u])
        s_ci = np.dot(a[:, i], bc_diff[p])
        delta += s_ci - a[i, i] * bc_diff[u] - a[j, i] * bc_diff[v]
        
        # Col j: sum_{k!=i,j} a[k,j] * (b[p[k],u] - b[p[k],v])
        s_cj = np.dot(a[:, j], bc_diff[p])
        delta -= s_cj - a[i, j] * bc_diff[u] - a[j, j] * bc_diff[v]
        
        return delta
    
    def hungarian_refine(p):
        """Apply Hungarian linearization to get a new permutation."""
        b_col_p = b[:, p]
        b_row_p = b[p, :]
        C = a @ b_col_p.T + a.T @ b_row_p
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        return p_new
    
    def local_search(p, cost, time_limit, max_iters=5000):
        """Fast hill climbing local search."""
        iters = 0
        while time.time() < time_limit and iters < max_iters:
            iters += 1
            improved = False
            # Try a batch of random swaps
            for _ in range(200):
                if time.time() > time_limit:
                    return p, cost
                i, j = rng.randint(0, n, 2)
                if i == j:
                    continue
                d = swap_delta(i, j, p)
                if d < -1e-9:
                    p[i], p[j] = p[j], p[i]
                    cost += d
                    improved = True
            if not improved:
                break
        return p, cost
    
    def systematic_polish(p, cost, time_limit):
        """Systematic 1-sweep: for each i, find best j to swap with."""
        while time.time() < time_limit:
            improved = False
            for i in range(n):
                if time.time() > time_limit:
                    return p, cost
                best_d = 0.0
                best_j = -1
                for j in range(i + 1, n):
                    d = swap_delta(i, j, p)
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j >= 0 and best_d < -1e-9:
                    p[i], p[best_j] = p[best_j], p[i]
                    cost += best_d
                    improved = True
            if not improved:
                break
        return p, cost
    
    # --- Initial solution ---
    # Spectral initialization
    try:
        eigvals, eigvecs = np.linalg.eigh(a)
        # Use top eigenvectors
        top_k = min(10, n)
        U = eigvecs[:, -top_k:]
        # Match to b's structure
        b_eigvals, b_eigvecs = np.linalg.eigh(b)
        V = b_eigvecs[:, -top_k:]
        # Compute similarity
        S = U.T @ V
        row_ind, col_ind = linear_sum_assignment(-S)
        p_spectral = np.empty(n, dtype=np.int64)
        # Map: facility i -> location based on eigenvector alignment
        # Simpler: use rank matching
    except:
        pass
    
    # Rank-matching initialization
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    a_order = np.argsort(-a_row_sums, kind="stable")
    b_order = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[a_order] = b_order
    
    # Hungarian refinement of initial
    p_hung = hungarian_refine(p_rank)
    cost_hung = compute_cost(p_hung)
    cost_rank = compute_cost(p_rank)
    
    if cost_hung < cost_rank:
        best_p = p_hung.copy()
        best_cost = cost_hung
    else:
        best_p = p_rank.copy()
        best_cost = cost_rank
    
    # Initial local search
    t_phase1_end = t_start + 30.0
    best_p, best_cost = local_search(best_p, best_cost, t_phase1_end, max_iters=20000)
    
    # --- Simulated Annealing with large-step moves ---
    # SA with geometric cooling
    T_initial = max(1.0, abs(best_cost) * 0.01)
    T_final = 1.0
    cooling_rate = (T_final / T_initial) ** (1.0 / 100000)  # ~100k steps
    
    T = T_initial
    sa_p = best_p.copy()
    sa_cost = best_cost
    sa_best_p = best_p.copy()
    sa_best_cost = best_cost
    
    sa_start = time.time()
    sa_deadline = t_start + 120.0  # Leave 25s for final polish
    
    step_count = 0
    no_improve_count = 0
    hungarian_interval = 50000  # Every 50k SA steps, try Hungarian
    
    while time.time() < sa_deadline:
        step_count += 1
        
        # Large-step move: swap two random positions
        i, j = rng.randint(0, n, 2)
        if i == j:
            continue
        
        d = swap_delta(i, j, sa_p)
        
        # Metropolis criterion
        if d < 0 or rng.rand() < np.exp(-d / T):
            sa_p[i], sa_p[j] = sa_p[j], sa_p[i]
            sa_cost += d
            
            if sa_cost < sa_best_cost:
                sa_best_cost = sa_cost
                sa_best_p = sa_p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_final:
            T = T_final
        
        # Periodic Hungarian refinement
        if step_count % hungarian_interval == 0 and time.time() < sa_deadline - 10:
            p_hung = hungarian_refine(sa_p)
            cost_hung = compute_cost(p_hung)
            if cost_hung < sa_best_cost:
                sa_best_cost = cost_hung
                sa_best_p = p_hung.copy()
                sa_p = p_hung.copy()
                sa_cost = cost_hung
                no_improve_count = 0
        
        # If stuck too long, restart SA from best
        if no_improve_count > 200000:
            sa_p = sa_best_p.copy()
            sa_cost = sa_best_cost
            T = T_initial * 0.1
            no_improve_count = 0
    
    # Update best
    if sa_best_cost < best_cost:
        best_p = sa_best_p
        best_cost = sa_best_cost
    
    # --- Final polish phase ---
    t_polish_start = time.time()
    polish_deadline = deadline - 5.0
    
    # First: local search
    if time.time() < polish_deadline - 15:
        best_p, best_cost = local_search(best_p, best_cost, polish_deadline - 15, max_iters=30000)
    
    # Second: Hungarian refinement
    if time.time() < polish_deadline - 10:
        p_hung = hungarian_refine(best_p)
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
        # Another local search round
        if time.time() < polish_deadline - 8:
            best_p, best_cost = local_search(best_p, best_cost, polish_deadline - 8, max_iters=20000)
    
    # Third: systematic polish
    if time.time() < polish_deadline - 2:
        best_p, best_cost = systematic_polish(best_p, best_cost, polish_deadline - 1)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
