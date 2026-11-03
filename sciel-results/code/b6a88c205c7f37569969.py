"""QAP solver: spectral/Hungarian init + fast incremental 2-swap + 3-swap + multi-start."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Multiple diverse initializations (rank-matching, spectral, Hungarian linearization).
    2. Fast local search with O(n) incremental updates per swap (no full recomputation).
    3. 3-cycle moves to escape 2-swap local optima.
    4. Multi-start with perturbation if time permits.
    """
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s limit

    def elapsed():
        return time.time() - t0

    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
        total = 0.0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total

    # --- Precomputation helpers ---
    def precompute(p):
        """Precompute dot_row and dot_col for fast swap deltas.
        
        dot_row[i, l] = sum_k a[i,k] * b[l, p[k]]  = (a @ b[:, p].T)[i, l]
        dot_col[i, l] = sum_k a[k,i] * b[p[k], l]  = (a.T @ b[p, :])[i, l]
        """
        b_col_p = b[:, p]   # (n, n): [l, k] = b[l, p[k]]
        b_row_p = b[p, :]   # (n, n): [k, l] = b[p[k], l]
        dot_row = a @ b_col_p.T    # (n, n)
        dot_col = a.T @ b_row_p    # (n, n)
        return dot_row, dot_col

    def update_precompute_swap(p, dot_row, dot_col, i, j):
        """Incrementally update dot_row and dot_col after swapping p[i] and p[j]. O(n)."""
        u = p[i]
        v = p[j]
        
        # For all k, the change in dot_row[k, l]:
        # dot_row[k, l] = sum_m a[k,m] * b[l, p[m]]
        # After swap: p[i]=v, p[j]=u
        # Change for row k: a[k,i]*(b[l,v]-b[l,u]) + a[k,j]*(b[l,u]-b[l,v])
        #                   = (a[k,i] - a[k,j]) * (b[l,v] - b[l,u])
        
        # diff_row[l] = b[l,v] - b[l,u]
        diff_row = b[:, v] - b[:, u]  # (n,)
        
        # For all k: delta_row[k, l] = (a[k,i] - a[k,j]) * diff_row[l]
        # This is an outer product: (a[:,i] - a[:,j])[:,None] * diff_row[None,:]
        delta_row = (a[:, i] - a[:, j])[:, None] * diff_row[None, :]
        dot_row += delta_row
        
        # For all k, the change in dot_col[k, l]:
        # dot_col[k, l] = sum_m a[m,k] * b[p[m], l]
        # After swap: p[i]=v, p[j]=u
        # Change for col k: a[i,k]*(b[v,l]-b[u,l]) + a[j,k]*(b[u,l]-b[v,l])
        #                   = (a[i,k] - a[j,k]) * (b[v,l] - b[u,l])
        
        # diff_col[l] = b[v,l] - b[u,l]
        diff_col = b[v, :] - b[u, :]  # (n,)
        
        # For all k: delta_col[k, l] = (a[i,k] - a[j,k]) * diff_col[l]
        delta_col = (a[i, :] - a[j, :])[:, None] * diff_col[None, :]
        dot_col += delta_col
        
        return dot_row, dot_col

    def swap_delta(i, j, p, dot_row, dot_col):
        """Compute delta for swapping positions i and j. O(1) with precomputed arrays."""
        u = p[i]
        v = p[j]
        if u == v:
            return 0.0
        
        buu = b[u, u]
        bvv = b[v, v]
        buv = b[u, v]
        bvu = b[v, u]
        
        delta = 0.0
        # Diagonal and cross terms
        delta += a[i, i] * (bvv - buu)
        delta += a[i, j] * (bvu - buv)
        delta += a[j, i] * (buv - bvu)
        delta += a[j, j] * (buu - bvv)
        
        # Row terms: sum_{k not in {i,j}} a[i,k] * (b[v, p[k]] - b[u, p[k]])
        # = dot_row[i, v] - dot_row[i, u] - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])
        delta += (dot_row[i, v] - dot_row[i, u]) - a[i, i] * (bvv - buu) - a[i, j] * (bvu - buv)
        
        # Row terms for j
        delta += (dot_row[j, u] - dot_row[j, v]) - a[j, i] * (buv - bvu) - a[j, j] * (buu - bvv)
        
        # Col terms: sum_{k not in {i,j}} a[k,i] * (b[p[k], v] - b[p[k], u])
        # = dot_col[i, v] - dot_col[i, u] - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[v,u]-b[u,v])
        delta += (dot_col[i, v] - dot_col[i, u]) - a[i, i] * (bvv - buu) - a[j, i] * (bvu - buv)
        
        # Col terms for j
        delta += (dot_col[j, u] - dot_col[j, v]) - a[i, j] * (buv - bvu) - a[j, j] * (buu - bvv)
        
        return delta

    # --- Initialization methods ---
    def init_rank_matching():
        """Simple rank-matching initialization."""
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def init_spectral():
        """Spectral initialization using eigenvectors."""
        try:
            a_sym = (a + a.T) / 2.0
            b_sym = (b + b.T) / 2.0
            
            # Use leading eigenvectors
            evals_a, evecs_a = np.linalg.eigh(a_sym)
            evals_b, evecs_b = np.linalg.eigh(b_sym)
            
            idx_a = np.argsort(-evals_a)
            idx_b = np.argsort(evals_b)
            
            # Project onto leading eigenvectors
            spec_a = evecs_a[:, idx_a[0]]
            spec_b = evecs_b[:, idx_b[0]]
            
            fac_rank = np.argsort(spec_a, kind="stable")
            loc_rank = np.argsort(spec_b, kind="stable")
            
            p = np.empty(n, dtype=np.int64)
            p[fac_rank] = loc_rank
            return p
        except Exception:
            return init_rank_matching()

    def init_hungarian():
        """Hungarian-based initialization from linearized cost."""
        p_init = init_rank_matching()
        b_col_p = b[:, p_init]
        b_row_p = b[p_init, :]
        
        C = a @ b_col_p.T + a.T @ b_row_p
        
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def init_hungarian_iterative(num_rounds=3):
        """Iterative Hungarian refinement for better initialization."""
        p = init_rank_matching()
        for _ in range(num_rounds):
            b_col_p = b[:, p]
            b_row_p = b[p, :]
            
            C = a @ b_col_p.T + a.T @ b_row_p
            
            row_ind, col_ind = linear_sum_assignment(C)
            p_new = np.empty(n, dtype=np.int64)
            p_new[row_ind] = col_ind
            
            if np.array_equal(p, p_new):
                break
            p = p_new
        return p

    # --- Local search with fast incremental updates ---
    def local_search_2swap(p, cost, time_limit):
        """Hill-climbing 2-swap local search with O(n) incremental updates."""
        p = p.copy()
        dot_row, dot_col = precompute(p)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            num_trials = 50000
            for _ in range(num_trials):
                if time.time() > time_limit:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                
                d = swap_delta(i, j, p, dot_row, dot_col)
                if d < -1e-9:
                    u, v = p[i], p[j]
                    p[i], p[j] = v, u
                    cost += d
                    dot_row, dot_col = update_precompute_swap(p, dot_row, dot_col, i, j)
                    improved = True
        
        return p, cost

    # --- 3-cycle moves ---
    def swap3_delta(i, j, k, p, dot_row, dot_col):
        """Compute delta for 3-cycle: p[i]->p[j]->p[k]->p[i]. O(n)."""
        u, v, w = p[i], p[j], p[k]
        if u == v or v == w or u == w:
            return 0.0
        
        # Compute the change by considering the three positions
        # New assignment: p[i]=v, p[j]=w, p[k]=u
        
        delta = 0.0
        
        # Terms involving i, j, k with each other
        # a[i,i]: b[v,v] - b[u,u]
        delta += a[i, i] * (b[v, v] - b[u, u])
        # a[i,j]: b[v,w] - b[u,v]
        delta += a[i, j] * (b[v, w] - b[u, v])
        # a[i,k]: b[v,u] - b[u,w]
        delta += a[i, k] * (b[v, u] - b[u, w])
        
        # a[j,i]: b[w,u] - b[v,u]
        delta += a[j, i] * (b[w, u] - b[v, u])
        # a[j,j]: b[w,w] - b[v,v]
        delta += a[j, j] * (b[w, w] - b[v, v])
        # a[j,k]: b[w,u] - b[v,w]
        delta += a[j, k] * (b[w, u] - b[v, w])
        
        # a[k,i]: b[u,v] - b[w,v]
        delta += a[k, i] * (b[u, v] - b[w, v])
        # a[k,j]: b[u,w] - b[w,w]
        delta += a[k, j] * (b[u, w] - b[w, w])
        # a[k,k]: b[u,u] - b[w,w]
        delta += a[k, k] * (b[u, u] - b[w, w])
        
        # Terms with other positions m not in {i,j,k}
        # For each m, the change involves a[i,m], a[j,m], a[k,m] and their column counterparts
        
        mask = np.ones(n, dtype=bool)
        mask[i] = mask[j] = mask[k] = False
        others = np.where(mask)[0]
        
        if len(others) > 0:
            # Row terms: a[x,m] * (b[new_p[x], p[m]] - b[old_p[x], p[m]])
            # For x=i: new=v, old=u -> a[i,m]*(b[v,p[m]] - b[u,p[m]])
            # For x=j: new=w, old=v -> a[j,m]*(b[w,p[m]] - b[v,p[m]])
            # For x=k: new=u, old=w -> a[k,m]*(b[u,p[m]] - b[w,p[m]])
            
            b_v_others = b[v, p[others]]  # (len(others),)
            b_u_others = b[u, p[others]]
            b_w_others = b[w, p[others]]
            
            delta += np.dot(a[i, others], b_v_others - b_u_others)
            delta += np.dot(a[j, others], b_w_others - b_v_others)
            delta += np.dot(a[k, others], b_u_others - b_w_others)
            
            # Col terms: a[m,x] * (b[p[m], new_p[x]] - b[p[m], old_p[x]])
            # For x=i: new=v, old=u -> a[m,i]*(b[p[m],v] - b[p[m],u])
            # For x=j: new=w, old=v -> a[m,j]*(b[p[m],w] - b[p[m],v])
            # For x=k: new=u, old=w -> a[m,k]*(b[p[m],u] - b[p[m],w])
            
            b_others_v = b[p[others], v]  # (len(others),)
            b_others_u = b[p[others], u]
            b_others_w = b[p[others], w]
            
            delta += np.dot(a[others, i], b_others_v - b_others_u)
            delta += np.dot(a[others, j], b_others_w - b_others_v)
            delta += np.dot(a[others, k], b_others_u - b_others_w)
        
        return delta

    def local_search_3swap(p, cost, time_limit):
        """Try 3-cycle moves to escape 2-swap local optima."""
        p = p.copy()
        dot_row, dot_col = precompute(p)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            num_trials = 10000
            for _ in range(num_trials):
                if time.time() > time_limit:
                    break
                i, j, k = np.random.randint(0, n, 3)
                if len(set([i, j, k])) < 3:
                    continue
                
                d = swap3_delta(i, j, k, p, dot_row, dot_col)
                if d < -1e-9:
                    u, v, w = p[i], p[j], p[k]
                    p[i], p[j], p[k] = v, w, u
                    cost += d
                    # Recompute precomputation (simpler for 3-swap)
                    dot_row, dot_col = precompute(p)
                    improved = True
        
        return p, cost

    # --- Hungarian refinement ---
    def hungarian_refine(p, cost, time_limit):
        """Use Hungarian on linearized cost to improve solution."""
        if time.time() > time_limit:
            return p, cost
        
        b_col_p = b[:, p]
        b_row_p = b[p, :]
        
        C = a @ b_col_p.T + a.T @ b_row_p
        
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        
        cost_hung = compute_cost(p_hung)
        if cost_hung < cost:
            p = p_hung
            cost = cost_hung
        
        return p, cost

    # --- Perturbation for multi-start ---
    def perturb(p, num_swaps=5):
        """Perturb a solution by performing random swaps."""
        p = p.copy()
        for _ in range(num_swaps):
            i, j = np.random.randint(0, n, 2)
            if i != j:
                p[i], p[j] = p[j], p[i]
        return p

    # --- Main solver ---
    best_p = None
    best_cost = np.inf
    
    # Generate diverse initial solutions
    init_methods = [
        init_rank_matching,
        init_spectral,
        init_hungarian,
        init_hungarian_iterative,
    ]
    
    # Time allocation: give each initial solution a budget for local search
    num_starts = len(init_methods)
    # Reserve time for final polish
    final_polish_time = 10.0
    available_time = deadline - time.time() - final_polish_time
    if available_time < 0:
        available_time = 0
    
    time_per_start = max(available_time / num_starts, 5.0)
    
    for method_idx, init_method in enumerate(init_methods):
        if time.time() > deadline - final_polish_time:
            break
        
        start_time = time.time()
        p = init_method()
        cost = compute_cost(p)
        
        # Phase 1: 2-swap local search
        time_limit_2swap = start_time + time_per_start * 0.6
        if time.time() < time_limit_2swap:
            p, cost = local_search_2swap(p, cost, time_limit_2swap)
        
        # Phase 2: Hungarian refinement
        time_limit_hung = start_time + time_per_start * 0.75
        if time.time() < time_limit_hung:
            p, cost = hungarian_refine(p, cost, time_limit_hung)
        
        # Phase 3: More 2-swap local search
        time_limit_2swap2 = start_time + time_per_start * 0.9
        if time.time() < time_limit_2swap2:
            p, cost = local_search_2swap(p, cost, time_limit_2swap2)
        
        # Phase 4: 3-swap moves
        time_limit_3swap = start_time + time_per_start
        if time.time() < time_limit_3swap:
            p, cost = local_search_3swap(p, cost, time_limit_3swap)
        
        # Update best
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    # Final polish with remaining time
    if best_p is not None and time.time() < deadline - 2.0:
        time_limit_final = deadline - 2.0
        best_p, best_cost = local_search_2swap(best_p, best_cost, time_limit_final)
    
    if best_p is None:
        best_p = init_rank_matching()
        best_cost = compute_cost(best_p)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
