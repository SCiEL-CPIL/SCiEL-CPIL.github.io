import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """
    Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Spectral (eigenvalue) initialization via Procrustes alignment.
    2. Fast vectorized 2-swap local search with precomputed row/col sums.
    3. Multiple restarts with perturbation to escape local minima.
    4. Hungarian linearization as a global move between restarts.
    5. Final polish with tight time budget.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    b_col_sums = b.sum(axis=0)

    def compute_cost(p):
        """Compute exact QAP cost for permutation p."""
        return np.sum(a * b[np.ix_(p, p)])

    def spectral_init():
        """
        Initialize permutation using spectral embedding / Procrustes alignment.
        Uses the fact that the QAP can be approximated by aligning the 
        eigenvector bases of a and b.
        """
        # Center the matrices for better spectral properties
        a_c = a - a.mean()
        b_c = b - b.mean()
        
        # Get eigen-decompositions
        try:
            evals_a, evecs_a = np.linalg.eigh(a_c)
            evals_b, evecs_b = np.linalg.eigh(b_c)
        except:
            evals_a, evecs_a = np.linalg.eig(a_c)
            evals_b, evecs_b = np.linalg.eig(b_c)
        
        # Sort by eigenvalue magnitude (descending) for alignment
        idx_a = np.argsort(-np.abs(evals_a))
        idx_b = np.argsort(-np.abs(evals_b))
        evecs_a = evecs_a[:, idx_a]
        evecs_b = evecs_b[:, idx_b]
        
        # Procrustes: find optimal rotation R such that evecs_a @ R ≈ evecs_b
        M = evecs_a.T @ evecs_b
        U, S, Vt = np.linalg.svd(M)
        R = U @ Vt
        
        # Project: X = evecs_a @ R should align with evecs_b
        X = evecs_a @ R  # facility embeddings in aligned space
        
        # For each facility, find the location whose embedding is closest
        C = -X @ evecs_b.T  # negative distance (we want to minimize)
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p

    def rank_init():
        """Rank-based initialization: match facilities by flow strength to locations by distance centrality."""
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def hungarian_init(p_current):
        """Hungarian linearization given current permutation."""
        # C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # This is a@b[:,p].T
        B_p = b[:, p_current]
        C = a @ B_p.T
        row_ind, col_ind = linear_sum_assignment(C)
        p_h = np.empty(n, dtype=np.int64)
        p_h[row_ind] = col_ind
        return p_h

    def swap_delta(i, j, p):
        """
        Compute the exact change in cost if we swap facilities i and j.
        delta = new_cost - old_cost
        """
        u = p[i]
        v = p[j]
        
        # The change in cost comes from terms involving i or j.
        # Old terms:
        # sum_k a[i,k] b[u, p[k]] + sum_k a[k,i] b[p[k], u]
        # sum_k a[j,k] b[v, p[k]] + sum_k a[k,j] b[p[k], v]
        
        # New terms:
        # sum_k a[i,k] b[v, p[k]] + sum_k a[k,i] b[p[k], v]
        # sum_k a[j,k] b[u, p[k]] + sum_k a[k,j] b[p[k], u]
        
        # Let's compute the difference.
        # Delta_row_i = sum_k a[i,k] (b[v, p[k]] - b[u, p[k]])
        # Delta_col_i = sum_k a[k,i] (b[p[k], v] - b[p[k], u])
        # Delta_row_j = sum_k a[j,k] (b[u, p[k]] - b[v, p[k]])
        # Delta_col_j = sum_k a[k,j] (b[p[k], u] - b[p[k], v])
        
        # We can vectorize this.
        # Note that p[i]=u and p[j]=v.
        
        # Vectorized computation:
        # diff_b_row = b[v, :] - b[u, :]
        # diff_b_col = b[:, v] - b[:, u]
        
        # Delta_row_i = a[i, :] dot diff_b_row[p]
        # Delta_col_i = a[:, i] dot diff_b_col[p]
        # Delta_row_j = a[j, :] dot diff_b_row[p]  (note sign flip in diff) -> actually b[u]-b[v]
        # Let's just use the explicit formula for safety and speed.
        
        # delta = (a[i,:] - a[j,:]) dot (b[v,p] - b[u,p]) + (a[:,i] - a[:,j]) dot (b[:,v] - b[:,u])[p]
        # This formula works for k != i, j.
        # We need to correct for k=i and k=j.
        
        # Let's stick to the robust O(n) computation derived in the previous thought process.
        # It was:
        # row_part = np.dot(a[i] - a[j], (b[v] - b[u])[p])
        # col_part = np.dot(a[:, i] - a[:, j], (b[:, v] - b[:, u])[p])
        # Correction terms for the diagonal/cross interactions between i and j.
        
        buu = b[u, u]
        bvv = b[v, v]
        buv = b[u, v]
        bvu = b[v, u]
        
        # The "double counted" part in the dot products for k=i, j:
        # k=i: (a[i,i]-a[j,i])*(b[v,u]-b[u,u]) in row_part
        #      (a[i,i]-a[j,i])*(b[v,u]-b[u,u]) in col_part (since p[i]=u, b[:,v][u]=b[v,u], b[:,u][u]=b[u,u])
        #      Wait, col_part uses (b[:,v]-b[:,u])[p]. For k=i, p[i]=u.
        #      So term is (a[i,i]-a[j,i]) * (b[v,u] - b[u,u]).
        # k=j: (a[i,j]-a[j,j])*(b[v,v]-b[u,v]) in row_part
        #      (a[i,j]-a[j,j])*(b[v,v]-b[u,v]) in col_part (since p[j]=v, b[:,v][v]=b[v,v], b[:,u][v]=b[u,v])
        
        # So the sum of dot products includes:
        # 2 * (a[i,i]-a[j,i]) * (b[v,u] - b[u,u])
        # 2 * (a[i,j]-a[j,j]) * (b[v,v] - b[u,v])
        
        # The TRUE change for the 4 terms involving i,i; i,j; j,i; j,j:
        # Old: a[i,i]b[u,u] + a[i,j]b[u,v] + a[j,i]b[v,u] + a[j,j]b[v,v]
        # New: a[i,i]b[v,v] + a[i,j]b[v,u] + a[j,i]b[u,v] + a[j,j]b[u,u]
        # True_Delta_4 = a[i,i](b[v,v]-b[u,u]) + a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u]) + a[j,j](b[u,u]-b[v,v])
        #              = (a[i,i]-a[j,j])(b[v,v]-b[u,u]) - (a[i,j]+a[j,i])(b[v,u]-b[u,v])
        
        # So Delta = Dot_Sum - Double_Counted + True_Delta_4
        # Wait, the dot sums include the k=i,j terms.
        # The true delta is the change in the total sum.
        # The total sum change is sum_{k!=i,j} delta_k + delta_{i,i} + delta_{i,j} + delta_{j,i} + delta_{j,j}
        # The dot sum computes sum_{all k} (a[i,k]-a[j,k])b[v,p[k]] - ... which effectively computes
        # sum_{k!=i,j} ... + (terms for i,j).
        # The terms for i,j in the dot sum are exactly the "Double_Counted" part derived above.
        # So Dot_Sum = Sum_{k!=i,j} + Double_Counted.
        # We want Sum_{k!=i,j} + True_Delta_4.
        # So Delta = Dot_Sum - Double_Counted + True_Delta_4.
        
        row_part = np.dot(a[i] - a[j], (b[v] - b[u])[p])
        col_part = np.dot(a[:, i] - a[:, j], (b[:, v] - b[:, u])[p])
        
        double_counted = 2.0 * (a[i, i] - a[j, i]) * (bvu - buu) + 2.0 * (a[i, j] - a[j, j]) * (bvv - buv)
        
        true_block = (a[i, i] - a[j, j]) * (bvv - buu) - (a[i, j] + a[j, i]) * (bvu - buv)
        
        return row_part + col_part - double_counted + true_block

    def local_search_2swap(p, cost, time_limit, batch_size=300):
        """
        Fast 2-swap local search.
        """
        p = p.copy()
        improved = True
        no_improve = 0
        max_no_improve = 2000 # Reduced to allow more restarts
        
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0.0
            best_i, best_j = -1, -1
            
            for _ in range(batch_size):
                if time.time() > time_limit:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                u = p[i]
                v = p[j]
                if u == v:
                    continue
                
                delta = swap_delta(i, j, p)
                
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
            
            if best_i != -1 and best_delta < -1e-9:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += best_delta
                improved = True
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break

        return p, cost

    def perturb(p, num_swaps):
        """Randomly swap num_swaps pairs in permutation."""
        p = p.copy()
        for _ in range(num_swaps):
            i, j = np.random.randint(0, n, 2)
            if i != j:
                p[i], p[j] = p[j], p[i]
        return p

    # --- Generate initial solutions ---
    candidates = []
    
    # Spectral initialization
    p_spec = spectral_init()
    candidates.append(p_spec)
    
    # Rank initialization
    p_rank = rank_init()
    candidates.append(p_rank)
    
    # Hungarian from spectral
    p_hung1 = hungarian_init(p_spec)
    candidates.append(p_hung1)
    
    # Hungarian from rank
    p_hung2 = hungarian_init(p_rank)
    candidates.append(p_hung2)
    
    # Evaluate all candidates
    best_p = None
    best_cost = float('inf')
    for p in candidates:
        c = compute_cost(p)
        if c < best_cost:
            best_cost = c
            best_p = p.copy()
    
    # --- Main search loop with restarts ---
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    
    # Phase 1: Local search on best initial solution
    # Use a significant chunk of time here to find a good local optimum
    phase1_time = min(30.0, remaining * 0.3)
    if phase1_time > 2.0:
        best_p, best_cost = local_search_2swap(best_p, best_cost, time.time() + phase1_time)
    
    # Phase 2: Multiple restarts with perturbation + Hungarian
    # Use the remaining time for diversification
    num_restarts = 10
    restart_budget = (deadline - time.time()) * 0.7
    per_restart = restart_budget / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb current best
        # Adaptive perturbation: more swaps if we are stuck
        num_swaps = max(5, n // 10)
        p_pert = perturb(best_p, num_swaps)
        
        # Apply Hungarian linearization
        p_h = hungarian_init(p_pert)
        c_h = compute_cost(p_h)
        
        # Also try local search on perturbed directly
        # Allocate half the restart budget to local search on perturbed
        ls_time = time.time() + per_restart * 0.5
        p_ls, c_ls = local_search_2swap(p_pert, compute_cost(p_pert), ls_time)
        
        # Choose best
        if c_ls < c_h:
            p_new, c_new = p_ls, c_ls
        else:
            p_new, c_new = p_h, c_h
            # Quick local search on Hungarian result
            if time.time() < deadline - 8.0:
                hs_time = time.time() + per_restart * 0.3
                p_new, c_new = local_search_2swap(p_new, c_new, hs_time)
        
        if c_new < best_cost:
            best_cost = c_new
            best_p = p_new.copy()
            
            # If we found a new best, we might want to reduce perturbation strength?
            # Or keep it constant. Let's keep it constant for simplicity.
    
    # Phase 3: Final polish
    if time.time() < deadline - 3.0:
        best_p, best_cost = local_search_2swap(best_p, best_cost, deadline - 1.0, batch_size=200)
    
    # Final check
    if time.time() > deadline - 2.0:
        return best_p.tolist()
    
    return best_p.tolist()
