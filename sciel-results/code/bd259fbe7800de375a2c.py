import numpy as np
import time
import math

def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    def compute_cost(p):
        """Compute the exact integer cost of permutation p."""
        return int(np.sum(a * b[np.ix_(p, p)]))

    def compute_precomp(p):
        """
        Compute R and C matrices for O(1) swap delta evaluation.
        R[i, l] = sum_k A[i, k] * B[l, p[k]]
        C[i, l] = sum_k A[k, i] * B[p[k], l]
        """
        B_col_p = b[:, p]
        B_row_p = b[p, :]
        R = a @ B_col_p.T
        C = a.T @ B_row_p
        return R, C

    def update_precomp(R, C, i, k, j, l):
        """
        Update R and C after swapping p[i] and p[k].
        j = old p[i], l = old p[k].
        R += outer(A[:, i] - A[:, k], B[:, l] - B[:, j])
        C += outer(A[i, :] - A[k, :], B[l, :] - B[j, :])
        """
        R += np.outer(a[:, i] - a[:, k], b[:, l] - b[:, j])
        C += np.outer(a[i, :] - a[k, :], b[l, :] - b[j, :])
        return R, C

    def calc_swap_delta(i, k, p, R, C):
        """
        Calculate the cost delta for swapping facilities i and k.
        Returns delta (int).
        """
        j = p[i]
        l = p[k]
        
        if i == k:
            return 0
            
        # Self terms and cross terms between i and k
        # Old contribution involving i,k:
        # A[i,i]B[j,j] + A[k,k]B[l,l] + A[i,k]B[j,l] + A[k,i]B[l,j]
        # New contribution:
        # A[i,i]B[l,l] + A[k,k]B[j,j] + A[i,k]B[l,j] + A[k,i]B[j,l]
        
        d_self = (a[i, i] * b[l, l] + a[k, k] * b[j, j] + a[i, k] * b[l, j] + a[k, i] * b[j, l]) - \
                 (a[i, i] * b[j, j] + a[k, k] * b[l, l] + a[i, k] * b[j, l] + a[k, i] * b[l, j])
        
        # Terms involving other facilities v (v != i, k)
        # Delta_v = sum_{v != i, k} [ A[i,v](B[l,p[v]] - B[j,p[v]]) + A[v,i](B[p[v],l] - B[p[v],j])
        #                    + A[k,v](B[j,p[v]] - B[l,p[v]]) + A[v,k](B[p[v],j] - B[p[v],l]) ]
        
        # Using R and C:
        # R[i, l] - R[i, j] = sum_v A[i,v](B[l,p[v]] - B[j,p[v]])
        # We need to exclude v=i and v=k.
        # Term for v=i: A[i,i](B[l, p[i]] - B[j, p[i]]) = A[i,i](B[l, j] - B[j, j])
        # Term for v=k: A[i,k](B[l, p[k]] - B[j, p[k]]) = A[i,k](B[l, l] - B[j, l])
        
        # Similarly for C:
        # C[i, l] - C[i, j] = sum_v A[v,i](B[p[v],l] - B[p[v],j])
        # Term for v=i: A[i,i](B[p[i],l] - B[p[i],j]) = A[i,i](B[j, l] - B[j, j])
        # Term for v=k: A[k,i](B[p[k],l] - B[p[k],j]) = A[k,i](B[l, l] - B[l, j])
        
        # And for k:
        # R[k, j] - R[k, l] = sum_v A[k,v](B[j,p[v]] - B[l,p[v]])
        # Term for v=k: A[k,k](B[j, l] - B[l, l])
        # Term for v=i: A[k,i](B[j, j] - B[l, j])
        
        # C[k, j] - C[k, l] = sum_v A[v,k](B[p[v],j] - B[p[v],l])
        # Term for v=k: A[k,k](B[j, j] - B[l, l])
        # Term for v=i: A[i,k](B[j, j] - B[l, j])  # Wait, B[p[i],j] = B[j,j], B[p[i],l]=B[j,l]
        # Let's re-verify C[k,j] - C[k,l]:
        # C[k,y] = sum_v A[v,k] B[p[v],y]
        # C[k,j] - C[k,l] = sum_v A[v,k] (B[p[v],j] - B[p[v],l])
        # v=k: A[k,k](B[l,j] - B[l,l])  # p[k]=l
        # v=i: A[i,k](B[j,j] - B[j,l])  # p[i]=j
        
        term_Ri = (R[i, l] - R[i, j]) - a[i, i] * (b[l, j] - b[j, j]) - a[i, k] * (b[l, l] - b[j, l])
        term_Rk = (R[k, j] - R[k, l]) - a[k, k] * (b[j, l] - b[l, l]) - a[k, i] * (b[j, j] - b[l, j])
        
        term_Ci = (C[i, l] - C[i, j]) - a[i, i] * (b[j, l] - b[j, j]) - a[k, i] * (b[l, l] - b[l, j])
        term_Ck = (C[k, j] - C[k, l]) - a[k, k] * (b[j, j] - b[l, l]) - a[i, k] * (b[j, j] - b[j, l])
        
        d_v = term_Ri + term_Rk + term_Ci + term_Ck
        
        return d_self + d_v

    def local_search(p, current_cost, time_limit):
        """
        Deterministic 2-opt local search with incremental updates.
        """
        R, C = compute_precomp(p)
        improved = True
        passes = 0
        max_passes = 100  # Safety cap, though time limit is primary
        
        while improved and time.time() < time_limit and passes < max_passes:
            improved = False
            passes += 1
            # Iterate through all pairs
            # For n=150, 150*149/2 = 11175 pairs.
            # Each delta is O(1).
            # Update is O(n^2) = 22500 ops.
            # If we accept many swaps, it gets expensive.
            # But typically few swaps per pass.
            
            for i in range(n):
                if time.time() >= time_limit:
                    return p, current_cost
                for k in range(i + 1, n):
                    if time.time() >= time_limit:
                        return p, current_cost
                    
                    j = p[i]
                    l = p[k]
                    
                    delta = calc_swap_delta(i, k, p, R, C)
                    
                    if delta < 0:
                        # Accept swap
                        p[i], p[k] = p[k], p[i]
                        current_cost += delta
                        improved = True
                        
                        # Update precomputed matrices
                        R, C = update_precomp(R, C, i, k, j, l)
                        
        return p, current_cost

    def perturbation(p, strength=10):
        """
        Apply random 2-opt swaps to perturb the solution.
        """
        p = p.copy()
        for _ in range(strength):
            i = np.random.randint(0, n)
            k = np.random.randint(0, n)
            if i != k:
                p[i], p[k] = p[k], p[i]
        return p

    def sdp_relaxation_init(a, b, time_limit, n_iter=200):
        """
        Solve QAP relaxation on Birkhoff polytope via projected gradient descent.
        Returns a permutation obtained by spectral rounding of the relaxed solution.
        
        The QAP can be written as: min Tr(A X B X^T) s.t. X is doubly stochastic.
        We use projected gradient descent on the Birkhoff polytope.
        """
        # Initialize with a random doubly stochastic matrix
        np.random.seed(42)
        X = np.ones((n, n)) / n
        
        # Precompute for gradient: grad = 2 * (A @ X @ B + A.T @ X @ B.T)
        # Actually for QAP: f(X) = sum_{i,j,k,l} A[i,j] B[k,l] X[i,k] X[j,l]
        # grad[i,k] = sum_{j,l} A[i,j] B[k,l] X[j,l] + sum_{j,l} A[j,i] B[l,k] X[j,l]
        #            = (A @ X @ B)[i,k] + (A.T @ X @ B.T)[i,k]
        
        best_X = X.copy()
        best_cost = np.inf
        
        # Adaptive step size
        lr = 1.0 / (np.max(np.abs(a)) + np.max(np.abs(b))) * 0.5
        
        for it in range(n_iter):
            if time.time() >= time_limit:
                break
            
            # Compute gradient
            grad = a @ X @ b + a.T @ X @ b.T
            
            # Gradient step
            X_new = X - lr * grad
            
            # Project onto Birkhoff polytope (doubly stochastic)
            X_new = _project_birkhoff(X_new)
            
            # Compute objective
            cost = np.trace(a @ X_new @ b @ X_new.T)
            if cost < best_cost:
                best_cost = cost
                best_X = X_new.copy()
            
            # Adaptive learning rate
            if it > 0 and cost > best_cost * 1.1:
                lr *= 0.5
            elif it > 0 and cost < best_cost * 0.99:
                lr *= 1.1
                lr = min(lr, 1.0 / (np.max(np.abs(a)) + np.max(np.abs(b))))
            
            X = X_new
        
        # Spectral rounding: use the leading eigenvectors of the relaxed solution
        p = _spectral_round(best_X)
        return p, compute_cost(p)

    def _project_birkhoff(X, tol=1e-6, max_iter=100):
        """
        Project a matrix onto the Birkhoff polytope (doubly stochastic matrices)
        using the Sinkhorn-Knopp algorithm.
        """
        X = np.maximum(X, 1e-10)  # Ensure positive for log
        
        for _ in range(max_iter):
            # Row normalization
            row_sums = X.sum(axis=1, keepdims=True)
            X = X / row_sums
            
            # Column normalization
            col_sums = X.sum(axis=0, keepdims=True)
            X = X / col_sums
            
            # Check convergence
            row_diff = np.abs(row_sums - 1.0).max()
            col_diff = np.abs(col_sums - 1.0).max()
            if max(row_diff, col_diff) < tol:
                break
        
        return X

    def _spectral_round(X, n):
        """
        Convert a doubly stochastic matrix to a permutation using spectral methods.
        Uses the fact that permutation matrices are extreme points of Birkhoff polytope.
        We use a greedy assignment based on the strongest entries, refined by
        checking consistency with the relaxation.
        """
        # First try: Hungarian algorithm on the relaxed matrix (exact for assignment)
        try:
            from scipy.optimize import linear_sum_assignment
            # Minimize -X to maximize correlation with the relaxed solution
            row_ind, col_ind = linear_sum_assignment(-X)
            p = np.empty(n, dtype=np.int64)
            p[row_ind] = col_ind
            return p
        except:
            pass
        
        # Fallback: greedy assignment
        p = np.empty(n, dtype=np.int64)
        used_cols = np.zeros(n, dtype=bool)
        for i in range(n):
            # Find best unassigned column
            candidates = np.where(~used_cols)[0]
            if len(candidates) == 0:
                p[i] = i
                used_cols[i] = True
                continue
            best_col = candidates[np.argmax(X[i, candidates])]
            p[i] = best_col
            used_cols[best_col] = True
        return p

    # --- Phase 1: Constructive Heuristic ---
    # Use rank matching as a fast, robust initial solution.
    # Sort facilities by row sum (heavy first) and locations by row sum (central first).
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    current_cost = compute_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # --- Phase 2: Local Search from Initial Solution ---
    if time.time() < deadline - 10.0:
        p, current_cost = local_search(p, current_cost, deadline - 10.0)
        if current_cost < best_cost:
            best_p = p.copy()
            best_cost = current_cost

    # --- Phase 3: Spectral Initialization + Local Search ---
    # Try a different initial solution based on Linear Assignment Problem (LAP) relaxation.
    # Minimize sum A[i,j] B[p[i], p[j]] is hard.
    # Approximate by maximizing sum A[i,j] B[i,j]? No.
    # Use the idea: M = A @ B.T. Solve LAP on -M.
    if time.time() < deadline - 20.0:
        try:
            from scipy.optimize import linear_sum_assignment
            M = a @ b.T
            row_ind, col_ind = linear_sum_assignment(-M)
            p_spec = np.empty(n, dtype=np.int64)
            p_spec[row_ind] = col_ind
            cost_spec = compute_cost(p_spec)
            
            if cost_spec < best_cost:
                best_p = p_spec.copy()
                best_cost = cost_spec
            
            # Refine with local search
            if time.time() < deadline - 10.0:
                p_spec, cost_spec = local_search(p_spec, cost_spec, deadline - 10.0)
                if cost_spec < best_cost:
                    best_p = p_spec.copy()
                    best_cost = cost_spec
        except Exception:
            pass

    # --- Phase 3.5: SDP Relaxation Initialization + Local Search ---
    # Use SDP relaxation on Birkhoff polytope to get a high-quality continuous solution,
    # then round to a permutation and refine with local search.
    if time.time() < deadline - 15.0:
        try:
            p_sdp, cost_sdp = sdp_relaxation_init(a, b, deadline - 15.0, n_iter=150)
            
            if cost_sdp < best_cost:
                best_p = p_sdp.copy()
                best_cost = cost_sdp
            
            # Refine with local search
            if time.time() < deadline - 10.0:
                p_sdp, cost_sdp = local_search(p_sdp, cost_sdp, deadline - 10.0)
                if cost_sdp < best_cost:
                    best_p = p_sdp.copy()
                    best_cost = cost_sdp
        except Exception:
            pass

    # --- Phase 4: Iterated Local Search (ILS) with Simulated Annealing ---
    # Use the best solution found so far as the starting point.
    # Perturb, then local search. Accept worse solutions with SA probability.
    p_ils = best_p.copy()
    cost_ils = best_cost
    
    if time.time() < deadline - 5.0:
        no_improve_iter = 0
        max_no_improve = 100
        sa_temperature = max(1000.0, (best_cost - 8133398) * 0.1)  # Start temp based on gap
        sa_cooling = 0.95
        
        while time.time() < deadline - 5.0 and no_improve_iter < max_no_improve:
            # Perturb
            strength = 5 + no_improve_iter // 15
            p_pert = perturbation(p_ils, strength)
            
            # Local Search
            p_ls, cost_ls = local_search(p_pert, compute_cost(p_pert), deadline - 5.0)
            
            if cost_ls < best_cost:
                best_p = p_ls.copy()
                best_cost = cost_ls
                p_ils = p_ls.copy()
                no_improve_iter = 0
                sa_temperature *= sa_cooling
            else:
                # Simulated Annealing acceptance
                delta_cost = cost_ls - cost_ils
                if delta_cost < 0 or np.random.random() < math.exp(-delta_cost / max(sa_temperature, 1e-10)):
                    p_ils = p_ls.copy()
                    cost_ils = cost_ls
                no_improve_iter += 1
                sa_temperature *= sa_cooling
                sa_temperature = max(sa_temperature, 1.0)
                
                # Occasionally restart from best_p to avoid getting stuck in a bad basin
                if no_improve_iter % 8 == 0:
                    p_ils = best_p.copy()
                    cost_ils = best_cost

    return best_p.tolist()
