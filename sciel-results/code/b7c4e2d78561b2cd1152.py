# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute row sums for initial heuristic
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    # Initial rank-matching: heavy flows to central locations
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()

    def compute_cost(p):
        """Compute exact QAP cost."""
        return np.sum(a * b[np.ix_(p, p)])

    def two_opt_delta(p, i, j):
        """
        Compute delta cost for swapping facilities i and j in assignment p.
        Returns negative delta if swap improves (reduces) cost.
        """
        li = p[i]
        lj = p[j]
        
        d = b[lj, p] - b[li, p]
        d[i] = 0
        d[j] = 0
        
        a_diff = a[i, :] - a[j, :] + a[:, i] - a[:, j]
        a_diff[i] = 0
        a_diff[j] = 0
        
        return np.dot(a_diff, d)

    def two_opt_search(p, max_time=None):
        """Perform 2-opt local search starting from assignment p."""
        if max_time is None:
            max_time = deadline - 5.0
        
        best_p = p.copy()
        best_cost = compute_cost(best_p)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n):
                if time.time() > max_time:
                    break
                for j in range(i+1, n):
                    delta = two_opt_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        best_cost += delta
                        best_p = p.copy()
                        improved = True
                        break
            # Continue scanning from where we left off (first-improvement style)
        return best_p, best_cost

    def frank_wolfe(p_init, max_iters=20, tol=1e-6):
        """
        Frank-Wolfe algorithm on the Birkhoff polytope for QAP.
        Uses Hungarian algorithm for projection onto permutation polytope.
        """
        p = p_init.copy()
        X = np.zeros((n, n), dtype=np.float64)
        X[np.arange(n), p] = 1.0
        
        best_p = p.copy()
        best_cost = compute_cost(best_p)
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            
            # Compute gradient of the QAP objective w.r.t. X
            # f(X) = sum_{i,j} a[i,j] * (X @ B @ X.T)[i,j]
            # grad[i,j] = 2 * sum_k sum_l a[i,k] * B[k,j] * X[l,k] * ... 
            # Actually, for QAP: f(X) = trace(X^T A X B) = sum a[i,j] X[i,k] X[j,l] B[k,l]
            # Gradient: G[i,j] = 2 * (A @ X @ B)[i,j] + 2 * (A.T @ X @ B.T)[i,j]
            # But we need to be careful. Let's use:
            # f(X) = sum_{i,j,k,l} a[i,j] * X[i,k] * X[j,l] * B[k,l]
            # df/dX[p,q] = sum_{j,l} a[p,j] * X[j,l] * B[q,l] + sum_{i,k} a[i,p] * X[i,k] * B[k,q]
            #           = (A @ X @ B)[p,q] + (A.T @ X @ B.T)[p,q]
            #           = (A @ X @ B)[p,q] + ((A.T @ X @ B.T)[p,q])
            # Note: (A.T @ X @ B.T)[p,q] = (B @ X.T @ A)[q,p].T... let me just compute directly.
            
            # G = A @ X @ B + A.T @ X @ B.T
            # But B is symmetric for QAPLIB distance matrices, so B.T = B
            # G = A @ X @ B + A.T @ X @ B
            
            G = a @ X @ b + a.T @ X @ b
            
            # Find the vertex of the Birkhoff polytope that minimizes <G, Y>
            # This is the Hungarian algorithm on cost matrix G
            row_ind, col_ind = linear_sum_assignment(G)
            Y = np.zeros((n, n), dtype=np.float64)
            Y[row_ind, col_ind] = 1.0
            
            # Check if Y == X (convergence)
            if np.allclose(X, Y, atol=1e-10):
                break
            
            # Line search: minimize f(X + alpha*(Y-X)) for alpha in [0,1]
            # f is quadratic in alpha, so we can do exact line search
            # f(X + alpha*Y) where Y here means the direction D = Y - X
            D = Y - X
            
            # f(alpha) = f(X + alpha*D)
            # = sum a[i,j] * (X[i,k] + alpha*D[i,k]) * (X[j,l] + alpha*D[j,l]) * B[k,l]
            # = f(X) + alpha * sum a[i,j] (D[i,k]X[j,l] + X[i,k]D[j,l]) B[k,l]
            #   + alpha^2 * sum a[i,j] D[i,k]D[j,l] B[k,l]
            
            # c1 = <G, D> where G is the gradient
            c1 = np.sum(G * D)
            
            # c2 = D^T (A ⊗ B) D... actually c2 = sum a[i,j] D[i,k] D[j,l] B[k,l]
            # = trace(D^T A D B)
            c2 = np.sum(a * D.T @ D @ b)  # This isn't right dimensionally
            # Let me compute c2 properly:
            # c2 = sum_{i,j,k,l} a[i,j] * D[i,k] * D[j,l] * B[k,l]
            # = sum_{k,l} B[k,l] * (sum_i a[i,k] D[i,k])... no
            # Let M = D @ B @ D.T
            # c2 = sum_{i,j} a[i,j] * M[i,j] = np.sum(a * (D @ b @ D.T))
            c2 = np.sum(a * (D @ b @ D.T))
            
            # f(alpha) = f0 + c1*alpha + c2*alpha^2
            # Minimum at alpha = -c1/(2*c2) if c2 > 0, else alpha=1
            if c2 > 1e-12:
                alpha = -c1 / (2.0 * c2)
                alpha = np.clip(alpha, 0.0, 1.0)
            else:
                alpha = 1.0 if c1 < 0 else 0.0
            
            if alpha < 1e-8:
                break
            
            X = X + alpha * D
            
            # Project back to permutation (extract best permutation from X)
            # Use Hungarian on -X to find the closest permutation
            row_ind, col_ind = linear_sum_assignment(-X)
            p_new = col_ind.copy()  # p[i] = col_ind[i] means facility i goes to location col_ind[i]
            
            cost_new = compute_cost(p_new)
            if cost_new < best_cost:
                best_cost = cost_new
                best_p = p_new.copy()
            
            p = p_new
            X = np.zeros((n, n), dtype=np.float64)
            X[np.arange(n), p] = 1.0
        
        return best_p, best_cost

    best_p = p.copy()
    best_cost = compute_cost(best_p)
    
    # Phase 1: 2-opt local search from initial heuristic
    best_p, best_cost = two_opt_search(best_p, max_time=deadline - 60.0)
    
    # Phase 2: Frank-Wolfe large moves from current best, with 2-opt polish
    remaining_time = deadline - time.time()
    if remaining_time > 30.0:
        # Try FW from the current best solution
        fw_p, fw_cost = frank_wolfe(best_p, max_iters=15)
        if fw_cost < best_cost:
            best_p = fw_p.copy()
            best_cost = fw_cost
        
        # Polish with 2-opt
        remaining_time = deadline - time.time()
        if remaining_time > 10.0:
            best_p, best_cost = two_opt_search(best_p, max_time=deadline - 5.0)
        
        # Try another FW restart from a perturbed version
        remaining_time = deadline - time.time()
        if remaining_time > 30.0:
            # Perturb: swap a few random pairs
            p_pert = best_p.copy()
            rng = np.random.default_rng(42)
            for _ in range(5):
                i, j = rng.choice(n, size=2, replace=False)
                p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
            fw_p2, fw_cost2 = frank_wolfe(p_pert, max_iters=10)
            if fw_cost2 < best_cost:
                best_p = fw_p2.copy()
                best_cost = fw_cost2
            
            # Final polish
            remaining_time = deadline - time.time()
            if remaining_time > 10.0:
                best_p, best_cost = two_opt_search(best_p, max_time=deadline - 5.0)
    
    # Final check
    if time.time() < deadline - 5.0:
        final_p, final_cost = two_opt_search(best_p, max_time=deadline - 5.0)
        if final_cost < best_cost:
            best_p = final_p
            best_cost = final_cost
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
