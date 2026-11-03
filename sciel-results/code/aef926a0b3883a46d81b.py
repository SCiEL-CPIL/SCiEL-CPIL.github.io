"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    def project_birkhoff(X):
        """
        Project a matrix onto the Birkhoff polytope (doubly stochastic matrices).
        Uses the Sinkhorn-Knopp algorithm (iterative row/col normalization).
        """
        X = X.copy()
        # Ensure non-negative for Sinkhorn stability
        X = np.maximum(X, 1e-10)
        
        for _ in range(200): # Fixed iterations for stability, n=150 is small
            # Row normalization
            row_sums = X.sum(axis=1, keepdims=True)
            X = X / row_sums
            # Column normalization
            col_sums = X.sum(axis=0, keepdims=True)
            X = X / col_sums
        
        # Final adjustment to ensure exact 1s (Sinkhorn converges well)
        # A more robust projection might be needed, but for n=150 this is often sufficient for initialization
        return X

    def hungarian_solve(C):
        """
        Solve the linear assignment problem for cost matrix C.
        Returns the permutation p.
        """
        # linear_sum_assignment returns (row_ind, col_ind)
        # We want p[i] = location for facility i.
        # In standard LAP, we assign rows to cols.
        # Here, cost of assigning facility i to location j is C[i, j].
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))

    # 1. Continuous Relaxation Phase (PGD)
    # We minimize f(X) = sum_{i,j} a[i,j] X[i,j] ... wait.
    # The QAP cost is sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l].
    # Gradient wrt X[i,k]:
    # d/dX[i,k] sum_{i',j',k',l'} a[i',j'] b[k',l'] X[i',k'] X[j',l']
    # = sum_{j',l'} a[i,j'] b[k,l'] X[j',l'] + sum_{i',j'} a[i',j'] b[k',k] X[i',k']
    # Let's simplify.
    # Cost = Tr(X^T A X B) ? No.
    # Cost = sum_{i,j} a[i,j] b[p[i], p[j]].
    # If X is the permutation matrix where X[i, p[i]] = 1.
    # Cost = sum_{i,j,k,l} a[i,j] b[k,l] X[i,k] X[j,l].
    
    # Gradient G[i,k] = sum_j a[i,j] (sum_l b[k,l] X[j,l]) + sum_i' (sum_j a[i',j] X[i',k]) b[j,k] ?
    # Let's derive carefully.
    # C = sum_{i,j} a[i,j] sum_{k,l} b[k,l] X[i,k] X[j,l]
    # Partial derivative wrt X[p,q]:
    # Term 1: i=p, k=q. sum_j a[p,j] sum_l b[q,l] X[j,l] = (A X B^T)[p, q] ?
    #    (A X B^T)_{p,q} = sum_j A_{p,j} X_{j,q} ... no.
    #    Let's use matrix multiplication.
    #    C = Tr(X^T A X B) is not quite right for indices.
    #    C = sum_{i,j,k,l} A_{i,j} B_{k,l} X_{i,k} X_{j,l}
    #    C = sum_{i,k} X_{i,k} ( sum_{j,l} A_{i,j} X_{j,l} B_{k,l} )
    #    Let M = A X B^T. M_{i,k} = sum_j sum_l A_{i,j} X_{j,l} B_{k,l} = sum_j A_{i,j} (X B^T)_{j,k}.
    #    So C = sum_{i,k} X_{i,k} M_{i,k} = sum(X * M).
    #    Gradient G = M = A X B^T.
    #    Since A and B are symmetric, B^T = B.
    #    So Gradient G = A X B.
    
    # We will perform PGD on the Birkhoff polytope.
    # Initialize X with a spectral heuristic to provide a good starting point.
    try:
        w_a, v_a = np.linalg.eigh(a)
        w_b, v_b = np.linalg.eigh(b)
        # Use top eigenvectors
        idx_a = np.argmax(np.abs(w_a))
        idx_b = np.argmax(np.abs(w_b))
        vec_a = v_a[:, idx_a]
        vec_b = v_b[:, idx_b]
        if np.dot(vec_a, vec_b) < 0:
            vec_b = -vec_b
        
        # Create a rank-1 matrix X0 = vec_a vec_b^T
        X = np.outer(vec_a, vec_b)
        X = project_birkhoff(X)
    except Exception:
        # Fallback to uniform matrix
        X = np.ones((n, n)) / n

    best_p = None
    best_cost = np.inf
    
    # PGD Loop
    lr = 0.01 # Learning rate, needs to be small
    # We can adaptively adjust lr or just use a fixed one for simplicity in this budget
    # For n=150, the scale of gradients depends on matrix values.
    # Let's estimate gradient scale.
    grad_scale = np.linalg.norm(a) * np.linalg.norm(b)
    if grad_scale > 0:
        lr = 1.0 / (10.0 * grad_scale) # Heuristic
    
    # Precompute for speed? A and B are fixed.
    # In each step:
    # 1. Compute G = A X B
    # 2. X_new = X - lr * G
    # 3. Project X_new to Birkhoff
    # 4. Check time.
    
    # To speed up projection, we might use a faster algorithm or fewer iterations.
    # Sinkhorn is O(n^2 * iters). n=150 -> 22500 ops. 200 iters -> 4.5M ops.
    # 150s budget. We can do many steps.
    
    # However, the projection via Sinkhorn is the bottleneck.
    # Alternative: Use a different relaxation?
    # Or, run PGD for a while, then use the result to initialize a local search?
    # The prompt suggests a structural change. Pure PGD + Hungarian is a strong structural change.
    
    # Let's run PGD for a fixed number of steps or until time runs out.
    # We will periodically convert X to a permutation using Hungarian on the cost matrix derived from X?
    # No, standard approach:
    # Minimize f(X) on Birkhoff.
    # At the end (or periodically), round X to a permutation.
    # Rounding: Solve LAP with cost matrix C where C[i,j] is related to X?
    # Actually, if we have a doubly stochastic matrix X, the best permutation is found by
    # minimizing sum C[i,j] X[i,j]? No, that's the dual.
    # The standard way to recover a permutation from a relaxed solution X is to solve the LAP
    # with the objective that aligns with the gradient?
    # Actually, a common heuristic is:
    # 1. Get X (doubly stochastic).
    # 2. Construct a cost matrix C for LAP.
    #    If we minimize sum A[i,j] B[p[i],p[j]], and we have X approximating P,
    #    we can use the gradient G = A X B.
    #    The linear approximation of the cost around X is Tr(G^T (P - X)) + const.
    #    So we want to minimize Tr(G^T P) = sum G[i,j] P[i,j].
    #    So we solve LAP with cost matrix G.
    
    # Strategy:
    # 1. PGD to find good X.
    # 2. Solve LAP with cost matrix G = A X B.
    # 3. Evaluate cost of resulting permutation.
    # 4. If time remains, restart PGD with noise or different init, or do local search on the result.
    
    # Given the 150s budget, we can afford a few PGD runs.
    
    def run_pgd_and_round(X_init, max_time):
        X = X_init.copy()
        last_check = time.time()
        
        while time.time() < max_time:
            # Gradient
            G = a @ X @ b # A X B
            # Update
            X = X - lr * G
            # Project
            X = project_birkhoff(X)
            
            # Periodic rounding and checking
            if time.time() - last_check > 2.0:
                last_check = time.time()
                p = hungarian_solve(G) # Use current gradient as cost for LAP
                c = calc_cost(p)
                return p, c, X
        
        # Final rounding
        G = a @ X @ b
        p = hungarian_solve(G)
        c = calc_cost(p)
        return p, c, X

    # Run 1: Spectral Init
    try:
        w_a, v_a = np.linalg.eigh(a)
        w_b, v_b = np.linalg.eigh(b)
        idx_a = np.argmax(np.abs(w_a))
        idx_b = np.argmax(np.abs(w_b))
        vec_a = v_a[:, idx_a]
        vec_b = v_b[:, idx_b]
        if np.dot(vec_a, vec_b) < 0:
            vec_b = -vec_b
        X0 = np.outer(vec_a, vec_b)
        X0 = project_birkhoff(X0)
        
        p1, c1, X1 = run_pgd_and_round(X0, deadline - 5.0)
        if p1 is not None:
            best_p = p1
            best_cost = c1
    except Exception:
        pass

    # Run 2: Random Init (if time remains)
    if time.time() < deadline - 5.0 and best_p is not None:
        try:
            X2 = np.random.rand(n, n)
            X2 = project_birkhoff(X2)
            p2, c2, X2_final = run_pgd_and_round(X2, deadline - 5.0)
            if p2 is not None and c2 < best_cost:
                best_p = p2
                best_cost = c2
        except Exception:
            pass

    # If we have time, do a quick local search refinement on the best P found?
    # The prompt asks for a structural change. PGD+Hungarian is the main change.
    # Adding a tiny local search might help polish, but let's stick to the core new strategy first.
    # Actually, the "error_repair" idea in the prompt might imply fixing invalid states?
    # But PGD+Hungarian always produces a valid permutation.
    
    # Let's add a very fast 1-swap local search to polish the result if time allows.
    # This is a hybrid, but the *initialization* and *main search* are structural.
    
    if best_p is not None and time.time() < deadline - 2.0:
        p = best_p.copy()
        current_cost = best_cost
        Bp = b[:, p]
        
        # Simple 2-opt with time check
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            # Vectorized best move search is complex, let's do a random subset or full scan if fast enough
            # For n=150, full scan is 11175 pairs.
            # Dot product is O(n). 11175 * 150 = 1.6M ops. Very fast.
            
            best_delta = 0
            best_i, best_j = -1, -1
            
            for i in range(n):
                u = p[i]
                ai = a[i]
                for j in range(i + 1, n):
                    v = p[j]
                    aj = a[j]
                    # delta = (ai - aj) dot (Bp[v] - Bp[u])
                    delta = np.dot(ai - aj, Bp[v] - Bp[u])
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i != -1:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += int(best_delta)
                Bp = b[:, p] # Update Bp
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                improved = True
            else:
                improved = False

    if best_p is None:
        # Fallback
        return list(range(n))
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
