"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))

    def hungarian_solve(C):
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def project_birkhoff(X, iters=50):
        X = X.copy()
        X = np.maximum(X, 1e-10)
        for _ in range(iters):
            row_sums = X.sum(axis=1, keepdims=True)
            X = X / row_sums
            col_sums = X.sum(axis=0, keepdims=True)
            X = X / col_sums
        return X

    def spectral_init():
        """Spectral initialization using eigenvectors of A and B."""
        try:
            w_a, v_a = np.linalg.eigh(a)
            w_b, v_b = np.linalg.eigh(b)
            
            # Use top eigenvectors by absolute value
            idx_a = np.argmax(np.abs(w_a))
            idx_b = np.argmax(np.abs(w_b))
            vec_a = v_a[:, idx_a]
            vec_b = v_b[:, idx_b]
            
            # Ensure alignment
            if np.dot(vec_a, vec_b) < 0:
                vec_b = -vec_b
            
            X0 = np.outer(vec_a, vec_b)
            X0 = project_birkhoff(X0, iters=100)
            return X0
        except Exception:
            return None

    def random_init():
        X = np.random.rand(n, n)
        X = project_birkhoff(X, iters=100)
        return X

    def run_pgd(X_init, max_time, lr_factor=1.0):
        """Run projected gradient descent on Birkhoff polytope."""
        X = X_init.copy()
        last_check = time.time()
        
        # Estimate learning rate
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        grad_scale = norm_a * norm_b
        if grad_scale == 0:
            grad_scale = 1.0
        lr = (1.0 / (5.0 * grad_scale)) * lr_factor
        
        # Adaptive learning rate
        best_cost_so_far = np.inf
        
        while time.time() < max_time:
            G = a @ X @ b
            X = X - lr * G
            X = project_birkhoff(X, iters=30)
            
            # Check periodically
            if time.time() - last_check > 3.0:
                last_check = time.time()
                p = hungarian_solve(G)
                cost = calc_cost(p)
                if cost < best_cost_so_far:
                    best_cost_so_far = cost
                    best_p = p.copy()
                
                # Adjust learning rate
                if cost < best_cost_so_far * 0.95:
                    lr *= 0.7  # Decrease if improving
                elif cost > best_cost_so_far * 1.05:
                    lr *= 1.2  # Increase if stagnating
                
                if time.time() > max_time - 2.0:
                    return best_p, X
        
        # Final round
        G = a @ X @ b
        p = hungarian_solve(G)
        return p, X

    def local_search(p, max_time):
        """2-opt local search with vectorized delta computation."""
        current_cost = calc_cost(p)
        Bp = b[:, p]
        improved = True
        
        while improved and time.time() < max_time:
            improved = False
            
            # Compute M = a @ Bp
            M = a @ Bp
            D = M + M.T
            diag_M = np.diag(M)
            
            # Delta(i,j) = D[i,j] - diag_M[i] - diag_M[j]
            C = D - diag_M[:, None] - diag_M[None, :]
            
            # Mask out i >= j
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            C_masked = np.where(mask, C, 0)
            
            # Find best swap
            min_val = np.min(C_masked)
            if min_val < 0:
                idx = np.unravel_index(np.argmin(C_masked), C_masked.shape)
                i, j = idx
                p[i], p[j] = p[j], p[i]
                current_cost += int(min_val)
                Bp = b[:, p]
                improved = True
        
        return p, current_cost

    def simulated_annealing(p, max_time, T_start=1000.0, T_min=1.0, alpha=0.995):
        """Simulated annealing with 2-opt neighborhood."""
        rng = np.random.default_rng(42)
        current_cost = calc_cost(p)
        best_p = p.copy()
        best_cost = current_cost
        T = T_start
        
        while time.time() < max_time and T > T_min:
            # Pick random swap
            i, j = rng.choice(n, size=2, replace=False)
            i, j = int(i), int(j)
            if i == j:
                continue
            
            # Compute delta for swap
            Bp = b[:, p]
            M = a @ Bp
            D = M + M.T
            diag_M = np.diag(M)
            delta = D[i, j] - diag_M[i] - diag_M[j]
            
            # Acceptance test
            if delta < 0 or rng.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost += int(delta)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            T *= alpha
        
        return best_p, best_cost

    def tabu_search(p, max_time, tabu_size=10, max_iter=None):
        """Tabu search with 2-opt neighborhood."""
        if max_iter is None:
            max_iter = 10000
        
        current_cost = calc_cost(p)
        best_p = p.copy()
        best_cost = current_cost
        
        # Tabu list: store (i, j, iteration_added)
        tabu = {}
        rng = np.random.default_rng(42)
        
        for iteration in range(max_iter):
            if time.time() > max_time:
                break
            
            Bp = b[:, p]
            M = a @ Bp
            D = M + M.T
            diag_M = np.diag(M)
            
            # Compute all deltas
            C = D - diag_M[:, None] - diag_M[None, :]
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            C_masked = np.where(mask, C, 0)
            
            # Find best non-tabu move
            best_delta = 0
            best_i, best_j = -1, -1
            best_val = np.inf
            
            for i in range(n):
                for j in range(i + 1, n):
                    key = (i, j)
                    if key in tabu and tabu[key] + tabu_size > iteration:
                        continue
                    val = C_masked[i, j]
                    if val < best_val:
                        best_val = val
                        best_i, best_j = i, j
                        best_delta = val
            
            if best_i >= 0 and best_delta < 0:
                # Add to tabu list
                tabu[(best_i, best_j)] = iteration
                tabu[(best_j, best_i)] = iteration
                
                # Apply move
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += int(best_delta)
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        
        return best_p, best_cost

    best_p = None
    best_cost = np.inf

    # 1. Spectral Initialization + PGD
    try:
        X0 = spectral_init()
        if X0 is not None:
            p1, X1 = run_pgd(X0, deadline - 30.0, lr_factor=1.0)
            if p1 is not None:
                p1, c1 = local_search(p1, deadline - 20.0)
                if c1 < best_cost:
                    best_cost = c1
                    best_p = p1
    except Exception:
        pass

    # 2. Random Restart + PGD
    if best_p is not None and time.time() < deadline - 25.0:
        try:
            X2 = random_init()
            p2, X2_final = run_pgd(X2, deadline - 25.0, lr_factor=0.5)
            if p2 is not None:
                p2, c2 = local_search(p2, deadline - 15.0)
                if c2 < best_cost:
                    best_cost = c2
                    best_p = p2
        except Exception:
            pass

    # 3. Simulated Annealing from best solution
    if best_p is not None and time.time() < deadline - 20.0:
        try:
            p3, c3 = simulated_annealing(best_p.copy(), deadline - 15.0)
            if c3 < best_cost:
                best_cost = c3
                best_p = p3
        except Exception:
            pass

    # 4. Tabu Search from best solution
    if best_p is not None and time.time() < deadline - 15.0:
        try:
            p4, c4 = tabu_search(best_p.copy(), deadline - 10.0)
            if c4 < best_cost:
                best_cost = c4
                best_p = p4
        except Exception:
            pass

    # 5. Final local search pass
    if best_p is not None and time.time() < deadline - 5.0:
        try:
            p5, c5 = local_search(best_p.copy(), deadline - 3.0)
            if c5 < best_cost:
                best_cost = c5
                best_p = p5
        except Exception:
            pass

    if best_p is None:
        # Fallback: Hungarian on A? Or identity?
        best_p = np.arange(n)
        best_cost = calc_cost(best_p)

    return best_p.tolist()
# EVOLVE-BLOCK-END
