# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    # Deadline guard: evaluator kills at 630s; keep margin.
    t0 = time.time()
    deadline = t0 + 590.0

    N = 200

    # Compute Möbius function up to N
    mu = np.ones(N + 1, dtype=np.float64)
    is_prime = np.ones(N + 1, dtype=bool)
    for i in range(2, N + 1):
        if is_prime[i]:
            for j in range(i, N + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            for j in range(i2, N + 1, i2):
                mu[j] = 0.0

    sq_free = np.array([k for k in range(2, N + 1) if mu[k] != 0], dtype=np.int64)
    m = len(sq_free)

    try:
        from scipy.optimize import linprog
        have_scipy = True
    except ImportError:
        have_scipy = False

    c = np.array([np.log(k) / k for k in sq_free], dtype=np.float64)

    # Vectorized construction of A_ub
    x_vals = np.arange(1, 10 * N + 1, dtype=np.float64)
    # A_ub[i, j] = floor(x_i / k_j) - x_i / k_j
    # Shape: (len(x_vals), m)
    x_col = x_vals[:, np.newaxis]  # (10N, 1)
    k_row = sq_free[np.newaxis, :].astype(np.float64)  # (1, m)
    A_ub = np.floor(x_col / k_row) - x_col / k_row  # (10N, m)

    b_ub = np.ones(len(x_vals), dtype=np.float64)
    bounds = [(-10.0, 10.0)] * m

    best_g = None
    best_obj = -np.inf

    if have_scipy:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            best_g = result.x.copy()
            best_obj = -np.dot(best_g, c)  # = -sum g(k) log(k)/k

    # Coordinate descent polish
    if best_g is not None and time.time() < deadline - 30:
        g = best_g.copy()
        # Precompute row sums for fast constraint evaluation
        # M(x) = sum_j g_j * A_ub[x, j]
        # We track M_max = max_x M(x)
        # For coordinate descent: try moving g_j to -10 or 10, see if M(x) <= 1 still holds.
        # Marginal objective gain for setting g_j = v: delta_obj = v * (-log(k_j)/k_j)
        # Since we're maximizing -sum g_j log(k_j)/k_j, increasing g_j (toward 10) helps
        # if log(k_j)/k_j > 0 (always true for k>=2).
        
        # Actually, we want to maximize a = -sum g(k) log(k)/k.
        # So we want g(k) as negative as possible for large log(k)/k.
        # But constraints limit us.
        
        # Let's do iterative: for each coordinate, try to push it to its bound
        # that maximizes objective while keeping feasibility.
        
        for iteration in range(50):
            if time.time() > deadline - 30:
                break
            improved = False
            for j in range(m):
                k_j = sq_free[j]
                # Current contribution to M(x) from this coordinate: g[j] * A_ub[:, j]
                # If we change g[j] to v, new M(x) = M_old(x) - g[j]*A_ub[x,j] + v*A_ub[x,j]
                # = M_old(x) + (v - g[j]) * A_ub[x,j]
                
                # Try v = -10 (push negative to maximize -g*log(k)/k)
                # and v = 10 (push positive)
                
                for v in [-10.0, 10.0]:
                    if v == g[j]:
                        continue
                    delta = v - g[j]
                    # New M(x) = M_old + delta * A_ub[:, j]
                    # Need max_x (M_old + delta * A_ub[:, j]) <= 1
                    # M_old = A_ub @ g
                    # But computing full M_old each time is expensive.
                    # Let's maintain M_vec = A_ub @ g
                    pass
                
                # Simpler: just evaluate if setting to bound is feasible
                M_vec = A_ub @ g
                for v in [-10.0, 10.0]:
                    if abs(v - g[j]) < 1e-12:
                        continue
                    delta = v - g[j]
                    new_M = M_vec + delta * A_ub[:, j]
                    if new_M.max() <= 1.0:
                        # Check objective improvement
                        old_obj = -g[j] * np.log(k_j) / k_j
                        new_obj = -v * np.log(k_j) / k_j
                        if new_obj > old_obj + 1e-12:
                            g[j] = v
                            M_vec = new_M
                            improved = True
                            break
            
            if not improved:
                break
        
        # Update best if improved
        obj_g = -np.dot(g, c)
        if obj_g > best_obj:
            best_g = g
            best_obj = obj_g

    # Final result
    if best_g is not None:
        g_vals = best_g
        f1 = -np.sum(g_vals / sq_free)
        domain = np.concatenate(([1], sq_free))
        values = np.concatenate(([f1], g_vals))
        
        # Verify
        K = int(domain.max())
        M_max = 0.0
        for x in range(1, 10 * K + 1):
            Mx = f1 * x + np.sum(g_vals * np.floor(x / sq_free))
            if Mx > M_max:
                M_max = Mx
        
        if M_max <= 1.0001:
            return domain.astype(np.int64), values.astype(np.float64)

    # Fallback: Use Möbius values directly
    mu_vals = np.array([mu[k] for k in sq_free], dtype=np.float64)
    f1 = -np.sum(mu_vals / sq_free)
    domain = np.concatenate(([1], sq_free))
    values = np.concatenate(([f1], mu_vals))
    return domain.astype(np.int64), values.astype(np.float64)

# EVOLVE-BLOCK-END
