# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Maximize D/M where D = H_{-1} (entropy of x-y) and M = max(H_0, H_1, H_2).
    Uses gradient-based optimization with precomputed pooling matrices and analytic gradients.
    """
    n = 10
    size = n * n  # 100

    # Precompute pooling matrices for slopes -1, 0, 1, 2
    def build_pooling_matrix(r):
        if r == -1:
            min_val = -9
            max_val = 9
        elif r == 0:
            min_val = 0
            max_val = 9
        elif r == 1:
            min_val = 0
            max_val = 18
        elif r == 2:
            min_val = 0
            max_val = 27
        else:
            min_val = 0
            max_val = 9 + abs(r) * 9

        num_bins = int(max_val - min_val + 1)
        A = np.zeros((num_bins, size), dtype=np.float64)

        for y in range(n):
            for x in range(n):
                val = x + r * y
                idx = y * n + x
                bin_idx = int(val - min_val)
                if 0 <= bin_idx < num_bins:
                    A[bin_idx, idx] = 1.0
        return A

    A_minus1 = build_pooling_matrix(-1)
    A_0 = build_pooling_matrix(0)
    A_1 = build_pooling_matrix(1)
    A_2 = build_pooling_matrix(2)

    # Precompute log2 for gradient computation
    log2 = np.log(2.0)

    def entropy_and_grad(p):
        """
        Compute Shannon entropy in bits and its gradient w.r.t. p.
        p is a 1D array of probabilities (non-negative, sum to 1).
        Returns (entropy, grad) where grad[i] = dH/dp[i].
        """
        p = np.asarray(p, dtype=np.float64)
        # Mask for positive probabilities
        mask = p > 1e-15
        if not np.any(mask):
            return 0.0, np.zeros_like(p)
        
        pm = p[mask]
        # Entropy: H = -sum(p * log2(p)) = -sum(p * ln(p)) / ln(2)
        entropy = -np.sum(pm * np.log(pm)) / log2
        
        # Gradient: dH/dp[i] = -log2(p[i]) - 1/ln(2) for p[i] > 0, else 0
        grad = np.zeros_like(p)
        grad[mask] = (-np.log(pm) - 1.0) / log2
        
        return entropy, grad

    def value_and_grad(p_vec):
        """
        Compute value = D/M and its gradient w.r.t. p_vec.
        Uses smooth max approximation for M = max(H0, H1, H2).
        """
        p_vec = np.asarray(p_vec, dtype=np.float64)
        p_vec = np.maximum(p_vec, 1e-15)
        p_vec = p_vec / np.sum(p_vec)

        # Compute marginals
        m_minus1 = A_minus1 @ p_vec
        m_0 = A_0 @ p_vec
        m_1 = A_1 @ p_vec
        m_2 = A_2 @ p_vec

        # Entropies and gradients
        D, grad_D_m = entropy_and_grad(m_minus1)
        H0, grad_H0_m = entropy_and_grad(m_0)
        H1, grad_H1_m = entropy_and_grad(m_1)
        H2, grad_H2_m = entropy_and_grad(m_2)

        # Smooth max for M
        # Use softmax-like weighting to compute smooth max
        H_vec = np.array([H0, H1, H2])
        # Temperature for smooth max
        tau = 0.1
        # Softmax weights
        exp_H = np.exp(H_vec / tau)
        weights = exp_H / np.sum(exp_H)
        
        # Smooth max value
        M = np.sum(weights * H_vec)
        
        # Gradient of M w.r.t. H0, H1, H2
        # dM/dHi = (weights[i] * H_i - M * weights[i]) / tau + weights[i]
        # Actually, for M = tau * log(sum(exp(Hi/tau))), dM/dHi = exp(Hi/tau)/sum(exp(Hj/tau)) = weights[i]
        # But we want gradient of value = D/M, so we need dM/dp
        # dM/dp = sum_i weights[i] * dHi/dp
        grad_M = weights[0] * (A_0.T @ grad_H0_m) + \
                  weights[1] * (A_1.T @ grad_H1_m) + \
                  weights[2] * (A_2.T @ grad_H2_m)

        # Gradient of D w.r.t. p
        grad_D = A_minus1.T @ grad_D_m

        if M < 1e-9:
            return 0.0, np.zeros(size)

        # value = D / M
        value = D / M

        # Gradient of value = (grad_D * M - D * grad_M) / M^2
        grad_value = (grad_D * M - D * grad_M) / (M * M)

        return value, grad_value

    def objective_and_grad(p_vec):
        """
        Returns (-value, -grad_value) for minimization.
        """
        value, grad_value = value_and_grad(p_vec)
        return (-value, -grad_value)

    # Bounds
    bounds = Bounds(1e-8, 1.0)

    # Constraint: sum(p) = 1
    A_eq = np.ones((1, size))
    constraint = LinearConstraint(A_eq, 1.0, 1.0)

    # Multiple restarts
    best_p = np.ones(size) / size
    best_value = -np.inf

    # Initial candidates
    candidates = [np.ones(size) / size]
    
    # Random Dirichlet initializations
    rng = np.random.default_rng(42)
    for _ in range(20):
        alpha = rng.uniform(0.1, 2.0, size=size)
        p_init = rng.dirichlet(alpha)
        candidates.append(p_init)
    
    # Some structured initializations
    # Diagonal heavy
    p_diag = np.zeros(size)
    for i in range(n):
        p_diag[i * n + i] = 1.0 / n
    candidates.append(p_diag)
    
    # Anti-diagonal heavy
    p_anti = np.zeros(size)
    for i in range(n):
        p_anti[i * n + (n - 1 - i)] = 1.0 / n
    candidates.append(p_anti)
    
    # Row-heavy
    p_row = np.zeros(size)
    for i in range(n):
        p_row[i * n + (i % n)] = 1.0 / n
    candidates.append(p_row)

    # Evaluate each candidate and run local optimization
    for p_init in candidates:
        try:
            # First evaluate the initial point
            val_init, _ = value_and_grad(p_init)
            
            # Run L-BFGS-B
            result = minimize(
                lambda p: objective_and_grad(p)[0],
                p_init,
                jac=lambda p: objective_and_grad(p)[1],
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraint,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            p_opt = result.x
            val_opt, _ = value_and_grad(p_opt)
            
            if val_opt > best_value:
                best_value = val_opt
                best_p = p_opt.copy()
                
        except Exception:
            continue

    # Final refinement on best result
    try:
        result = minimize(
            lambda p: objective_and_grad(p)[0],
            best_p,
            jac=lambda p: objective_and_grad(p)[1],
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraint,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        best_p = result.x
    except Exception:
        pass

    # Reshape and normalize
    P = best_p.reshape((n, n))
    P = np.maximum(P, 0.0)
    total = np.sum(P)
    if total > 1e-9:
        P = P / total
    else:
        P = np.ones((n, n)) / size

    return P


# EVOLVE-BLOCK-END
