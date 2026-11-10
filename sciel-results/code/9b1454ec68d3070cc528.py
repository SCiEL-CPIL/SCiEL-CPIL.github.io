# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Maximize D/M where D = H_{-1} (entropy of x-y) and M = max(H_0, H_1, H_2).
    Uses analytic gradient ascent with simplex projection.
    """
    n = 10
    size = n * n  # 100

    # Precompute pooling matrices for slopes -1, 0, 1, 2
    # For slope r, value = x + r*y
    # P[y, x] corresponds to index y*n + x

    def build_pooling_matrix(r):
        """Build pooling matrix A such that A @ p_flat gives marginal distribution of x + r*y."""
        if r == -1:
            min_val = 0 + (-1) * 9  # x=0, y=9 -> -9
            max_val = 9 + (-1) * 0  # x=9, y=0 -> 9
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

    def shannon_entropy(p):
        """Compute Shannon entropy in bits, ignoring zero probabilities."""
        p = np.asarray(p)
        p = p[p > 1e-15]
        if len(p) == 0:
            return 0.0
        return float(-np.sum(p * np.log2(p)))

    def grad_shannon_entropy(A, p_vec):
        """
        Compute gradient of H(A p_vec) with respect to p_vec.
        H(m) = -sum(m_i log2 m_i)
        dH/dm_i = -log2(m_i) - 1/ln(2) for m_i > 0, else 0.
        dH/dp = A.T @ (dH/dm)
        """
        m = A @ p_vec
        m_safe = np.where(m > 1e-15, m, 1.0)
        # Gradient w.r.t m
        grad_m = -np.log2(m_safe) - 1.0 / np.log(2.0)
        # Zero out where m was effectively 0
        grad_m[m <= 1e-15] = 0.0
        
        return A.T @ grad_m

    def project_to_simplex(z):
        """
        Project vector z onto the probability simplex {x >= 0, sum(x)=1}.
        Uses the algorithm from Duchi et al. (2008).
        """
        n = len(z)
        u = np.sort(z)[::-1]
        cssv = np.cumsum(u)
        ind = np.arange(1, n + 1)
        cond = u - (cssv - 1) / ind > 0
        rho = ind[cond][-1]
        theta = (cssv[cond][-1] - 1) / float(rho)
        p = np.maximum(z - theta, 0)
        return p

    def compute_value_and_grad(p_vec):
        """
        Compute value D/M and its gradient.
        """
        # Ensure p is valid for entropy calculation (avoid log(0))
        p_vec = np.maximum(p_vec, 1e-15)
        p_vec = p_vec / np.sum(p_vec)
        
        m_minus1 = A_minus1 @ p_vec
        m_0 = A_0 @ p_vec
        m_1 = A_1 @ p_vec
        m_2 = A_2 @ p_vec

        D = shannon_entropy(m_minus1)
        H0 = shannon_entropy(m_0)
        H1 = shannon_entropy(m_1)
        H2 = shannon_entropy(m_2)
        
        Hs = [H0, H1, H2]
        M = max(Hs)
        
        if M < 1e-9:
            return 0.0, np.zeros(size)

        value = D / M

        # Gradients of entropies w.r.t p_vec
        g_D = grad_shannon_entropy(A_minus1, p_vec)
        g_H0 = grad_shannon_entropy(A_0, p_vec)
        g_H1 = grad_shannon_entropy(A_1, p_vec)
        g_H2 = grad_shannon_entropy(A_2, p_vec)
        
        g_Hs = [g_H0, g_H1, g_H2]
        
        # Find argmax H
        k_star = np.argmax(Hs)
        g_M = g_Hs[k_star]

        # Gradient of V = D/M
        # dV = (1/M) dD - (D/M^2) dM
        grad_V = (1.0 / M) * g_D - (D / (M * M)) * g_M

        return value, grad_V

    # Initial guess: uniform distribution
    p = np.ones(size) / size
    
    # Optimization loop
    # We use gradient ascent with adaptive step size or fixed small step
    # Since the domain is compact, we can use a moderate number of iterations
    
    max_iter = 2000
    best_p = p.copy()
    best_val = 0.0
    
    # Simple adaptive step size
    step_size = 0.05
    
    for i in range(max_iter):
        val, grad = compute_value_and_grad(p)
        
        if val > best_val:
            best_val = val
            best_p = p.copy()
            
        # Check convergence
        if i > 100 and abs(val - best_val) < 1e-9:
            break
            
        # Gradient ascent step
        p_new = p + step_size * grad
        
        # Project to simplex
        p_new = project_to_simplex(p_new)
        
        # Ensure numerical stability
        p_new = np.maximum(p_new, 1e-15)
        p_new = p_new / np.sum(p_new)
        
        # Simple line search / step adjustment
        val_new, _ = compute_value_and_grad(p_new)
        
        if val_new > val:
            p = p_new
            # Can increase step size slightly
            # step_size = min(step_size * 1.05, 0.5)
        else:
            # Decrease step size
            step_size *= 0.5
            if step_size < 1e-5:
                break
                
    p = best_p
    
    # Reshape and normalize
    P = p.reshape((n, n))
    P = np.maximum(P, 0.0)
    total = np.sum(P)
    if total > 1e-9:
        P = P / total
    else:
        P = np.ones((n, n)) / size

    return P


# EVOLVE-BLOCK-END
