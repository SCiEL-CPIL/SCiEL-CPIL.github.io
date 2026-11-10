# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin (total limit is 630s, leave buffer)
    
    # Pre-compute grid coordinates
    y, x = np.mgrid[0:n, 0:n]
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    
    # Pre-compute projection indices for fast bincount
    # x - y ranges from -9 to 9 -> offset by 9, size 19
    idx_m1 = (x - y + n - 1).astype(np.int32).ravel()
    # x ranges from 0 to 9 -> size 10
    idx_0 = x.astype(np.int32).ravel()
    # x + y ranges from 0 to 18 -> size 19
    idx_1 = (x + y).astype(np.int32).ravel()
    # x + 2y ranges from 0 to 27 -> size 28
    idx_2 = (x + 2 * y).astype(np.int32).ravel()
    
    size_m1 = 2 * n - 1
    size_0 = n
    size_1 = 2 * n - 1
    size_2 = 3 * n - 1

    def compute_entropies(P):
        """Compute H_{-1}, H_0, H_1, H_2 for a given distribution P."""
        w = P.ravel()
        
        counts_m1 = np.bincount(idx_m1, weights=w, minlength=size_m1)
        p_m1 = counts_m1[counts_m1 > 0]
        H_m1 = -np.sum(p_m1 * np.log2(p_m1)) if len(p_m1) > 0 else 0.0

        counts_0 = np.bincount(idx_0, weights=w, minlength=size_0)
        p_0 = counts_0[counts_0 > 0]
        H_0 = -np.sum(p_0 * np.log2(p_0)) if len(p_0) > 0 else 0.0

        counts_1 = np.bincount(idx_1, weights=w, minlength=size_1)
        p_1 = counts_1[counts_1 > 0]
        H_1 = -np.sum(p_1 * np.log2(p_1)) if len(p_1) > 0 else 0.0

        counts_2 = np.bincount(idx_2, weights=w, minlength=size_2)
        p_2 = counts_2[counts_2 > 0]
        H_2 = -np.sum(p_2 * np.log2(p_2)) if len(p_2) > 0 else 0.0

        return H_m1, H_0, H_1, H_2

    def get_score(P):
        H_m1, H_0, H_1, H_2 = compute_entropies(P)
        M = max(H_0, H_1, H_2)
        if M <= 1e-9:
            return 0.0
        return H_m1 / M

    # Baseline: Uniform
    P_uniform = np.ones((n, n)) / (n * n)
    best_P = P_uniform.copy()
    best_score = get_score(best_P)
    
    # Strategy 1: Product of Marginals P[y, x] = R[y] * C[x]
    # This reduces the search space to 20 dimensions (10 for R, 10 for C)
    # We normalize R and C so that sum(R)*sum(C) = 1, or just normalize P at the end.
    # Let R and C be non-negative vectors. P = outer(C, R) / sum(outer(C, R))
    
    def product_marginal_score(params):
        """
        params: [C_0...C_9, R_0...R_9]
        Returns negative score for minimization.
        """
        if time.time() > deadline - 5:
            return 1000.0
        C = params[:n]
        R = params[n:]
        
        # Ensure non-negative (clip small negatives from optimization noise)
        C = np.clip(C, 0, None)
        R = np.clip(R, 0, None)
        
        P = np.outer(C, R)
        s = np.sum(P)
        if s <= 1e-12:
            return 1000.0
        P /= s
        
        score = get_score(P)
        return -score

    # Initial guesses for Product Marginals
    # Uniform marginals
    init_uniform = np.concatenate([np.ones(n), np.ones(n)])
    
    # Try some structured marginals
    # E.g., concentrated on diagonals?
    # If P is on diagonal x=y, then x-y=0 (H_m1=0), bad.
    # We want high H_m1 (spread out x-y) and low M (concentrated x, x+y, x+2y).
    # High H_m1 means x-y is spread. Low H_0 means x is concentrated.
    # This implies y is also spread relative to x?
    # If x is concentrated at 4.5, and x-y is spread, then y must be spread.
    # If y is spread, x+y is spread -> H_1 high. Bad.
    # Maybe x is spread, y is concentrated?
    # If y is concentrated at 4.5, x-y is spread if x is spread.
    # Then H_0 (x) is high. Bad.
    
    # The trade-off is complex. Let's use Differential Evolution for global search in this 20D space.
    
    bounds_pm = [(0, 1)] * (2 * n)
    
    # Run DE for product marginals
    try:
        res_de = differential_evolution(
            product_marginal_score, 
            bounds_pm, 
            seed=42, 
            maxiter=100, 
            tol=1e-6,
            popsize=20,
            polish=False # We'll polish manually
        )
        if time.time() < deadline - 30:
            P_cand = np.outer(res_de.x[:n], res_de.x[n:])
            s = np.sum(P_cand)
            if s > 1e-12:
                P_cand /= s
                score_cand = get_score(P_cand)
                if score_cand > best_score:
                    best_score = score_cand
                    best_P = P_cand
    except Exception:
        pass

    # Strategy 2: Direct Optimization of P using SLSQP with gradients
    # This allows for non-product distributions, which might yield higher scores.
    # We implement a smooth approximation of the max function to get gradients.
    # M = max(H0, H1, H2).
    # Use soft max: M_beta = (1/beta) * log(sum(exp(beta*H_i)))
    # Gradient of M_beta w.r.t P is weighted average of gradients of H_i.
    
    def grad_entropy(P_flat, idx_flat, n_bins):
        """
        Computes gradient of Shannon entropy H = -sum q_i log2(q_i) w.r.t P.
        q_i = sum_{j in bin i} P_j
        dH/dP_j = -log2(q_{bin(j)}) - 1/ln(2)
        """
        q = np.bincount(idx_flat, weights=P_flat, minlength=n_bins)
        # Mask zeros to avoid log(0). If q_i is 0, gradient is undefined/0 effectively in limit?
        # Actually if q_i=0, P_j in that bin is 0, so term is 0.
        # But we need to be careful.
        mask = q > 0
        grad_q = np.zeros(n_bins)
        grad_q[mask] = -np.log2(q[mask]) - 1.0 / np.log(2.0)
        grad_P = grad_q[idx_flat]
        return grad_P

    def objective_grad_product(P_flat, beta=10.0):
        """
        Objective: - (H_m1 / M)
        Using soft max for M.
        """
        if time.time() > deadline - 5:
            return 1000.0, np.zeros_like(P_flat)
        
        # Compute H_m1
        q_m1 = np.bincount(idx_m1, weights=P_flat, minlength=size_m1)
        mask_m1 = q_m1 > 0
        if np.sum(mask_m1) == 0:
            return 1000.0, np.zeros_like(P_flat)
        p_m1 = q_m1[mask_m1]
        H_m1 = -np.sum(p_m1 * np.log2(p_m1))
        grad_H_m1 = np.zeros(size_m1)
        grad_H_m1[mask_m1] = -np.log2(p_m1) - 1.0 / np.log(2.0)
        grad_P_m1 = grad_H_m1[idx_m1]
        
        # Compute H_0, H_1, H_2 and their grads
        Hs = []
        grad_Ps = []
        
        for idx_flat, n_bins in [(idx_0, size_0), (idx_1, size_1), (idx_2, size_2)]:
            q = np.bincount(idx_flat, weights=P_flat, minlength=n_bins)
            mask = q > 0
            if np.sum(mask) == 0:
                H = 0.0
                gP = np.zeros_like(P_flat)
            else:
                p = q[mask]
                H = -np.sum(p * np.log2(p))
                grad_q = np.zeros(n_bins)
                grad_q[mask] = -np.log2(p) - 1.0 / np.log(2.0)
                gP = grad_q[idx_flat]
            Hs.append(H)
            grad_Ps.append(gP)
            
        Hs = np.array(Hs)
        if np.max(Hs) < 1e-9:
            return 1000.0, np.zeros_like(P_flat)
            
        # Soft max M
        # To avoid overflow, subtract max
        H_max = np.max(Hs)
        exp_Hs = np.exp(beta * (Hs - H_max))
        sum_exp = np.sum(exp_Hs)
        M = (H_max + np.log(sum_exp)) / beta
        
        if M < 1e-9:
            return 1000.0, np.zeros_like(P_flat)
            
        # Weights for soft max gradient
        w = exp_Hs / sum_exp
        
        # Gradient of M
        grad_M = np.zeros_like(P_flat)
        for i in range(3):
            grad_M += w[i] * grad_Ps[i]
            
        # Ratio R = H_m1 / M
        # dR/dP = (M * dH_m1/dP - H_m1 * dM/dP) / M^2
        grad_R = (M * grad_P_m1 - H_m1 * grad_M) / (M * M)
        
        # We want to maximize R, so minimize -R
        obj = -H_m1 / M
        grad_obj = -grad_R
        
        return obj, grad_obj

    # Run SLSQP from best_P found so far
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                   'jac': lambda x: np.ones_like(x)}]
    bounds = [(0.0, 1.0) for _ in range(n * n)]
    
    # Try multiple starts for SLSQP
    starts = [best_P.ravel().copy()]
    
    # Add some random starts
    rng = np.random.default_rng(123)
    for _ in range(10):
        p = rng.random((n, n))
        p /= p.sum()
        starts.append(p.ravel())
        
    # Add product marginal best if distinct
    if not np.allclose(best_P.ravel(), starts[0]):
         # best_P is already in starts[0] if it was updated
         pass

    for p0 in starts:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(
                lambda x: objective_grad_product(x)[0],
                p0,
                jac=lambda x: objective_grad_product(x)[1],
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            if res.success:
                P_cand = np.clip(res.x, 0, None).reshape(n, n)
                s = np.sum(P_cand)
                if s > 1e-12:
                    P_cand /= s
                    score_cand = get_score(P_cand)
                    if score_cand > best_score:
                        best_score = score_cand
                        best_P = P_cand
        except Exception:
            pass

    # Final validation
    best_P = np.clip(best_P, 0, None)
    s = np.sum(best_P)
    if s > 1e-9:
        best_P = best_P / s
    else:
        best_P = P_uniform

    return best_P

# EVOLVE-BLOCK-END
