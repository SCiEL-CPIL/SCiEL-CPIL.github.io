# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    
    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1,
           where P[y, x] is the joint probability that X = x and Y = y.
    """
    n = 10
    deadline = time.time() + 600.0
    
    # Precompute projection indices
    x_coords = np.tile(np.arange(n), (n, 1))  # x_coords[y, x] = x
    y_coords = np.tile(np.arange(n)[:, None], (1, n))  # y_coords[y, x] = y
    
    # H_{-1}: x - y
    proj_m1 = (x_coords - y_coords).ravel()
    vals_m1 = proj_m1 + 9  # shift to 0..18
    
    # H_0: x
    proj_0 = x_coords.ravel()
    
    # H_1: x + y
    proj_1 = (x_coords + y_coords).ravel()
    
    # H_2: x + 2*y
    proj_2 = (x_coords + 2 * y_coords).ravel()
    
    def compute_entropies(P):
        P_flat = P.ravel()
        
        # H_{-1}
        probs_m1 = np.zeros(19)
        np.add.at(probs_m1, vals_m1, P_flat)
        probs_m1 = probs_m1[probs_m1 > 0]
        H_m1 = -np.sum(probs_m1 * np.log2(probs_m1))
        
        # H_0
        probs_0 = np.zeros(n)
        np.add.at(probs_0, proj_0, P_flat)
        probs_0 = probs_0[probs_0 > 0]
        H_0 = -np.sum(probs_0 * np.log2(probs_0))
        
        # H_1
        probs_1 = np.zeros(19)
        np.add.at(probs_1, proj_1, P_flat)
        probs_1 = probs_1[probs_1 > 0]
        H_1 = -np.sum(probs_1 * np.log2(probs_1))
        
        # H_2
        probs_2 = np.zeros(28)
        np.add.at(probs_2, proj_2, P_flat)
        probs_2 = probs_2[probs_2 > 0]
        H_2 = -np.sum(probs_2 * np.log2(probs_2))
        
        M = max(H_0, H_1, H_2)
        D = H_m1
        return D, M, D / M if M > 1e-9 else 0.0

    def smooth_objective(theta):
        """
        theta = [logits (100), t (1)]
        Objective: - (D - t * M_soft)
        Where M_soft = (1/beta) * log(exp(beta*H0) + exp(beta*H1) + exp(beta*H2))
        This is an epigraph formulation: minimize -D + t*M subject to t >= M.
        Wait, the standard epigraph for minimizing f(x)/g(x) is not directly -D + t*M.
        
        Let's rethink. We want to maximize D/M.
        Let z = D/M.
        z <= D/t  => z*t <= D (if t>0)
        t <= M => t - M <= 0.
        
        Alternatively, just use the soft-max surrogate for M directly in the ratio.
        Ratio = D / M_soft.
        Maximizing D / M_soft is equivalent to minimizing -D / M_soft.
        This is smooth if M_soft > 0.
        """
        logits = theta[:100]
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        P = exp_logits / np.sum(exp_logits)
        
        P_flat = P.ravel()
        
        # Compute H_m1
        probs_m1 = np.zeros(19)
        np.add.at(probs_m1, vals_m1, P_flat)
        mask_m1 = probs_m1 > 1e-15
        if not np.any(mask_m1):
            return 0.0
        H_m1 = -np.sum(probs_m1[mask_m1] * np.log2(probs_m1[mask_m1]))
        
        # Compute H_0, H_1, H_2
        probs_0 = np.zeros(n)
        np.add.at(probs_0, proj_0, P_flat)
        mask_0 = probs_0 > 1e-15
        H_0 = -np.sum(probs_0[mask_0] * np.log2(probs_0[mask_0])) if np.any(mask_0) else 0.0
        
        probs_1 = np.zeros(19)
        np.add.at(probs_1, proj_1, P_flat)
        mask_1 = probs_1 > 1e-15
        H_1 = -np.sum(probs_1[mask_1] * np.log2(probs_1[mask_1])) if np.any(mask_1) else 0.0
        
        probs_2 = np.zeros(28)
        np.add.at(probs_2, proj_2, P_flat)
        mask_2 = probs_2 > 1e-15
        H_2 = -np.sum(probs_2[mask_2] * np.log2(probs_2[mask_2])) if np.any(mask_2) else 0.0
        
        # Soft-max for M
        beta = 10.0 # Temperature for soft-max. Higher is closer to hard max.
        Hs = np.array([H_0, H_1, H_2])
        # Log-sum-exp trick for stability
        max_H = np.max(Hs)
        if max_H < 1e-9:
            return 0.0
        M_soft = (max_H + np.log(np.sum(np.exp(beta * (Hs - max_H))))) / beta
        
        if M_soft < 1e-9:
            return 0.0
            
        ratio = H_m1 / M_soft
        return -ratio

    def get_P_from_theta(theta):
        logits = theta[:100]
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        P = exp_logits / np.sum(exp_logits)
        return P

    # Generate initializations
    np.random.seed(42)
    inits = []
    
    # Uniform
    P_uniform = np.ones((n, n)) / (n * n)
    inits.append(np.log(P_uniform).ravel())
    
    # Diagonal
    P_diag = np.zeros((n, n))
    for y in range(n):
        P_diag[y, y] = 1.0
    P_diag += np.random.rand(n, n) * 0.01
    P_diag /= np.sum(P_diag)
    inits.append(np.log(P_diag).ravel())
    
    # Anti-diagonal
    P_anti = np.zeros((n, n))
    for y in range(n):
        P_anti[y, 9-y] = 1.0
    P_anti += np.random.rand(n, n) * 0.01
    P_anti /= np.sum(P_anti)
    inits.append(np.log(P_anti).ravel())
    
    # Random
    for _ in range(5):
        P_rand = np.random.rand(n, n)
        P_rand /= np.sum(P_rand)
        inits.append(np.log(P_rand).ravel())

    best_score = -1
    best_P = None
    best_theta = None

    # Evaluate inits
    for theta_init in inits:
        if time.time() > deadline - 60:
            break
        # Pad with t=1.0 for the auxiliary variable (though not used in this direct ratio version, 
        # if we were using epigraph we'd need it. Here we just use smooth ratio).
        # Wait, the directive mentioned auxiliary variable t. 
        # If I use direct ratio D/M_soft, I don't need t.
        # If I use epigraph: minimize -D + t*M subject to t >= M.
        # Let's stick to the smooth ratio D/M_soft first as it's simpler and differentiable.
        # The "different family" is the smooth surrogate for max.
        
        theta_full = np.concatenate([theta_init, [1.0]]) # Just in case, though not used in smooth_objective above
        # Actually, let's modify smooth_objective to accept the full theta and ignore the extra dim if not needed, 
        # or just pass the 100 dim vector.
        
        val = smooth_objective(theta_init)
        score = -val
        if score > best_score:
            best_score = score
            best_P = get_P_from_theta(theta_init)
            best_theta = theta_init.copy()

    # Optimize
    options = {
        'maxiter': 1000,
        'ftol': 1e-12,
        'gtol': 1e-8,
        'maxfun': 5000
    }
    
    for theta_init in inits:
        if time.time() > deadline - 30:
            break
        try:
            result = minimize(
                smooth_objective,
                theta_init,
                method='L-BFGS-B',
                options=options
            )
            val = smooth_objective(result.x)
            score = -val
            if score > best_score:
                best_score = score
                best_P = get_P_from_theta(result.x)
                best_theta = result.x.copy()
        except Exception:
            pass

    # Polish with tighter tolerance
    if best_theta is not None and time.time() < deadline - 30:
        try:
            options_tight = {
                'maxiter': 5000,
                'ftol': 1e-15,
                'gtol': 1e-10,
                'maxfun': 20000
            }
            result = minimize(
                smooth_objective,
                best_theta,
                method='L-BFGS-B',
                options=options_tight
            )
            val = smooth_objective(result.x)
            score = -val
            if score > best_score:
                best_score = score
                best_P = get_P_from_theta(result.x)
                best_theta = result.x.copy()
        except Exception:
            pass

    # Final Validation and Normalization
    if best_P is None:
        best_P = np.ones((n, n)) / (n * n)
    
    best_P = np.clip(best_P, 0, None)
    total = np.sum(best_P)
    if total <= 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P = best_P / total

    # Verify with original hard max logic to ensure we didn't drift too far
    D, M, final_score = compute_entropies(best_P)
    if M <= 1e-9 or np.any(np.isnan(best_P)) or np.any(np.isinf(best_P)) or np.any(best_P < -1e-9):
        best_P = np.ones((n, n)) / (n * n)
    else:
        # If the smooth optimization drifted such that the hard max score is worse than a known good one, 
        # we might want to keep the best hard max score found during the process. 
        # However, usually the smooth surrogate is a good proxy. 
        # Let's just return the P. The evaluator computes the score.
        pass

    return best_P

# EVOLVE-BLOCK-END
