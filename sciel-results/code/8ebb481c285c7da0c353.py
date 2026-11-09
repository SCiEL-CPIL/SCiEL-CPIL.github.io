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
    
    # Precompute projection indices for each slope
    # For slope r: value = x + r * y
    # x in 0..9, y in 0..9
    
    # H_{-1}: x - y
    # H_0: x
    # H_1: x + y
    # H_2: x + 2*y
    
    def compute_entropies(P):
        """
        Compute D = H_{-1} and M = max(H_0, H_1, H_2) from distribution P[y, x].
        """
        P_flat = P.ravel()
        
        # x - y: for cell (y, x), value = x - y, range -9 to 9
        # Map to index: x - y + 9, range 0 to 18
        idx_m1 = np.arange(n*n).reshape(n, n)
        # Actually, let's build it properly
        # P is indexed as P[y, x]
        # value = x - y
        X_idx, Y_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
        # But P[y, x], so we need to be careful
        # Let's create arrays where entry [y, x] gives us x and y
        x_coords = np.tile(np.arange(n), (n, 1))  # x_coords[y, x] = x
        y_coords = np.tile(np.arange(n)[:, None], (1, n))  # y_coords[y, x] = y
        
        # H_{-1}: x - y
        proj_m1 = (x_coords - y_coords).ravel()
        # Unique values: -9 to 9
        # Pool probabilities
        vals_m1 = proj_m1 + 9  # shift to 0..18
        probs_m1 = np.zeros(19)
        np.add.at(probs_m1, vals_m1, P_flat)
        probs_m1 = probs_m1[probs_m1 > 0]
        H_m1 = -np.sum(probs_m1 * np.log2(probs_m1))
        
        # H_0: x
        proj_0 = x_coords.ravel()
        probs_0 = np.zeros(n)
        np.add.at(probs_0, proj_0, P_flat)
        probs_0 = probs_0[probs_0 > 0]
        H_0 = -np.sum(probs_0 * np.log2(probs_0))
        
        # H_1: x + y
        proj_1 = (x_coords + y_coords).ravel()
        # Range: 0 to 18
        probs_1 = np.zeros(19)
        np.add.at(probs_1, proj_1, P_flat)
        probs_1 = probs_1[probs_1 > 0]
        H_1 = -np.sum(probs_1 * np.log2(probs_1))
        
        # H_2: x + 2*y
        proj_2 = (x_coords + 2 * y_coords).ravel()
        # Range: 0 to 9 + 2*9 = 27
        probs_2 = np.zeros(28)
        np.add.at(probs_2, proj_2, P_flat)
        probs_2 = probs_2[probs_2 > 0]
        H_2 = -np.sum(probs_2 * np.log2(probs_2))
        
        M = max(H_0, H_1, H_2)
        D = H_m1
        return D, M, D / M if M > 1e-9 else 0.0
    
    def objective(theta):
        """
        Minimize negative log-likelihood ratio.
        theta is the logit vector of length 100.
        """
        # Convert logits to probabilities via softmax
        logits = theta.reshape(n, n)
        # Shift for numerical stability
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        P = exp_logits / np.sum(exp_logits)
        
        D, M, ratio = compute_entropies(P)
        if M <= 1e-9:
            return 0.0
        return -ratio
    
    def objective_with_P(theta):
        """Return (objective_value, P) for tracking."""
        logits = theta.reshape(n, n)
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        P = exp_logits / np.sum(exp_logits)
        
        D, M, ratio = compute_entropies(P)
        if M <= 1e-9:
            return 0.0, P
        return -ratio, P
    
    # Generate diverse initializations
    np.random.seed(42)
    
    inits = []
    
    # 1. Uniform
    P_uniform = np.ones((n, n)) / (n * n)
    inits.append(np.log(P_uniform).ravel())
    
    # 2. Concentrated on diagonal (x = y)
    for _ in range(3):
        P_diag = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if x == y:
                    P_diag[y, x] = 1.0
        P_diag += np.random.rand(n, n) * 0.01
        P_diag /= np.sum(P_diag)
        inits.append(np.log(P_diag).ravel())
    
    # 3. Concentrated on anti-diagonal (x + y = 9)
    for _ in range(3):
        P_anti = np.zeros((n, n))
        for y in range(n):
            x = 9 - y
            P_anti[y, x] = 1.0
        P_anti += np.random.rand(n, n) * 0.01
        P_anti /= np.sum(P_anti)
        inits.append(np.log(P_anti).ravel())
    
    # 4. Random sparse: pick k random cells
    for k in [5, 10, 20]:
        P_sparse = np.zeros((n, n))
        cells = np.random.choice(n * n, size=k, replace=False)
        P_sparse.ravel()[cells] = 1.0
        P_sparse /= np.sum(P_sparse)
        P_sparse += 1e-10  # avoid log(0)
        inits.append(np.log(P_sparse).ravel())
    
    # 5. Random dense
    for _ in range(5):
        P_rand = np.random.rand(n, n)
        P_rand /= np.sum(P_rand)
        inits.append(np.log(P_rand).ravel())
    
    # 6. Block structure: 3x3 blocks with different weights
    for _ in range(3):
        P_block = np.zeros((n, n))
        for by in range(0, n, 3):
            for bx in range(0, n, 3):
                weight = np.random.rand()
                P_block[by:min(by+3, n), bx:min(bx+3, n)] = weight
        P_block += 1e-10
        P_block /= np.sum(P_block)
        inits.append(np.log(P_block).ravel())
    
    # Evaluate initial guesses
    best_score = -1
    best_P = None
    best_theta = None
    
    for i, theta_init in enumerate(inits):
        if time.time() > deadline - 60:
            break
        
        val, P = objective_with_P(theta_init)
        score = -val
        if score > best_score:
            best_score = score
            best_P = P.copy()
            best_theta = theta_init.copy()
    
    # Run optimization from each init
    options_lbfgs = {
        'maxiter': 500,
        'ftol': 1e-12,
        'gtol': 1e-8,
        'maxfun': 2000
    }
    
    for i, theta_init in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                objective,
                theta_init,
                method='L-BFGS-B',
                options=options_lbfgs
            )
            
            val, P = objective_with_P(result.x)
            score = -val
            if score > best_score:
                best_score = score
                best_P = P.copy()
                best_theta = result.x.copy()
        except Exception:
            pass
    
    # If we have time, do a second round with more iterations from the best
    if best_theta is not None and time.time() < deadline - 60:
        try:
            result = minimize(
                objective,
                best_theta,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000}
            )
            val, P = objective_with_P(result.x)
            score = -val
            if score > best_score:
                best_score = score
                best_P = P.copy()
        except Exception:
            pass
    
    # Another polish round if time permits
    if best_theta is not None and time.time() < deadline - 30:
        try:
            result = minimize(
                objective,
                best_theta,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 20000}
            )
            val, P = objective_with_P(result.x)
            score = -val
            if score > best_score:
                best_score = score
                best_P = P.copy()
        except Exception:
            pass
    
    # Ensure validity
    if best_P is None:
        best_P = np.ones((n, n)) / (n * n)
    
    best_P = np.clip(best_P, 0, None)
    total = np.sum(best_P)
    if total <= 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P = best_P / total
    
    # Final validation
    D, M, final_score = compute_entropies(best_P)
    if M <= 1e-9 or np.any(np.isnan(best_P)) or np.any(np.isinf(best_P)) or np.any(best_P < -1e-9):
        best_P = np.ones((n, n)) / (n * n)
    
    return best_P

# EVOLVE-BLOCK-END
