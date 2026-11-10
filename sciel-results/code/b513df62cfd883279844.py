# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    
    Models P[y, x] = R[y] * C[x].
    Maximizes D / M where D = H_{-1} and M = max(H_0, H_1, H_2).
    """
    n = 10
    
    # Pre-compute index lists for each sum value v
    # For a given v, which pairs (y, x) satisfy x + k*y = v?
    
    # H_{-1}: v = x - y => x = v + y. y in [0, 9], x in [0, 9].
    # v ranges from -9 to 9.
    indices_neg1 = {}
    for v in range(-(n-1), n):
        ys = []
        for y in range(n):
            x = v + y
            if 0 <= x < n:
                ys.append(y)
        indices_neg1[v] = ys
        
    # H_0: v = x. v in [0, 9].
    # p_x[v] = C[v]. No need for complex index lists, just C itself.
    
    # H_1: v = x + y => x = v - y.
    # v ranges from 0 to 18.
    indices_1 = {}
    for v in range(2*n - 1):
        ys = []
        for y in range(n):
            x = v - y
            if 0 <= x < n:
                ys.append(y)
        indices_1[v] = ys
        
    # H_2: v = x + 2y => x = v - 2y.
    # v ranges from 0 to 9 + 18 = 27.
    indices_2 = {}
    for v in range(28):
        ys = []
        for y in range(n):
            x = v - 2*y
            if 0 <= x < n:
                ys.append(y)
        indices_2[v] = ys

    def compute_entropies(r_log, c_log):
        """
        Computes D and M from log-marginals r_log, c_log.
        """
        # Normalize R and C
        r = np.exp(r_log - np.max(r_log))
        c = np.exp(c_log - np.max(c_log))
        r = r / np.sum(r)
        c = c / np.sum(c)
        
        # H_0: Entropy of C
        p_0 = c
        mask_0 = p_0 > 1e-15
        H_0 = -np.sum(p_0[mask_0] * np.log2(p_0[mask_0]))
        
        # H_1: Entropy of X+Y
        # p_1[v] = sum_{y in indices_1[v]} R[y] * C[v-y]
        p_1 = np.zeros(2*n - 1)
        for v in range(2*n - 1):
            ys = indices_1[v]
            if not ys:
                continue
            xs = v - np.array(ys)
            p_1[v] = np.sum(r[ys] * c[xs])
        mask_1 = p_1 > 1e-15
        H_1 = -np.sum(p_1[mask_1] * np.log2(p_1[mask_1]))
        
        # H_2: Entropy of X+2Y
        p_2 = np.zeros(28)
        for v in range(28):
            ys = indices_2[v]
            if not ys:
                continue
            xs = v - 2 * np.array(ys)
            p_2[v] = np.sum(r[ys] * c[xs])
        mask_2 = p_2 > 1e-15
        H_2 = -np.sum(p_2[mask_2] * np.log2(p_2[mask_2]))
        
        # H_{-1}: Entropy of X-Y
        p_neg1 = np.zeros(2*n - 1) # Index 0..18 corresponds to v=-9..9
        for i, v in enumerate(range(-(n-1), n)):
            ys = indices_neg1[v]
            if not ys:
                continue
            xs = v + np.array(ys)
            p_neg1[i] = np.sum(r[ys] * c[xs])
        mask_neg1 = p_neg1 > 1e-15
        H_neg1 = -np.sum(p_neg1[mask_neg1] * np.log2(p_neg1[mask_neg1]))
        
        M = max(H_0, H_1, H_2)
        
        if M < 1e-9:
            return 1e10 # Penalty
            
        return -H_neg1 / M

    # Objective function for optimizer
    def objective(params):
        r_log = params[:n]
        c_log = params[n:]
        return compute_entropies(r_log, c_log)

    # Initial bounds for log probabilities
    # We can scale them to be in [-5, 5] relative to max
    bounds = [(-5, 5)] * (2*n)
    
    # Global optimization
    # Use differential evolution with a reasonable population size
    # 20 dimensions, popsize 20-30 is standard. 
    # maxiter 100-200.
    result_de = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=150,
        tol=1e-9,
        popsize=20,
        polish=True
    )
    
    best_params = result_de.x
    best_val = result_de.fun
    
    # Local refinement with L-BFGS-B
    # Try a few starting points: the DE result, and some heuristic ones
    starting_points = [
        best_params.copy(),
        np.zeros(2*n),
        np.concatenate([np.ones(n)*0.5, np.ones(n)*0.5]),
        np.concatenate([np.ones(n)*-0.5, np.ones(n)*-0.5]),
    ]
    
    for sp in starting_points:
        res = minimize(
            objective,
            sp,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x

    # Extract optimal marginals
    r_log_opt = best_params[:n]
    c_log_opt = best_params[n:]
    
    r = np.exp(r_log_opt - np.max(r_log_opt))
    c = np.exp(c_log_opt - np.max(c_log_opt))
    r = r / np.sum(r)
    c = c / np.sum(c)
    
    P = np.outer(r, c)
    
    # Ensure valid output
    P = np.clip(P, 0, None)
    total = np.sum(P)
    if total <= 1e-9:
        P = np.ones((n, n)) / (n * n)
    else:
        P = P / total
        
    return P


# EVOLVE-BLOCK-END
