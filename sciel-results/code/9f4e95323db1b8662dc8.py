# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Strategy:
    1. Precompute projection indices for x, y, x-y, x+y, x+2y.
    2. Directly optimize the 100 log-probabilities (unconstrained real numbers) 
       using L-BFGS-B with multiple random restarts.
    3. Two-phase approach: coarse search then fine polishing.
    4. Final coordinate-wise polish for extra precision.
    """
    n = 10
    deadline = time.time() + 580.0  # Safety margin
    
    # Grid coordinates
    # P[y, x] -> y is row, x is col
    ys, xs = np.mgrid[0:n, 0:n]
    
    # Flatten for binning
    d_val = (xs - ys).flatten()
    m0_val = xs.flatten()
    m1_val = (xs + ys).flatten()
    m2_val = (xs + 2 * ys).flatten()
    
    # Unique values and inverse mapping
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)
    
    num_d = len(d_vals)
    num_m0 = len(m0_vals)
    num_m1 = len(m1_vals)
    num_m2 = len(m2_vals)
    
    def compute_entropies_from_p(p_flat):
        """
        p_flat: shape (100,) normalized probabilities
        Returns: D, M
        """
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)
        
        def shannon(p_dist):
            p_nonzero = p_dist[p_dist > 1e-15]
            if len(p_nonzero) == 0:
                return 0.0
            return -np.sum(p_nonzero * np.log2(p_nonzero))
        
        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)
        
        M = max(H_m0, H_m1, H_m2)
        return H_d, M

    def soft_max_ents(H_m0, H_m1, H_m2, temperature=0.1):
        """
        Compute soft max of three entropies using log-sum-exp for smoothness.
        """
        H = np.array([H_m0, H_m1, H_m2])
        H_max = np.max(H)
        if H_max < 1e-15:
            return 0.0
        scaled = (H - H_max) / temperature
        log_sum_exp = H_max / temperature + np.log(np.sum(np.exp(scaled)))
        return temperature * log_sum_exp

    def objective(log_p_params, use_soft_max=True, temperature=0.1):
        """
        log_p_params: shape (100,) log-probabilities (unnormalized)
        Returns: negative ratio -D/M (to be minimized)
        """
        if time.time() > deadline:
            return 1e9
            
        log_p_flat = log_p_params - np.max(log_p_params)
        p_flat = np.exp(log_p_flat)
        sum_p = np.sum(p_flat)
        
        if sum_p < 1e-15:
            return 1e9
            
        p_flat /= sum_p
        
        H_d, M = compute_entropies_from_p(p_flat)
        
        if M < 1e-9:
            return 1e9
        
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)
        
        def shannon(p_dist):
            p_nonzero = p_dist[p_dist > 1e-15]
            if len(p_nonzero) == 0:
                return 0.0
            return -np.sum(p_nonzero * np.log2(p_nonzero))
        
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)
        
        if use_soft_max:
            M_soft = soft_max_ents(H_m0, H_m1, H_m2, temperature)
            if M_soft < 1e-9:
                return 1e9
            return -H_d / M_soft
        else:
            return -H_d / M

    def true_score(params):
        """Compute the true (non-soft) score for a parameter vector."""
        return -objective(params, use_soft_max=False)

    # Initial best: Uniform distribution
    init_params = np.zeros(100)
    best_score = true_score(init_params)
    best_params = init_params.copy()
    
    rng = np.random.default_rng(42)
    
    def run_optimization(start_params, maxiter=2000, temperature=0.1):
        try:
            res = minimize(
                lambda x: objective(x, use_soft_max=True, temperature=temperature), 
                start_params, 
                method='L-BFGS-B', 
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            return res
        except Exception:
            return None

    # Phase 1: Coarse search with diverse initializations
    def gen_gaussian(scale=1.0):
        return rng.normal(0, scale, 100)
    
    def gen_corner_biased():
        start = rng.normal(0, 0.5, 100)
        start[9] += 3.0   # (y=0, x=9)
        start[90] += 3.0  # (y=9, x=0)
        return start
    
    def gen_diagonal_biased():
        start = rng.normal(0, 0.5, 100)
        start[0] += 2.0   # (y=0, x=0)
        start[99] += 2.0  # (y=9, x=9)
        return start
    
    def gen_anti_diagonal():
        start = rng.normal(0, 0.3, 100)
        for y in range(10):
            for x in range(10):
                diff = abs(x - y)
                if diff >= 7:
                    start[y*10+x] += 1.5
        return start
    
    def gen_rank1_perturb():
        a = rng.normal(0, 1.0, 10)
        b = rng.normal(0, 1.0, 10)
        start = np.zeros(100)
        for y in range(10):
            for x in range(10):
                start[y*10+x] = a[y] + b[x]
        return start + rng.normal(0, 0.1, 100)
    
    def gen_concentrated_center():
        start = rng.normal(0, 0.3, 100)
        for y in range(3, 7):
            for x in range(3, 7):
                start[y*10+x] += 2.0
        return start
    
    generators = [
        gen_gaussian, gen_corner_biased, gen_diagonal_biased,
        gen_anti_diagonal, gen_rank1_perturb, gen_concentrated_center
    ]
    
    temperatures_coarse = [0.5, 0.3, 0.7, 0.2, 0.4, 0.6]
    
    num_coarse = 60
    for i in range(num_coarse):
        if time.time() > deadline - 60:
            break
            
        gen = generators[i % len(generators)]
        temp = temperatures_coarse[i % len(temperatures_coarse)]
        
        start = gen()
        res = run_optimization(start, maxiter=1500, temperature=temp)
        if res is not None:
            score = true_score(res.x)
            if score > best_score:
                best_score = score
                best_params = res.x.copy()

    # Phase 2: Fine polishing from best solution
    if time.time() < deadline - 30:
        for temp in [0.1, 0.05, 0.02]:
            if time.time() > deadline - 15:
                break
            res = run_optimization(best_params, maxiter=5000, temperature=temp)
            if res is not None:
                score = true_score(res.x)
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()

    # Phase 3: Coordinate-wise polish
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(100):
                    for step in [0.1, 0.05, 0.01, -0.01, -0.05, -0.1]:
                        trial = best_params.copy()
                        trial[i] += step
                        score = true_score(trial)
                        if score > best_score:
                            best_score = score
                            best_params = trial
                            improved = True
                            break

    # Convert best parameters back to P
    log_p_flat = best_params - np.max(best_params)
    p_flat = np.exp(log_p_flat)
    p_flat /= np.sum(p_flat)
    
    P = p_flat.reshape((n, n))
    
    # Final validation
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P

# EVOLVE-BLOCK-END
