<<<<<<< SEARCH
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
=======
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_chebyshev
>>>>>>> REPLACE
<<<<<<< SEARCH
    # --- Strategy 1: geometric supports with optimized weights ---
    configs = [
        (16, 0.01, 2.0),
        (20, 0.01, 2.0),
        (24, 0.01, 2.0),
        (28, 0.01, 2.0),
        (32, 0.01, 2.0),
        (16, 0.01, 3.0),
        (20, 0.01, 3.0),
        (24, 0.01, 3.0),
        (28, 0.01, 3.0),
        (16, 0.1, 2.0),
        (20, 0.1, 2.0),
        (24, 0.1, 2.0),
        (20, 0.5, 2.0),
        (24, 0.5, 2.0),
        (16, 1.0, 2.0),
        (20, 1.0, 2.0),
        (24, 1.0, 2.0),
        (16, 0.001, 5.0),
        (20, 0.001, 5.0),
        (24, 0.001, 5.0),
        (16, 0.01, 4.0),
        (20, 0.01, 4.0),
        (24, 0.01, 4.0),
        (16, 0.5, 3.0),
        (20, 0.5, 3.0),
        (24, 0.5, 3.0),
        (16, 0.001, 10.0),
        (20, 0.001, 10.0),
        (16, 0.0001, 20.0),
        (20, 0.0001, 20.0),
        (16, 0.0, 2.0),
        (20, 0.0, 2.0),
        (24, 0.0, 2.0),
    ]
=======
    # --- Strategy 1: geometric supports with optimized weights ---
    configs = [
        (16, 0.01, 2.0),
        (20, 0.01, 2.0),
        (24, 0.01, 2.0),
        (28, 0.01, 2.0),
        (32, 0.01, 2.0),
        (16, 0.01, 3.0),
        (20, 0.01, 3.0),
        (24, 0.01, 3.0),
        (28, 0.01, 3.0),
        (16, 0.1, 2.0),
        (20, 0.1, 2.0),
        (24, 0.1, 2.0),
        (20, 0.5, 2.0),
        (24, 0.5, 2.0),
        (16, 1.0, 2.0),
        (20, 1.0, 2.0),
        (24, 1.0, 2.0),
        (16, 0.001, 5.0),
        (20, 0.001, 5.0),
        (24, 0.001, 5.0),
        (16, 0.01, 4.0),
        (20, 0.01, 4.0),
        (24, 0.01, 4.0),
        (16, 0.5, 3.0),
        (20, 0.5, 3.0),
        (24, 0.5, 3.0),
        (16, 0.001, 10.0),
        (20, 0.001, 10.0),
        (16, 0.0001, 20.0),
        (20, 0.0001, 20.0),
        (16, 0.0, 2.0),
        (20, 0.0, 2.0),
        (24, 0.0, 2.0),
    ]
    
    # --- Strategy 0: Chebyshev node supports (maximally separated points) ---
    # Chebyshev roots scaled to [0, L] give well-separated points that
    # minimize floating-point boundary issues in the strict inequality.
    cheb_configs = [
        (16, 1.0),
        (20, 1.0),
        (24, 1.0),
        (16, 5.0),
        (20, 5.0),
        (24, 5.0),
        (16, 0.1),
        (20, 0.1),
        (24, 0.1),
    ]
    
    for n, L in cheb_configs:
        if time.time() > deadline - 120:
            break
        # Chebyshev roots of T_n are in (-1, 1); scale to [0, L]
        roots, _ = roots_chebyshev(n, kind='first')
        support = 0.5 * L * (roots + 1.0)
        support = np.sort(support)
        
        if np.any(support < -1e-9) or not np.all(np.isfinite(support)):
            continue
        
        w_uniform = np.ones(n) / n
        P_uniform = _compute_P(support, w_uniform)
        if P_uniform < best_P * 0.3:
            continue
        
        remaining = deadline - time.time() - 10
        if remaining < 10:
            break
        max_opt_time = min(60.0, remaining * 0.5)
        w_opt, P_opt = _optimize_weights_nm(support, max_time=max_opt_time, seed=42)
        
        if P_opt > best_P:
            best_P = P_opt
            best_support = support.copy()
            best_weights = w_opt.copy()
>>>>>>> REPLACE
<<<<<<< SEARCH
    # --- Strategy 2: bimodal support family (small mass + geometric tail) ---
=======
    # --- Strategy 2: bimodal support family (small mass + geometric tail) ---
>>>>>>> REPLACE
<<<<<<< SEARCH
    # --- Strategy 3: coarse-to-fine support refinement + differential evolution ---
=======
    # --- Strategy 3: coarse-to-fine support refinement + differential evolution ---
>>>>>>> REPLACE
<<<<<<< SEARCH
    # --- Strategy 4: support scale search on the best geometric family ---
=======
    # --- Strategy 4: support scale search on the best geometric family ---
>>>>>>> REPLACE
