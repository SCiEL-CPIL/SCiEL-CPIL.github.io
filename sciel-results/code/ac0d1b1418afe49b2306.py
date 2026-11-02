import numpy as np
from scipy.optimize import differential_evolution

N = 50

# EVOLVE-BLOCK-START
def construct():
    best_h = None
    best_score = 0.0

    # Candidate 1: Known 2-element solution [2/3, 1/3]
    h_cand1 = [0.0] * N
    h_cand1[0] = 2.0 / 3.0
    h_cand1[1] = 1.0 / 3.0
    score_cand1 = 1.25
    best_h = h_cand1
    best_score = score_cand1

    # Objective function
    def objective(weights):
        w = np.array(weights)
        if np.any(w < 0):
            return 1e10
        s = np.sum(w)
        if s == 0:
            return 1e10
        w = w / s
        q = np.sum(w**2)
        if q == 0:
            return 1e10
        conv = np.convolve(w, w, mode='full')
        m = np.max(conv)
        if m == 0:
            return 1e10
        return m / q

    # Try optimization for K=3 and K=4
    for K in [3, 4]:
        bounds = [(0, 1)] * K
        try:
            res = differential_evolution(objective, bounds, maxiter=50, popsize=10, tol=1e-6, seed=42, polish=False)
            if res.success:
                w_opt = res.x
                s = np.sum(w_opt)
                if s > 0:
                    w_opt = w_opt / s
                    q = np.sum(w_opt**2)
                    conv = np.convolve(w_opt, w_opt, mode='full')
                    m = np.max(conv)
                    score = q / m
                    if score > best_score:
                        best_score = score
                        best_h = [0.0] * N
                        for i in range(K):
                            best_h[i] = w_opt[i]
        except Exception:
            pass

    return best_h
# EVOLVE-BLOCK-END
