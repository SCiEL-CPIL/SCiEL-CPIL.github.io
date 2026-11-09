import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, dual_annealing

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 615  # safety margin before 630s kill

    n = 10
    ys = np.arange(n)
    xs = np.arange(n)
    X, Y = np.meshgrid(xs, ys)

    # Precompute bin indices (flattened)
    XY_diff = (X - Y + (n - 1)).flatten()   # 0..18
    X_flat = X.flatten()                     # 0..9
    XY_sum = (X + Y).flatten()               # 0..18
    X2Y = (X + 2 * Y).flatten()              # 0..27

    def compute_ratio(P):
        P = np.maximum(P, 0.0)
        total = P.sum()
        if total <= 1e-9:
            return 0.0
        P = P / total
        Pf = P.flatten()

        def entropy(indices, minlength):
            dist = np.bincount(indices, weights=Pf, minlength=minlength)
            nz = dist[dist > 1e-15]
            if len(nz) == 0:
                return 0.0
            return -np.sum(nz * np.log2(nz))

        H_diff = entropy(XY_diff, 2 * n - 1)
        H_x = entropy(X_flat, n)
        H_sum = entropy(XY_sum, 2 * n - 1)
        H_2y = entropy(X2Y, 3 * n - 1)

        M = max(H_x, H_sum, H_2y)
        if M <= 1e-9:
            return 0.0
        return H_diff / M

    def objective(params):
        P = np.exp(params).reshape(n, n)
        total = P.sum()
        if total <= 1e-9:
            return 1e10
        P = P / total
        return -compute_ratio(P)

    # Generate diverse starting points
    def make_starting_points():
        starts = []
        # Uniform
        P_uni = np.ones((n, n)) / (n * n)
        starts.append(np.log(P_uni).flatten())
        
        # Diagonal concentration
        P_diag = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                P_diag[y, x] = np.exp(-0.5 * abs(x-y))
        P_diag /= P_diag.sum()
        starts.append(np.log(P_diag).flatten())

        # Anti-diagonal concentration
        P_anti = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                P_anti[y, x] = np.exp(-0.5 * abs(x+y - (n-1)))
        P_anti /= P_anti.sum()
        starts.append(np.log(P_anti).flatten())

        # Random
        np.random.seed(42)
        for _ in range(50):
            P_r = np.random.dirichlet(np.ones(n*n))
            starts.append(np.log(np.maximum(P_r, 1e-10)).flatten())
            
        return starts

    starting_points = make_starting_points()
    
    best_ratio = 0.0
    P_best = np.ones((n, n)) / (n * n)

    # Evaluate starts
    for sp in starting_points:
        P = np.exp(sp).reshape(n, n)
        P = np.maximum(P, 0)
        P /= P.sum()
        r = compute_ratio(P)
        if r > best_ratio:
            best_ratio = r
            P_best = P.copy()

    # Use Dual Annealing for global optimization
    # This is often more effective than DE for this dimensionality and non-convexity
    bounds = [(-10, 10)] * (n * n)
    
    # We will run dual_annealing from the best start point
    if time.time() < deadline - 100:
        try:
            # dual_annealing uses L-BFGS-B internally for local optimization
            # maxiter: max iterations of the annealing loop
            # maxfun: max function evaluations for the local optimizer
            # retries: number of restarts if the local optimizer fails
            result = dual_annealing(objective, bounds, 
                                    maxiter=1000, 
                                    maxfun=1000, 
                                    retries=10,
                                    x0=np.log(np.maximum(P_best, 1e-10)).flatten(),
                                    seed=42)
            
            if result.fun < -best_ratio:
                P_opt = np.exp(result.x).reshape(n, n)
                P_opt = np.maximum(P_opt, 0)
                P_opt /= P_opt.sum()
                ratio = compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    P_best = P_opt.copy()
        except Exception:
            pass

    # If time permits, try Differential Evolution as a second global method
    if time.time() < deadline - 150:
        try:
            de_result = differential_evolution(objective, bounds, seed=43,
                                               maxiter=200, popsize=50,
                                               tol=1e-10, mutation=(0.5, 1.5),
                                               recombination=0.8,
                                               init='latinhypercube',
                                               polish=True,
                                               updating='immediate')
            if de_result.fun < -best_ratio:
                P_opt = np.exp(de_result.x).reshape(n, n)
                P_opt = np.maximum(P_opt, 0)
                P_opt /= P_opt.sum()
                ratio = compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    P_best = P_opt.copy()
        except Exception:
            pass

    # Final high-precision refinement
    if time.time() < deadline - 20:
        P_best_safe = np.maximum(P_best, 1e-12)
        params_best = np.log(P_best_safe).flatten()
        try:
            result = minimize(objective, params_best, method='L-BFGS-B',
                             options={'maxiter': 2000, 'gtol': 1e-15, 'maxls': 200})
            if result.fun < -best_ratio:
                P_opt = np.exp(result.x).reshape(n, n)
                P_opt = np.maximum(P_opt, 0)
                P_opt /= P_opt.sum()
                ratio = compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    P_best = P_opt.copy()
        except Exception:
            pass

        # One more Nelder-Mead polish
        if time.time() < deadline - 10:
            try:
                P_cur = np.maximum(P_best, 1e-12)
                params_cur = np.log(P_cur).flatten()
                result3 = minimize(objective, params_cur, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-16, 'fatol': 1e-16,
                                          'maxfev': 100000, 'adaptive': True})
                if result3.fun < -best_ratio:
                    P_opt = np.exp(result3.x).reshape(n, n)
                    P_opt = np.maximum(P_opt, 0)
                    P_opt /= P_opt.sum()
                    ratio = compute_ratio(P_opt)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        P_best = P_opt.copy()
            except Exception:
                pass

    # Final normalization and clipping
    P_best = np.maximum(P_best, 0)
    total = P_best.sum()
    if total <= 1e-9:
        P_best = np.ones((n, n)) / (n * n)
    else:
        P_best /= total

    return P_best
