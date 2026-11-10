import numpy as np
import scipy.optimize
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    start_time = time.time()
    deadline = start_time + 620  # 10s safety margin
    
    # Precompute constants
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n  # Shape (M,)
    A = a[:, None] * y[None, :]  # Shape (n, M)
    
    def calc_area(x_vals):
        """Compute area only (faster, no gradient)."""
        L = x_vals[:, None] + A
        L_sorted = np.sort(L, axis=0)
        d = L_sorted[1:, :] - L_sorted[:-1, :]
        mins = np.minimum(d, w[None, :])
        union_lens = w + np.sum(mins, axis=0)
        return np.mean(union_lens)
    
    def calc_area_and_grad(x_vals):
        """Compute area and gradient."""
        L = x_vals[:, None] + A
        I = np.argsort(L, axis=0)
        L_sorted = np.take_along_axis(L, I, axis=0)
        d = L_sorted[1:, :] - L_sorted[:-1, :]
        overlap_mask = d < w[None, :]
        
        grad_L_sorted = np.zeros_like(L)
        for k in range(n - 1):
            mask = overlap_mask[k, :]
            grad_L_sorted[k, mask] -= 1.0
            grad_L_sorted[k + 1, mask] += 1.0
        
        grad_x = np.zeros(n)
        np.add.at(grad_x, I, grad_L_sorted)
        grad_x /= M
        
        mins = np.minimum(d, w[None, :])
        union_lens = w + np.sum(mins, axis=0)
        area = np.mean(union_lens)
        
        return area, grad_x
    
    # Generate multiple initial guesses
    def keich_initial(i, n):
        k = int(np.log2(n))
        j = np.arange(1, k + 1)
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        return np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    def random_initial(n, seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(-0.5, 1.5, n)
    
    def structured_initial(n, seed):
        """Try a structured pattern based on sine/cosine."""
        rng = np.random.RandomState(seed)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # Add some randomness
        return rng.uniform(-0.5, 1.5, n) * 0.3 + 0.5 * np.sin(t) + 0.5 * np.cos(2 * t)
    
    # Generate candidate initial points
    candidates = []
    
    # Keich construction
    x_keich = np.array([keich_initial(i, n) for i in range(n)])
    candidates.append(x_keich)
    
    # Multiple random starts
    for seed in range(20):
        candidates.append(random_initial(n, seed))
    
    # Some structured starts
    for seed in range(10):
        candidates.append(structured_initial(n, seed))
    
    # Evaluate all candidates
    best_x = candidates[0].copy()
    best_area = calc_area(best_x)
    
    for i, x_cand in enumerate(candidates):
        if time.time() > deadline:
            break
        area = calc_area(x_cand)
        if area < best_area:
            best_area = area
            best_x = x_cand.copy()
    
    # Optimize the best candidate with L-BFGS-B
    def obj_func(x_vals):
        return calc_area(x_vals)
    
    def grad_func(x_vals):
        _, grad = calc_area_and_grad(x_vals)
        return grad
    
    bounds = [(-2.0, 3.0) for _ in range(n)]
    
    try:
        res = scipy.optimize.minimize(
            obj_func,
            best_x,
            method='L-BFGS-B',
            jac=grad_func,
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-7, 'disp': False}
        )
        if res.fun < best_area:
            best_area = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x
    
    # Coordinate descent refinement
    def coord_descent(x, max_rounds=5):
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                x_i_old = x[i]
                
                def obj_1d(xi):
                    x_new = x.copy()
                    x_new[i] = xi
                    return calc_area(x_new)
                
                # Use golden section search for 1D optimization
                # Search in a reasonable range around current value
                lo = x_i_old - 0.3
                hi = x_i_old + 0.3
                
                # First, find a good bracket
                f_mid = obj_1d(x_i_old)
                
                # Try a few points to bracket
                test_points = [x_i_old - 0.2, x_i_old - 0.1, x_i_old, x_i_old + 0.1, x_i_old + 0.2]
                f_vals = [obj_1d(tp) for tp in test_points]
                min_idx = np.argmin(f_vals)
                
                if min_idx > 0 and min_idx < len(test_points) - 1:
                    lo = test_points[min_idx - 1]
                    hi = test_points[min_idx + 1]
                elif min_idx == 0:
                    hi = x_i_old + 0.1
                elif min_idx == len(test_points) - 1:
                    lo = x_i_old - 0.1
                
                # Golden section search
                gr = (np.sqrt(5) - 1) / 2
                a1 = hi - gr * (hi - lo)
                b1 = lo + gr * (hi - lo)
                fa = obj_1d(a1)
                fb = obj_1d(b1)
                
                for _ in range(30):
                    if time.time() > deadline:
                        break
                    if fa < fb:
                        hi = b1
                        b1 = a1
                        fb = fa
                        a1 = hi - gr * (hi - lo)
                        fa = obj_1d(a1)
                    else:
                        lo = a1
                        a1 = b1
                        fa = fb
                        b1 = lo + gr * (hi - lo)
                        fb = obj_1d(b1)
                
                best_1d = (lo + hi) / 2
                f_best = obj_1d(best_1d)
                
                if f_best < f_mid - 1e-12:
                    x[i] = best_1d
                    improved = True
        
        return improved
    
    # Run coordinate descent
    improved = coord_descent(best_x, max_rounds=3)
    
    if time.time() > deadline:
        return best_x
    
    # If we have time, try a second round of optimization from a perturbed start
    if time.time() < deadline - 30:
        # Perturb the current best and optimize again
        rng = np.random.RandomState(42)
        x_perturbed = best_x + rng.normal(0, 0.05, n)
        
        try:
            res2 = scipy.optimize.minimize(
                obj_func,
                x_perturbed,
                method='L-BFGS-B',
                jac=grad_func,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7, 'disp': False}
            )
            if res2.fun < best_area:
                best_area = res2.fun
                best_x = res2.x.copy()
        except Exception:
            pass
        
        if time.time() < deadline - 10:
            coord_descent(best_x, max_rounds=2)
    
    return best_x
