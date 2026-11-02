import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def score(h):
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 0.0
        return l2_sq / (l1 * max_conv)

    def _smoothed_objective(z_flat):
        """
        Objective to minimize: -log(ratio) + barrier for non-negativity.
        z_flat is a real vector of length N (we use real z, h = z^2).
        We work with h = z^2 >= 0.
        """
        if time.time() > deadline - 3.0:
            return 1e10
        z = z_flat
        h = z**2
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 1e10
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 1e10
        ratio = l2_sq / (l1 * max_conv)
        if ratio < 1e-10:
            return 1e10
        # Add a small barrier to keep h away from zero
        barrier = -1e-6 * np.sum(np.log(h + 1e-12))
        return -np.log(ratio) + barrier

    def _smoothed_gradient(z_flat):
        """Numerical gradient for the smoothed objective."""
        eps = 1e-7
        grad = np.zeros_like(z_flat)
        f0 = _smoothed_objective(z_flat)
        for i in range(N):
            z_plus = z_flat.copy()
            z_plus[i] += eps
            z_minus = z_flat.copy()
            z_minus[i] -= eps
            grad[i] = (_smoothed_objective(z_plus) - _smoothed_objective(z_minus)) / (2 * eps)
        return grad

    def optimize_from_z(z0, max_iter=200, tol=1e-12):
        """Optimize starting from z0 using scipy minimize."""
        if time.time() > deadline - 5.0:
            return np.zeros(N), 0.0
        
        # Ensure z0 is real
        z0 = np.real(z0)
        h0 = z0**2
        s0 = score(h0)
        
        best_h = h0.copy()
        best_s = s0
        
        try:
            res = minimize(
                _smoothed_objective,
                z0,
                jac=_smoothed_gradient,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxfun': 5000}
            )
            if res.success or res.fun < 1e9:
                z_opt = res.x
                h_opt = z_opt**2
                h_opt = np.maximum(h_opt, 0)
                s_opt = score(h_opt)
                if s_opt > best_s:
                    best_h = h_opt
                    best_s = s_opt
        except Exception:
            pass
        
        return best_h, best_s

    def optimize_amplitudes(S, max_iter=100, tol=1e-12):
        """Optimize amplitudes for a fixed support set S."""
        k = len(S)
        if k == 0:
            return np.zeros(N), 0.0
        
        # Initialize z on support
        z = np.zeros(N)
        z[S] = 1.0 / np.sqrt(k)
        
        h = z**2
        h = h / np.sum(h)
        z = np.sqrt(h)
        
        best_h = h.copy()
        best_s = score(h)
        
        # Use multiplicative updates as a warm start, then L-BFGS-B
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            conv = np.convolve(h, h, mode='full')
            max_conv = np.max(conv)
            if max_conv < 1e-15:
                break
            
            peak_idx = np.argmax(conv)
            shift = peak_idx - (N - 1)
            
            d_peak = np.zeros(N)
            for i in S:
                if 0 <= i + shift < N:
                    if (i + shift) in S:
                        d_peak[i] += h[i + shift]
                if 0 <= i - shift < N:
                    if (i - shift) in S:
                        d_peak[i] += h[i - shift]
            
            l1 = np.sum(h)
            l2_sq = np.sum(h**2)
            
            grad = np.zeros(N)
            for i in S:
                grad[i] = 2.0 * h[i] / l2_sq - 1.0 / l1 - d_peak[i] / max_conv
            
            eta = 0.5
            for i in S:
                h[i] *= np.exp(eta * grad[i])
            
            h_sum = np.sum(h)
            if h_sum < 1e-15:
                break
            h = h / h_sum
            
            s = score(h)
            if s > best_s:
                best_s = s
                best_h = h.copy()
            
            if it > 10 and abs(s - best_s) < tol:
                break
        
        # Now try L-BFGS-B from this point
        z = np.sqrt(h)
        h_lbfgs, s_lbfgs = optimize_from_z(z, max_iter=100)
        if s_lbfgs > best_s:
            best_h = h_lbfgs
            best_s = s_lbfgs
        
        return best_h, best_s

    def find_sidon_set(k, seed=0):
        """Find a k-element Sidon-like set."""
        rng = np.random.RandomState(seed)
        best_set = None
        best_max_conv = np.inf
        
        for attempt in range(30):
            S = sorted(rng.choice(N, k, replace=False).tolist())
            improved = True
            while improved:
                if time.time() > deadline - 5.0:
                    return best_set if best_set is not None else S
                improved = False
                for i in range(k):
                    for j in range(N):
                        if j in S:
                            continue
                        S_new = sorted(S[:i] + [j] + S[i+1:])
                        h_tmp = np.zeros(N)
                        h_tmp[S_new] = 1.0
                        conv = np.convolve(h_tmp, h_tmp, mode='full')
                        mc = np.max(conv)
                        if mc < best_max_conv:
                            best_max_conv = mc
                            best_set = S_new
                            S = S_new
                            improved = True
                            break
                    if improved:
                        break
        
        if best_set is None:
            best_set = sorted(rng.choice(N, k, replace=False).tolist())
        
        return best_set

    # Try various support sizes
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Start with known good Sidon-like sets
    seed_sets = [
        [0, 1, 4, 10, 18, 23, 30],
        [0, 1, 4, 10, 18, 23, 31, 39],
        [0, 1, 4, 10, 18, 23, 31, 40, 47],
        [0, 1, 4, 10, 18, 23, 31, 40, 47, 49],
        [0, 2, 5, 11, 17, 24, 32, 41],
        [0, 1, 3, 7, 12, 18, 25, 33, 42],
    ]
    
    for S in seed_sets:
        if time.time() > deadline - 5.0:
            break
        h_opt, s = optimize_amplitudes(S)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    # Try various sizes with greedy construction
    for k in range(3, 14):
        if time.time() > deadline - 5.0:
            break
        for seed in range(3):
            if time.time() > deadline - 5.0:
                break
            S = find_sidon_set(k, seed=seed)
            h_opt, s = optimize_amplitudes(S, max_iter=50)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
    
    # Random restarts with L-BFGS-B
    rng = np.random.RandomState(42)
    for iter in range(300):
        if time.time() > deadline - 5.0:
            break
        
        k = rng.randint(3, 12)
        S = sorted(rng.choice(N, k, replace=False).tolist())
        
        # Random z initialization
        z = np.zeros(N)
        z[S] = rng.randn(k)
        
        h_opt, s = optimize_from_z(z, max_iter=50)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    # Final coordinate ascent polish
    for _ in range(100):
        if time.time() > deadline - 2.0:
            break
        improved = False
        support = list(np.where(best_h > 1e-12)[0])
        for i in support:
            for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                h_trial = best_h.copy()
                h_trial[i] = max(0, h_trial[i] + delta)
                s = score(h_trial)
                if s > best_score:
                    best_score = s
                    best_h = h_trial.copy()
                    improved = True
                    break
        if not improved:
            break
    
    return list(best_h)
