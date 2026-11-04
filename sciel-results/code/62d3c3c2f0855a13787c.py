"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
d = 1.0 / (2 * N)  # interval width

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin below 120s limit
    
    def score(h):
        """Compute ratio ||f||_2^2 / (||f||_1 * ||f*f||_inf) for step function."""
        h = np.asarray(h, dtype=float)
        sum_sq = np.sum(h**2)
        sum_h = np.sum(h)
        if sum_h < 1e-15 or sum_sq < 1e-30:
            return 0.0
        ac = np.correlate(h, h, mode='full')
        max_ac = np.max(ac)
        if max_ac < 1e-30:
            return 0.0
        return sum_sq / (d * sum_h * max_ac)

    def smooth_score_neg(h, alpha=100.0):
        """
        Smooth negative objective for optimization.
        Uses soft-max (log-sum-exp) to approximate the max in autoconvolution.
        Minimizing this maximizes the ratio.
        """
        h = np.asarray(h, dtype=float)
        h = np.maximum(0, h)
        sum_sq = np.sum(h**2)
        sum_h = np.sum(h)
        if sum_h < 1e-15 or sum_sq < 1e-30:
            return 1e10
        
        ac = np.correlate(h, h, mode='full')
        # Soft max of ac: (1/alpha) * log(sum(exp(alpha * ac)))
        # To avoid overflow, subtract max
        ac_max = np.max(ac)
        if ac_max < 1e-30:
            return 1e10
        
        # Use a moderate alpha to keep it smooth but close to hard max
        # If alpha is too high, it becomes non-differentiable in practice due to precision
        # If too low, it's too soft. 100-500 is usually good for this scale.
        # Let's use alpha=200.
        a = 200.0
        shifted = a * (ac - ac_max)
        # Clip shifted to avoid underflow issues in exp, though exp(-large) is 0
        # log(sum(exp(shifted))) + ac_max
        # sum(exp(shifted)) is safe if shifted <= 0
        sum_exp = np.sum(np.exp(shifted))
        if sum_exp < 1e-300:
            sum_exp = 1e-300
        log_sum_exp = np.log(sum_exp)
        soft_max = ac_max + log_sum_exp / a
        
        if soft_max < 1e-30:
            return 1e10
            
        ratio = sum_sq / (d * sum_h * soft_max)
        return -ratio

    def try_candidate(h, best_h, best_score):
        """Try a candidate and return updated best."""
        h = np.asarray(h, dtype=float)
        h = np.maximum(0, h)
        if np.sum(h) < 1e-15:
            return h, best_score
        s = score(h)
        if s > best_score:
            return h.copy(), s
        return best_h, best_score

    best_h = np.zeros(N)
    best_h[N // 2] = 1.0
    best_score = score(best_h)

    # Phase 1: Try all single-point spikes
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        best_h, best_score = try_candidate(h, best_h, best_score)

    # Phase 2: Try two-point spikes at various separations
    for i in range(N):
        if time.time() > deadline - 40:
            break
        for j in range(i + 1, N):
            if time.time() > deadline - 40:
                break
            for ratio in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]:
                h = np.zeros(N)
                h[i] = 1.0
                h[j] = ratio
                best_h, best_score = try_candidate(h, best_h, best_score)

    # Phase 3: Try small concentrated shapes (width 1-3)
    for center in range(N):
        if time.time() > deadline - 30:
            break
        for width in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            t = np.arange(N)
            h = np.exp(-0.5 * ((t - center) / width)**2)
            best_h, best_score = try_candidate(h, best_h, best_score)
            h = np.maximum(0, 1 - np.abs(t - center) / width)
            best_h, best_score = try_candidate(h, best_h, best_score)
            h = np.where(np.abs(t - center) <= width, 1.0, 0.0)
            best_h, best_score = try_candidate(h, best_h, best_score)
            h = np.exp(-np.abs(t - center) / width)
            best_h, best_score = try_candidate(h, best_h, best_score)

    # Phase 4: Broader parametric search
    centers = np.linspace(0, N-1, 20)
    widths = np.logspace(-1, 2, 30)
    for center in centers:
        if time.time() > deadline - 20:
            break
        for width in widths:
            if time.time() > deadline - 20:
                break
            t = np.arange(N)
            h = np.exp(-0.5 * ((t - center) / width)**2)
            best_h, best_score = try_candidate(h, best_h, best_score)
            h = 1.0 / (1.0 + ((t - center) / width)**2)
            best_h, best_score = try_candidate(h, best_h, best_score)
            for p in [1.0, 2.0, 3.0, 5.0]:
                h = (1.0 + np.abs(t - center) / width) ** (-p)
                best_h, best_score = try_candidate(h, best_h, best_score)

    # Phase 5: Smooth Optimization (L-BFGS-B) from diverse starts
    # This is the key improvement: using a smooth surrogate for the max operation
    # allows gradient-based methods to find local optima that coordinate descent might miss.
    
    if time.time() < deadline - 10:
        # Generate diverse starting points
        starts = []
        
        # 1. Current best
        starts.append(best_h.copy())
        
        # 2. Single spikes
        for i in [0, N//4, N//2, 3*N//4, N-1]:
            h = np.zeros(N)
            h[i] = 1.0
            starts.append(h)
            
        # 3. Bimodal peaks (to test if splitting mass helps)
        for gap in [5, 10, 15]:
            for w in [1.0, 2.0]:
                t = np.arange(N)
                h1 = np.exp(-0.5 * ((t - (N//2 - gap)) / w)**2)
                h2 = np.exp(-0.5 * ((t - (N//2 + gap)) / w)**2)
                starts.append(h1 + h2)
                
        # 4. Random smooth
        rng = np.random.default_rng(42)
        for _ in range(5):
            h = np.abs(rng.normal(1.0, 0.5, N))
            # Simple smoothing
            kernel = np.array([1, 2, 3, 2, 1]) / 9.0
            h = np.convolve(h, kernel, mode='same')
            starts.append(h)
            
        # 5. Uniform
        starts.append(np.ones(N))
        
        bounds = [(0, None)] * N
        
        for h0 in starts:
            if time.time() > deadline - 5:
                break
            
            h0 = np.asarray(h0, dtype=float)
            h0 = np.maximum(0, h0)
            
            # Normalize to a reasonable scale to help optimizer
            s = np.sum(h0)
            if s > 1e-10:
                h0 = h0 * (N / s) # Normalize sum to N
            
            try:
                res = minimize(
                    smooth_score_neg, h0, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                h_opt = np.maximum(0, res.x)
                s_opt = score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    # Phase 6: Coordinate descent polish on the best found so far
    if time.time() < deadline - 5:
        h = best_h.copy()
        for iteration in range(100):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(N):
                old = h[i]
                # Try scaling this coordinate
                for mult in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 5.0]:
                    h[i] = max(0.0, old * mult) if old > 0 else mult
                    s = score(h)
                    if s > best_score + 1e-15:
                        best_score = s
                        best_h = h.copy()
                        improved = True
                    else:
                        h[i] = old
                        break
                h[i] = old
            if not improved:
                break

    # Phase 7: Final high-precision polish with L-BFGS-B if time permits
    if time.time() < deadline - 2:
        h0 = best_h.copy()
        s = np.sum(h0)
        if s > 1e-10:
            h0 = h0 * (N / s)
            try:
                res = minimize(
                    smooth_score_neg, h0, method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                h_opt = np.maximum(0, res.x)
                s_opt = score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    # Ensure nonnegativity and return
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) < 1e-15:
        best_h = np.zeros(N)
        best_h[N // 2] = 1.0

    return list(best_h)

# EVOLVE-BLOCK-END
