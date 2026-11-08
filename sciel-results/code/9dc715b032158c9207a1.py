import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
M = 2 * N - 1  # len(g) = 99
W = 1.0 / (M + 1)  # 1/100
TIME_LIMIT = 150.0
SAFETY_MARGIN = 20.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN

    def _build_y(g):
        y = np.zeros(M + 2)
        y[1:-1] = g
        return y

    def compute_value(h):
        """Exact evaluator metric for given heights."""
        h = np.asarray(h, dtype=np.float64)
        if len(h) != N:
            return 0.0
        if not np.all(np.isfinite(h)):
            return 0.0
        h = np.maximum(0.0, h)
        g = np.convolve(h, h)
        m = len(g)
        y = _build_y(g)
        a = y[:m + 1]
        b = y[1:m + 2]
        l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
        norm_1 = np.sum(np.abs(g)) / (m + 1)
        norm_inf = np.max(np.abs(g))
        if norm_1 * norm_inf == 0:
            return 0.0
        return l2 / (norm_1 * norm_inf)

    def _smooth_value_and_grad(h, beta=50.0):
        """
        Smooth surrogate with analytic gradient.
        Returns (value, gradient) for minimization (negative value).
        """
        h = np.maximum(0.0, h)
        g = np.convolve(h, h)
        m = len(g)
        y = _build_y(g)
        a = y[:m + 1]
        b = y[1:m + 2]
        l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
        norm_1 = np.sum(g) / (m + 1)

        g_max = np.max(g)
        if g_max <= 0 or norm_1 <= 0:
            return 0.0, np.zeros(N)
        lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
        if lse <= 0:
            return 0.0, np.zeros(N)

        val = l2 / (norm_1 * lse)

        # Gradient computation via chain rule
        # d(l2)/dh: l2 = (W/3) * sum(a^2 + a*b + b^2)
        # a[i] = y[i], b[i] = y[i+1] for i in 0..m
        # y[1:-1] = g, so dy/dh is sparse
        # dl2/dy[i] for i in 1..m:
        #   from a[i-1]^2 term: 2*a[i-1]*dy[i-1]/dy[i] = 0 (a[i-1]=y[i-1])
        #   Actually let's compute dl2/dy directly
        # l2 = (W/3) * sum_{i=0}^{m} (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
        # dl2/dy[j] = (W/3) * [2*y[j] + y[j-1] + y[j+1] + 2*y[j] + y[j-1] + y[j+1]]... 
        # Let me be more careful:
        # Term i contributes to y[i]: 2*y[i] + y[i+1]
        # Term i-1 contributes to y[i]: y[i-1] + 2*y[i]
        # So dl2/dy[j] = (W/3) * (2*y[j] + y[j+1] + y[j-1] + 2*y[j]) for interior
        # = (W/3) * (4*y[j] + y[j-1] + y[j+1]) with boundary adjustments
        # y[0]=0, y[m+1]=0
        dl2_dy = np.zeros(M + 2)
        for i in range(m + 1):
            # term i: y[i]^2 + y[i]*y[i+1] + y[i+1]^2
            dl2_dy[i] += 2 * y[i] + y[i + 1]
            dl2_dy[i + 1] += y[i] + 2 * y[i + 1]
        dl2_dy *= (W / 3.0)

        # d(norm_1)/dh: norm_1 = sum(g)/(m+1), g = convolve(h,h)
        # dg/dh[i] = h + reversed(h) for each component... 
        # g[k] = sum_j h[j]*h[j+k-N+1]... actually g = convolve(h,h)
        # g[k] = sum_{j} h[j] * h[j+k-(N-1)] for valid j
        # d(g[k])/d(h[i]) = h[i+k-(N-1)] if 0<=i+k-(N-1)<N else 0
        # So d(norm_1)/d(h[i]) = (1/(m+1)) * sum_k d(g[k])/d(h[i])
        # = (1/(m+1)) * sum_k h[i+k-(N-1)] where valid
        # This is essentially the convolution of h with ones, scaled
        # Actually sum_k d(g[k])/d(h[i]) = sum of h over all positions that pair with h[i]
        # = sum_j h[j] (since each h[i] pairs with all h[j] in the convolution)
        # Wait no. g[k] = sum_j h[j] h[j+k-N+1]. 
        # d(g[k])/d(h[i]) = h[i+k-N+1] if 0 <= i+k-N+1 < N
        # Sum over k: for fixed i, sum_k h[i+k-N+1] where 0<=i+k-N+1<N
        # Let j = i+k-N+1, so k = j-i+N-1, and 0<=j<N
        # Sum over j=0..N-1 of h[j] = sum(h)
        # So d(norm_1)/d(h[i]) = sum(h)/(m+1) for all i!
        dnorm1_dh = np.full(N, np.sum(h) / (m + 1))

        # d(lse)/dh: lse = g_max + (1/beta)*log(sum(exp(beta*(g-g_max))))
        # = g_max + (1/beta)*log(sum(exp(beta*g))) - g_max
        # Let S = sum(exp(beta*g)), lse = (1/beta)*log(S)
        # d(lse)/d(g[k]) = exp(beta*g[k])/S
        # d(lse)/d(h[i]) = sum_k exp(beta*g[k])/S * d(g[k])/d(h[i])
        exp_beta_g = np.exp(beta * (g - g_max))
        S = np.sum(exp_beta_g)
        weights = exp_beta_g / S  # d(lse)/d(g[k])
        # d(lse)/d(h[i]) = sum_k weights[k] * d(g[k])/d(h[i])
        # d(g[k])/d(h[i]) = h[i+k-N+1] if valid
        # So d(lse)/d(h[i]) = sum_k weights[k] * h[i+k-N+1] (valid range)
        # This is a weighted convolution
        dlse_dh = np.zeros(N)
        for i in range(N):
            # k ranges such that 0 <= i+k-N+1 < N, i.e., N-1-i <= k <= N-1
            k_min = max(0, N - 1 - i)
            k_max = min(M - 1, N - 1 + i)  # M-1 = 2N-2, but k max is N-1+i
            if k_min <= k_max:
                j_range = np.arange(i + k_min - N + 1, i + k_max - N + 2)
                dlse_dh[i] = np.sum(weights[k_min:k_max+1] * h[j_range])
        dlse_dh /= beta

        # val = l2 / (norm_1 * lse)
        # d(val)/d(h[i]) = [dl2/dh[i] * norm_1 * lse - l2 * (dnorm1/dh[i] * lse + norm_1 * dlse/dh[i])] / (norm_1 * lse)^2
        # dl2/dh[i] = sum_j dl2_dy[j] * dy[j]/dh[i]
        # y[1:-1] = g, so dy[j]/dh[i] = d(g[j-1])/dh[i] = h[i+j-1-N+1] = h[i+j-N] if valid
        dl2_dh = np.zeros(N)
        for i in range(N):
            # j ranges 1..m such that 0 <= i+j-N < N
            j_min = max(1, N - i)
            j_max = min(M, N - 1 + i)
            if j_min <= j_max:
                g_idx = slice(j_min - 1, j_max)  # g indices
                h_idx = slice(i + j_min - N, i + j_max - N + 1)
                dl2_dh[i] = np.sum(dl2_dy[j_min:j_max+1] * h[h_idx])

        denom = norm_1 * lse
        if denom == 0:
            return 0.0, np.zeros(N)

        dval_dh = (dl2_dh * norm_1 * lse - val * (dnorm1_dh * lse + norm_1 * dlse_dh)) / (denom * denom)
        
        return -val, -dval_dh

    def _generate_structured_starts():
        """Generate diverse structured starting profiles."""
        x = np.linspace(0, 1, N)
        starts = []
        starts.append(np.ones(N))
        starts.append(1 - 2 * np.abs(x - 0.5))
        starts.append(np.maximum(0, 1 - 4 * (x - 0.5)**2))
        starts.append(0.5 * (1 + np.cos(2 * np.pi * x)))
        starts.append(np.exp(-((x - 0.5)**2) / (2 * 0.15**2)))
        p1 = np.exp(-((x - 0.25)**2) / (2 * 0.1**2))
        p2 = np.exp(-((x - 0.75)**2) / (2 * 0.1**2))
        starts.append(p1 + p2)
        starts.append(np.sqrt(np.maximum(0, 1 - (2*x - 1)**2)))
        starts.append(np.maximum(0, 1 - x)**0.5)
        
        rng = np.random.RandomState(42)
        for _ in range(15):
            K = rng.randint(2, 6)
            coeffs = rng.randint(1, 8, size=K)
            amps = rng.uniform(0.3, 2.0, size=K)
            h = np.zeros(N)
            for c, a in zip(coeffs, amps):
                h += a * np.cos(2 * np.pi * c * np.arange(N) / N)
            h = np.maximum(0, h)
            if np.max(h) > 0:
                h = h / np.max(h)
            starts.append(h)
        
        return starts

    def _plateau_search(rng, best_h, best_val, deadline, max_iter=5000):
        """
        Simulated annealing over plateau assignments.
        K distinct levels, assign each of N positions to a level.
        Variables: levels (K values) + assignment (N values in 0..K-1)
        """
        h_cur = best_h.copy()
        val_cur = compute_value(h_cur)
        h_best = h_cur.copy()
        val_best = val_cur
        
        T_start = 0.1
        T_end = 0.001
        alpha = 0.995  # cooling rate
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            T = T_start * (T_end / T_start) ** (it / max_iter)
            
            # Random move: either change a level value or reassign a position
            move_type = rng.random()
            
            if move_type < 0.3:
                # Change one level value
                # Identify distinct levels
                unique_levels = np.unique(np.round(h_cur, 6))
                if len(unique_levels) == 0:
                    continue
                # Pick a random position and perturb its value
                idx = rng.randint(N)
                h_new = h_cur.copy()
                h_new[idx] = max(0, h_new[idx] + rng.normal(0, 0.1))
            elif move_type < 0.6:
                # Swap two positions' values
                i, j = rng.choice(N, 2, replace=False)
                h_new = h_cur.copy()
                h_new[i], h_new[j] = h_cur[j], h_cur[i]
            else:
                # Set a random block to a new value
                block_len = rng.randint(3, 10)
                start = rng.randint(0, N - block_len + 1)
                h_new = h_cur.copy()
                h_new[start:start+block_len] = max(0, rng.uniform(0.1, 1.5))
            
            val_new = compute_value(h_new)
            
            # Accept if better or with probability exp(-(val_cur-val_new)/T) for worse
            if val_new > val_cur:
                h_cur = h_new
                val_cur = val_new
                if val_new > val_best:
                    val_best = val_new
                    h_best = h_new.copy()
            else:
                delta = val_cur - val_new
                if delta > 0 and rng.random() < np.exp(-delta / T):
                    h_cur = h_new
                    val_cur = val_new
        
        return h_best, val_best

    def _lbfgs_optimize(x0, beta, maxiter=200):
        """L-BFGS-B with analytic gradient."""
        bounds = [(0, None)] * N
        
        def obj_and_grad(x):
            val, grad = _smooth_value_and_grad(x, beta)
            return val, grad
        
        try:
            res = minimize(
                obj_and_grad,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                jac=True,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            return res.x
        except Exception:
            return x0

    # Initialize best
    best_h = np.ones(N)
    best_val = compute_value(best_h)

    def track(h):
        nonlocal best_h, best_val
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # Generate and evaluate starts
    starts = _generate_structured_starts()
    for s in starts:
        track(s)

    rng = np.random.RandomState(123)

    # Phase 1: L-BFGS-B from multiple starts with gradient
    for beta in [30.0, 50.0, 100.0]:
        if time.time() > deadline:
            break
        
        for s in starts[:8]:
            if time.time() > deadline:
                break
            h_opt = _lbfgs_optimize(s, beta, maxiter=150)
            track(h_opt)

    # Phase 2: Plateau-based simulated annealing
    if time.time() < deadline - 30:
        h_sa, val_sa = _plateau_search(rng, best_h, best_val, deadline, max_iter=8000)
        track(h_sa)

    # Phase 3: L-BFGS-B polish from best
    for beta in [50.0, 100.0, 200.0]:
        if time.time() > deadline:
            break
        h_opt = _lbfgs_optimize(best_h, beta, maxiter=200)
        track(h_opt)

    # Phase 4: More plateau search if time remains
    if time.time() < deadline - 15:
        h_sa2, val_sa2 = _plateau_search(rng, best_h, best_val, deadline, max_iter=5000)
        track(h_sa2)

    # Final L-BFGS polish
    if time.time() < deadline - 5:
        h_final = _lbfgs_optimize(best_h, 200.0, maxiter=100)
        track(h_final)

    # Final cleanup
    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    
    return [float(h) for h in best_h]
# EVOLVE-BLOCK-END
