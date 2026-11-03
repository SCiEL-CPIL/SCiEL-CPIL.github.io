import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def _autoconv_fft(h):
        """Compute full linear autoconvolution using FFT."""
        n = len(h)
        size = 2 * n - 1
        fsize = 1
        while fsize < size:
            fsize *= 2
        H = np.fft.rfft(h, fsize)
        ac = np.fft.irfft(H * H, fsize)
        return ac[:size]

    def _score(h):
        """Compute the true objective value: max|autoconv| / (sum h)^2."""
        s = np.sum(h)
        if abs(s) < 1e-9:
            return 1e10
        ac = _autoconv_fft(h)
        return np.max(np.abs(ac)) / (s * s)

    def _solve_qp_amplitudes(signs, time_budget=5.0):
        """
        Solve for continuous amplitudes a given binary signs.
        Minimize ||h * h||_2 subject to sum(h) = 1 and h_i = signs_i * a_i, a_i >= 0.
        This is equivalent to minimizing sum_k (h*h)_k^2.
        Since h is fixed in sign pattern, we can use L-BFGS-B on log(a) or just a directly
        with penalty for small a to avoid numerical issues, or treat it as a QP.
        
        Actually, let's use scipy.optimize.minimize with L-BFGS-B on the amplitudes a.
        Objective: f(a) = || h(a) * h(a) ||_2^2
        Constraint: sum(a * signs) = 1 -> sum(a * signs) - 1 = 0
        Bounds: a_i >= 0
        
        To handle the constraint efficiently, we can parameterize a_{N-1} = (1 - sum_{i<N-1} a_i * signs_i) / signs_{N-1}.
        But this requires signs_{N-1} != 0, which is true.
        
        However, L-BFGS-B doesn't handle equality constraints well directly without transformation.
        Let's use the transformation.
        Let h = signs * a.
        sum(h) = sum(signs * a) = 1.
        Let a_N = (1 - sum_{i=1}^{N-1} signs_i a_i) / signs_N.
        We optimize over a_1..a_{N-1}.
        
        Wait, if signs_N is negative, a_N must be positive, so the numerator must be negative.
        This creates complex bound constraints on the hyperplane.
        
        Alternative: Use scipy.optimize.minimize with method='SLSQP' or 'trust-constr' which handle constraints.
        Or, simpler: Use the Lagrangian approach or just minimize with penalty.
        
        Given time constraints, let's try a simpler approach: 
        Scale the signs to have sum 1, then optimize the magnitudes to reduce autoconvolution peak?
        
        Actually, the strategy says "minimize l2-norm of autoconvolution".
        Let's stick to the QP formulation but implement it via gradient descent on the log-amplitudes or just amplitudes with normalization.
        
        Let's try a direct optimization of h with the constraint that sign(h_i) = signs_i.
        We can do this by projecting h onto the feasible set (signs fixed, sum=1) after each gradient step.
        Feasible set: h_i = signs_i * |h_i|, sum(h_i) = 1.
        
        Let's use a projected gradient method.
        Objective: f(h) = max_k |(h*h)_k|  (or soft max for gradient)
        Gradient of max is sparse, so soft max is better.
        
        Let's use the soft-max objective from inspiration 1.
        """
        if time.time() > start_time + time_budget:
            return signs / np.sum(signs)  # Fallback
        
        # Initial guess: normalize signs to sum 1
        h = signs.astype(float)
        s = np.sum(h)
        if abs(s) < 1e-6:
            return np.ones(N) / N
        h = h / s
        
        def soft_max_abs(ac, beta=50.0):
            abs_ac = np.abs(ac)
            m = np.max(abs_ac)
            if m < 1e-15:
                return 0.0
            scaled = beta * (abs_ac - m)
            scaled = np.clip(scaled, -50, 50)
            log_sum = np.log(np.sum(np.exp(scaled)))
            return m + log_sum / beta

        def objective(h_vec):
            # Enforce signs and normalization during evaluation? 
            # No, we pass h directly, but we need to ensure signs are respected.
            # We will project in the gradient step or use a parameterization.
            # Let's use the parameterization: h_i = signs_i * exp(z_i).
            # Then sum(h) = sum(signs * exp(z)) = 1.
            # This is tricky to enforce directly in L-BFGS-B without constraints.
            
            # Let's stick to the transformation: 
            # h_{N-1} = (1 - sum_{i<N-1} h_i) / 1 ? No, signs are involved.
            # sum_{i=0}^{N-1} signs_i h_i = 1 where h_i >= 0.
            # Let's optimize h_i (magnitudes).
            # h_{N-1} = (1 - sum_{i=0}^{N-2} signs_i h_i) / signs_{N-1}
            
            mags = h_vec
            h_full = np.empty(N)
            h_full[:N-1] = signs[:N-1] * mags
            h_full[N-1] = (1.0 - np.sum(signs[:N-1] * mags)) / signs[N-1]
            
            # Check if h_full[N-1] is negative, which is invalid for magnitude.
            # If invalid, return large penalty.
            if h_full[N-1] < 0:
                return 1e10
            
            ac = _autoconv_fft(h_full)
            return soft_max_abs(ac, 50.0)

        # Initial magnitudes
        h_init = signs.astype(float) / np.sum(signs)
        # Magnitudes are |h_init|
        mags_init = np.abs(h_init[:N-1])
        
        try:
            res = minimize(objective, mags_init, method='L-BFGS-B', 
                           options={'maxiter': 100, 'disp': False, 'gtol': 1e-4})
            
            if res.success or res.fun < 1e10:
                mags_opt = res.x
                h_opt = np.empty(N)
                h_opt[:N-1] = signs[:N-1] * mags_opt
                h_opt[N-1] = (1.0 - np.sum(signs[:N-1] * mags_opt)) / signs[N-1]
                
                if h_opt[N-1] >= 0:
                    return h_opt
        except Exception:
            pass
        
        return h

    def _local_search_perturb(h, max_passes=2, deadline=None, step_size=0.05):
        """
        Local optimization: try small perturbations to each element, keep if improves score.
        """
        h = h.copy()
        best_val = _score(h)
        rng = np.random.default_rng(123)
        for _ in range(max_passes):
            if deadline is not None and time.time() > deadline:
                break
            improved = False
            for i in range(N):
                if deadline is not None and time.time() > deadline:
                    break
                old_val = h[i]
                # Try several perturbation sizes
                for delta in [step_size, -step_size, 2*step_size, -2*step_size]:
                    h[i] = old_val + delta
                    new_val = _score(h)
                    if new_val < best_val:
                        best_val = new_val
                        improved = True
                        break
                if not improved:
                    h[i] = old_val  # revert if no improvement
            if not improved:
                break
        return h, best_val

    # 1. Generate diverse binary sign templates
    sign_templates = []
    
    # Random signs
    rng = np.random.default_rng(42)
    for _ in range(10):
        signs = rng.choice([-1, 1], size=N)
        if abs(np.sum(signs)) > 10: # Ensure sum is not too close to 0 for stability
            sign_templates.append(signs)
            
    # Structured signs (Chirp-like)
    i = np.arange(N)
    phase = np.pi * i**2 / N
    sign_chirp = np.sign(np.cos(phase))
    sign_templates.append(sign_chirp)
    
    # Alternating
    sign_alt = np.ones(N)
    sign_alt[1::2] = -1
    sign_templates.append(sign_alt)

    best_h = np.ones(N) / N
    best_score = _score(best_h)
    
    # 2. For each template, solve for optimal amplitudes
    for signs in sign_templates:
        if time.time() > deadline - 20:
            break
            
        # Solve QP/Gradient for amplitudes given signs
        h_opt = _solve_qp_amplitudes(signs, time_budget=5.0)
        
        score = _score(h_opt)
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
            
    # 3. Refine the best result with local search
    if time.time() < deadline - 5:
        h_refined, val_refined = _local_search_perturb(best_h, max_passes=3, deadline=deadline-2, step_size=0.02)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
            
    return best_h
