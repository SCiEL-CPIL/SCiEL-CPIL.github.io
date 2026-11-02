import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _overlap_max(h):
    """Compute max cyclic overlap of h with 1-h, scaled by step width."""
    w = 2.0 / N
    # h is array of length N
    # overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    # = sum_i h[i] - sum_i h[i] * h[(i+k) % N]
    # = N/2 - autocorr[k]
    # Compute all autocorrelations via FFT
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr
    return w * np.max(overlaps)

def _smooth_overlap(h, beta=100.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    w = 2.0 / N
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr
    # log-sum-exp for max approximation
    max_ov = np.max(overlaps)
    if max_ov == 0:
        return 0.0
    log_sum_exp = max_ov + np.log(np.sum(np.exp((overlaps - max_ov) * beta))) / beta
    return w * log_sum_exp

def _smooth_overlap_grad(h, beta=100.0):
    """Gradient of smooth overlap (approximate via finite differences for simplicity)."""
    # For speed, we can use analytic gradient if possible, but finite diff is safer for now
    eps = 1e-6
    grad = np.zeros_like(h)
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += eps
        h_minus = h.copy()
        h_minus[i] -= eps
        # Ensure bounds are respected in finite diff? No, we're inside optimizer bounds.
        # But we need to maintain sum constraint? The optimizer handles constraints.
        # However, the objective function itself doesn't enforce sum=0.5*N.
        # The gradient of the objective w.r.t h is what we need.
        # Note: The constraint is handled by L-BFGS-B? No, L-BFGS-B handles bounds only.
        # We need to handle the equality constraint sum(x)=N/2.
        # L-BFGS-B does NOT handle equality constraints directly.
        # We must use SLSQP or a penalty method or project onto the constraint manifold.
        # Let's stick to SLSQP but with a better objective? Or use a penalty method with L-BFGS-B?
        # The prompt says "Keep the function names and signatures of the seed program."
        # The seed program uses SLSQP.
        # Let's try SLSQP with the smooth objective.
        pass
    return grad

def construct():
    deadline = time.time() + 110  # safety margin
    
    # Initial guess: all 0.5
    h0 = np.full(N, 0.5)
    
    best_h = h0.copy()
    best_val = _overlap_max(h0)
    
    # Try a few different initializations
    inits = [
        np.full(N, 0.5),
        np.tile([0.0, 1.0], N // 2).astype(float),
        np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False)) * 0.3 + 0.5,
        np.random.uniform(0, 1, N),
    ]
    
    # Ensure sum = N/2 for each init
    for i, h_init in enumerate(inits):
        if time.time() > deadline:
            break
        # Project to sum = N/2
        h_init = np.clip(h_init, 0, 1)
        diff = N / 2.0 - np.sum(h_init)
        h_init = h_init + diff / N
        h_init = np.clip(h_init, 0, 1)
        # Re-adjust sum
        for _ in range(10):
            diff = N / 2.0 - np.sum(h_init)
            if abs(diff) < 1e-10:
                break
            h_init = h_init + diff / N
            h_init = np.clip(h_init, 0, 1)
        
        val = _overlap_max(h_init)
        if val < best_val:
            best_val = val
            best_h = h_init.copy()
        
        # Try SLSQP optimization with smooth objective
        # We use the smooth objective for optimization, but evaluate with the true max objective
        # to ensure we are tracking the real metric.
        try:
            # Use a moderate beta for smoothness
            beta = 50.0
            res = minimize(
                lambda x: _smooth_overlap(x, beta), 
                h_init, 
                method='SLSQP',
                bounds=[(0, 1)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 300, 'ftol': 1e-8}
            )
            if time.time() > deadline:
                break
            if res.success or res.fun < best_val:
                h_opt = np.clip(res.x, 0, 1)
                # Enforce sum constraint
                diff = N / 2.0 - np.sum(h_opt)
                h_opt = h_opt + diff / N
                h_opt = np.clip(h_opt, 0, 1)
                val = _overlap_max(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final check
    if time.time() > deadline - 10:
        return best_h
    
    # Final high-precision polish pass with tighter tolerance
    try:
        # Use the true max objective for the final polish to be exact
        res = minimize(
            _overlap_max, 
            best_h, 
            method='SLSQP',
            bounds=[(0, 1)] * N,
            constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        if res.fun < best_val:
            h_opt = np.clip(res.x, 0, 1)
            diff = N / 2.0 - np.sum(h_opt)
            h_opt = h_opt + diff / N
            h_opt = np.clip(h_opt, 0, 1)
            best_h = h_opt
            best_val = _overlap_max(best_h)
    except Exception:
        pass
    
    return best_h
