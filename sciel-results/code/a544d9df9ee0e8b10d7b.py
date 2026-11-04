import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Objective: minimize max_k |autocorr(k)| / (sum x)^2
    # Use smooth surrogate: soft-max of |autocorr(k)|
    
    def compute_autocorr(x):
        # Full autocorrelation via FFT
        n = len(x)
        f = np.fft.rfft(x, n=2*n)
        ac = np.fft.irfft(f * np.conj(f), n=2*n)[:n]
        return ac
    
    def objective(params):
        x = np.tanh(params)  # bounded in (-1,1), differentiable
        s = np.sum(x)
        if abs(s) < 1e-8:
            return 1e10
        ac = compute_autocorr(x)
        abs_ac = np.abs(ac)
        # Soft max with temperature
        beta = 50.0
        log_sum_exp = np.log(np.sum(np.exp(beta * abs_ac))) / beta
        ratio = log_sum_exp / (s * s)
        return ratio
    
    def objective_grad(params):
        # Numerical gradient is too slow; use finite differences sparingly
        # Actually, let's just use scipy minimize with numerical gradient
        return None
    
    # Try multiple random restarts
    best_score = np.inf
    best_x = None
    rng = np.random.default_rng(42)
    
    num_restarts = 0
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Initialize: mix of random and structured
        if restart == 0:
            x0 = np.ones(N)
        elif restart == 1:
            x0 = np.random.choice([-1.0, 1.0], size=N)
        else:
            # Smooth random initialization
            freqs = rng.uniform(0, 5, size=N)
            phases = rng.uniform(0, 2*np.pi, size=N)
            x0 = np.cos(freqs * np.arange(N) + phases)
        
        params0 = np.arctanh(np.clip(x0, -0.99, 0.99))
        
        from scipy.optimize import minimize
        try:
            res = minimize(objective, params0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'gtol': 1e-4, 'maxfun': 2000})
            if res.success or res.fun < best_score:
                best_score = res.fun
                best_x = np.tanh(res.x)
        except Exception:
            pass
        
        num_restarts += 1
        if time.time() > deadline - 30:
            break
    
    if best_x is None:
        return np.ones(N)
    
    # Final polish: local coordinate descent on individual elements
    x = best_x.copy()
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            old_val = x[i]
            # Try small perturbations
            for delta in [-0.1, -0.01, 0.01, 0.1]:
                x[i] = np.clip(old_val + delta, -1.0, 1.0)
                s = np.sum(x)
                if abs(s) < 1e-8:
                    x[i] = old_val
                    continue
                ac = compute_autocorr(x)
                cur_score = np.max(np.abs(ac)) / (s * s)
                if cur_score < best_score - 1e-10:
                    best_score = cur_score
                    improved = True
                    break
                x[i] = old_val
        if not improved:
            break
    
    return x
