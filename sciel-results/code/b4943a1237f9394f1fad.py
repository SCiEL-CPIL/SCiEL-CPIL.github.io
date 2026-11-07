import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    def compute_value(h):
        """Compute the objective value for a given height array."""
        h_safe = np.clip(h, 0.0, 1.0)
        corr = np.correlate(h_safe, 1.0 - h_safe, mode='full')
        max_corr = np.max(corr)
        return max_corr / N * 2.0

    def soft_max(corr, beta):
        """Smooth approximation of max via log-sum-exp."""
        c = corr * beta
        c_max = np.max(c)
        if c_max > 500:
            return (c_max + np.log(np.sum(np.exp(c - c_max)))) / beta
        return (c_max + np.log(np.sum(np.exp(c - c_max)))) / beta

    def smooth_value(h, beta=200.0):
        """Smooth surrogate for the objective using log-sum-exp."""
        h_safe = np.clip(h, 0.0, 1.0)
        corr = np.correlate(h_safe, 1.0 - h_safe, mode='full')
        sm = soft_max(corr, beta)
        return sm / N * 2.0

    def smooth_value_grad(h, beta=200.0):
        """Gradient of smooth objective w.r.t. h via finite differences."""
        h_safe = np.clip(h, 0.0, 1.0)
        corr = np.correlate(h_safe, 1.0 - h_safe, mode='full')
        c = corr * beta
        c_max = np.max(c)
        exp_c = np.exp(c - c_max)
        weights = exp_c / np.sum(exp_c)
        # d(corr)/d(h) for each shift
        # corr[k] = sum_i h[i] * (1-h[i+k-N+100]) for appropriate indices
        # Full correlation: corr[m] = sum_i h[i] * (1-h[i+m-N+1]) for m in [-(N-1), N-1]
        # Actually np.correlate(a, v, 'full'): result[m] = sum_n a[n+m] * conj(v[n])
        # Here a=h, v=1-h, so corr[m] = sum_n h[n+m] * (1-h[n]) for valid n
        # Gradient: d/dh[j] corr[m] = (1-h[j-m]) if j+m in [0,N-1] plus -h[j+m] if j in [0,N-1]
        # Simpler: use finite differences for reliability
        eps = 1e-7
        grad = np.zeros(N)
        for i in range(N):
            hp = h_safe.copy()
            hp[i] += eps
            hp = np.clip(hp, 0.0, 1.0)
            cp = np.correlate(hp, 1.0 - hp, mode='full')
            fp = soft_max(cp, beta) / N * 2.0
            
            hm = h_safe.copy()
            hm[i] -= eps
            hm = np.clip(hm, 0.0, 1.0)
            cm = np.correlate(hm, 1.0 - hm, mode='full')
            fm = soft_max(cm, beta) / N * 2.0
            
            grad[i] = (fp - fm) / (2.0 * eps)
        return grad

    def project_sum(h, target=50.0):
        """Project h to have sum = target while staying in [0,1]."""
        h = np.clip(h, 0.0, 1.0)
        for _ in range(50):
            err = np.sum(h) - target
            if abs(err) < 1e-14:
                break
            h = h - err / N
            h = np.clip(h, 0.0, 1.0)
        # Final exact adjustment for free variables
        err = np.sum(h) - target
        if abs(err) > 1e-14:
            # Adjust only interior points
            interior = (h > 1e-6) & (h < 1.0 - 1e-6)
            if np.sum(interior) > 0:
                h[interior] -= err / np.sum(interior)
                h = np.clip(h, 0.0, 1.0)
        return h

    def parametric_objective(params, beta=200.0):
        """
        Objective function for low-dimensional parametric model using smooth surrogate.
        params: [A1, k1, phase1, A2, k2, phase2, A3, k3, phase3]
        """
        A1, k1, p1, A2, k2, p2, A3, k3, p3 = params
        
        x = np.arange(N)
        h = 0.5
        h += A1 * np.cos(2 * np.pi * k1 * x / N + p1)
        h += A2 * np.cos(2 * np.pi * k2 * x / N + p2)
        h += A3 * np.cos(2 * np.pi * k3 * x / N + p3)
        
        h = project_sum(h)
        return smooth_value(h, beta)

    def parametric_objective_grad(params, beta=200.0):
        """Gradient of parametric objective using central finite differences."""
        grad = np.zeros(len(params))
        eps = 1e-7
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            f_plus = parametric_objective(params_plus, beta)
            f_minus = parametric_objective(params_minus, beta)
            grad[i] = (f_plus - f_minus) / (2.0 * eps)
        return grad

    # Best solution tracking
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)
    
    # Generate diverse starting points in parameter space
    np.random.seed(12345)
    
    param_starts = []
    
    # Start with zero amplitudes (uniform)
    param_starts.append(np.zeros(9))
    
    # Single sinusoid starts
    for A in [0.2, 0.3, 0.4, 0.45, 0.5]:
        for k in [1, 2, 3, 4, 5]:
            for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
                param_starts.append(np.array([A, k, phase, 0, 0, 0, 0, 0, 0]))
    
    # Two sinusoid starts
    for A1 in [0.2, 0.3, 0.4]:
        for k1 in [1, 2, 3]:
            for A2 in [0.1, 0.2, 0.3]:
                for k2 in [2, 3, 4, 5]:
                    if k1 != k2:
                        for p1 in [0, np.pi/2]:
                            for p2 in [0, np.pi/2]:
                                param_starts.append(np.array([A1, k1, p1, A2, k2, p2, 0, 0, 0]))
    
    # Three sinusoid starts (sparse)
    for _ in range(50):
        A1 = np.random.uniform(0.1, 0.4)
        k1 = np.random.randint(1, 6)
        p1 = np.random.uniform(0, 2*np.pi)
        A2 = np.random.uniform(0.1, 0.3)
        k2 = np.random.randint(1, 6)
        p2 = np.random.uniform(0, 2*np.pi)
        A3 = np.random.uniform(0.1, 0.2)
        k3 = np.random.randint(1, 6)
        p3 = np.random.uniform(0, 2*np.pi)
        param_starts.append(np.array([A1, k1, p1, A2, k2, p2, A3, k3, p3]))
    
    # Optimize from each starting point in parameter space using smooth surrogate
    beta_seq = [100.0, 200.0, 400.0, 800.0]  # anneal beta for tighter max approximation
    
    for param_start in param_starts:
        if time.time() > deadline - 10.0:
            break
        
        try:
            # Multi-stage beta annealing: start smooth, tighten
            current_params = param_start.copy()
            for beta in beta_seq:
                if time.time() > deadline - 10.0:
                    break
                result = minimize(
                    lambda p: parametric_objective(p, beta),
                    current_params,
                    method='L-BFGS-B',
                    bounds=[(0, 0.5), (0.5, 6), (0, 2*np.pi),
                            (0, 0.5), (0.5, 6), (0, 2*np.pi),
                            (0, 0.5), (0.5, 6), (0, 2*np.pi)],
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current_params = result.x
            
            # Evaluate final solution with exact max
            params_opt = current_params
            A1, k1, p1, A2, k2, p2, A3, k3, p3 = params_opt
            x = np.arange(N)
            h_opt = 0.5
            h_opt += A1 * np.cos(2 * np.pi * k1 * x / N + p1)
            h_opt += A2 * np.cos(2 * np.pi * k2 * x / N + p2)
            h_opt += A3 * np.cos(2 * np.pi * k3 * x / N + p3)
            h_opt = project_sum(h_opt)
            
            val_opt = compute_value(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Now do high-dimensional local refinement using smooth surrogate
    if time.time() < deadline - 15.0:
        h_cur = best_h.copy()
        val_cur = best_val
        
        # Gradient descent with adaptive step size using smooth surrogate
        lr = 0.02
        beta_refine = 500.0
        
        for step in range(800):
            if time.time() > deadline - 15.0:
                break
            
            # Compute gradient of smooth objective
            grad = smooth_value_grad(h_cur, beta_refine)
            
            # Gradient step
            h_new = h_cur - lr * grad
            h_new = project_sum(h_new)
            
            val_new = compute_value(h_new)
            val_smooth_new = smooth_value(h_new, beta_refine)
            
            if val_new < val_cur:
                h_cur = h_new
                val_cur = val_new
                lr *= 0.998
            else:
                lr *= 0.85
            
            if lr < 1e-8:
                break
        
        if val_cur < best_val:
            best_val = val_cur
            best_h = h_cur.copy()
    
    # Final coordinate descent polish with adaptive step size
    if time.time() < deadline - 5.0:
        h_cur = best_h.copy()
        np.random.seed(999)
        step_size = 0.05
        
        for step in range(5000):
            if time.time() > deadline - 5.0:
                break
            
            idx = np.random.randint(N)
            delta = np.random.uniform(-step_size, step_size)
            h_new = h_cur.copy()
            h_new[idx] += delta
            h_new = project_sum(h_new)
            
            val_new = compute_value(h_new)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                h_cur = h_new
                step_size = max(0.005, step_size * 0.999)
            else:
                h_new2 = h_cur.copy()
                h_new2[idx] -= delta
                h_new2 = project_sum(h_new2)
                val_new2 = compute_value(h_new2)
                if val_new2 < best_val:
                    best_val = val_new2
                    best_h = h_new2.copy()
                    h_cur = h_new2
                    step_size = max(0.005, step_size * 0.999)
                else:
                    step_size *= 0.95
    
    # Final high-precision polish: tiny perturbations around best solution
    if time.time() < deadline - 3.0:
        h_cur = best_h.copy()
        np.random.seed(777)
        for _ in range(3000):
            if time.time() > deadline - 3.0:
                break
            idx = np.random.randint(N)
            delta = np.random.uniform(-0.002, 0.002)
            h_new = h_cur.copy()
            h_new[idx] += delta
            h_new = project_sum(h_new)
            val_new = compute_value(h_new)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                h_cur = h_new
    
    # Final verification and cleanup
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = project_sum(best_h)
    
    return best_h
