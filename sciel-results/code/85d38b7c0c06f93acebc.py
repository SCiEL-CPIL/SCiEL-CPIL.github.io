import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 600

def compute_score(h):
    s = np.sum(h)
    if s == 0:
        return 0.0
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n=n_fft)
    conv = np.real(np.fft.ifft(H * H))[:2*N-1]
    max_conv = np.max(conv)
    return max_conv / (s * s)

def smooth_max_surrogate(conv, beta=100.0):
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta

def objective(params, beta=100.0):
    h = params
    s = np.sum(h)
    if s < 1e-10:
        return 1e10
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n=n_fft)
    conv = np.real(np.fft.ifft(H * H))[:2*N-1]
    smooth_max = smooth_max_surrogate(conv, beta)
    return smooth_max / (s * s)

def objective_grad(params, beta=100.0):
    h = params
    s = np.sum(h)
    if s < 1e-10:
        return np.zeros_like(h)
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n=n_fft)
    conv = np.real(np.fft.ifft(H * H))[:2*N-1]
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    W = np.fft.fft(weights, n=n_fft)
    grad_h = 2.0 * np.real(np.fft.ifft(W * np.conj(H)))[:N]
    smooth_max_val = max_val + np.log(sum_exp) / beta
    grad_obj = grad_h / (s * s) - 2 * smooth_max_val / (s * s * s)
    return grad_obj

def super_ellipse_h(center, width, exponent, asymmetry):
    x = np.linspace(-0.25, 0.25, N)
    xc = center * 0.5
    w = width * 0.5
    if w < 1e-10:
        return np.zeros(N)
    t = (x - xc) / w
    t_asym = t * (1.0 + asymmetry * t)
    val = np.power(np.abs(t_asym), exponent)
    h = np.where(val <= 1.0, 1.0, 0.0)
    return h

def coordinate_descent(h, deadline, max_rounds=50, step_size=0.1):
    best_score = compute_score(h)
    best_h = h.copy()
    
    for round_idx in range(max_rounds):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 3:
                break
            
            h_old = h[i]
            for ds in [step_size, -step_size, step_size*0.5, -step_size*0.5]:
                h_new = h.copy()
                h_new[i] = max(0, h_old + ds)
                if np.all(h_new == h):
                    continue
                score = compute_score(h_new)
                if score < best_score - 1e-12:
                    h = h_new
                    best_score = score
                    best_h = h.copy()
                    improved = True
                    break
        
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)
    
    np.random.seed(42)
    
    # Phase 1: Global search over super-ellipse parameters
    param_grids = {
        'center': np.linspace(-0.5, 0.5, 11),
        'width': np.linspace(0.2, 1.0, 9),
        'exponent': np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]),
        'asymmetry': np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    }
    
    for c in param_grids['center']:
        if time.time() > deadline - 60:
            break
        for w in param_grids['width']:
            if time.time() > deadline - 60:
                break
            for e in param_grids['exponent']:
                if time.time() > deadline - 60:
                    break
                for a in param_grids['asymmetry']:
                    if time.time() > deadline - 60:
                        break
                    h = super_ellipse_h(c, w, e, a)
                    if np.sum(h) < 10:
                        continue
                    score = compute_score(h)
                    if score < best_score:
                        best_score = score
                        best_h = h.copy()
    
    # Phase 2: Local refinement with coordinate descent
    if time.time() < deadline - 40:
        best_h, best_score = coordinate_descent(best_h, deadline, max_rounds=30, step_size=0.2)
    
    # Phase 3: Random perturbations and refine
    for attempt in range(5):
        if time.time() > deadline - 20:
            break
        h_pert = best_h.copy()
        noise = np.random.normal(0, 0.1, N)
        h_pert = np.maximum(h_pert + noise, 0)
        h_ref, score_ref = coordinate_descent(h_pert, deadline, max_rounds=20, step_size=0.1)
        if score_ref < best_score:
            best_score = score_ref
            best_h = h_ref.copy()
    
    # Phase 4: Final L-BFGS-B polish
    if time.time() < deadline - 10:
        try:
            def obj_full(h):
                s = np.sum(h)
                if s < 1e-10:
                    return 1e10
                n_fft = 2 * N - 1
                H = np.fft.fft(h, n=n_fft)
                conv = np.real(np.fft.ifft(H * H))[:2*N-1]
                smooth_max = smooth_max_surrogate(conv, 200.0)
                return smooth_max / (s * s)
            
            def grad_full(h):
                s = np.sum(h)
                if s < 1e-10:
                    return np.zeros_like(h)
                n_fft = 2 * N - 1
                H = np.fft.fft(h, n=n_fft)
                conv = np.real(np.fft.ifft(H * H))[:2*N-1]
                max_val = np.max(conv)
                shifted = 200.0 * (conv - max_val)
                shifted = np.clip(shifted, -100, 100)
                exp_shifted = np.exp(shifted)
                sum_exp = np.sum(exp_shifted)
                weights = exp_shifted / sum_exp
                W = np.fft.fft(weights, n=n_fft)
                grad_h = 2.0 * np.real(np.fft.ifft(W * np.conj(H)))[:N]
                smooth_max_val = max_val + np.log(sum_exp) / 200.0
                grad_obj = grad_h / (s * s) - 2 * smooth_max_val / (s * s * s)
                return grad_obj
            
            res = minimize(
                obj_full,
                best_h,
                jac=grad_full,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt = np.maximum(res.x, 0)
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 5: Final coordinate descent polish
    if time.time() < deadline - 3:
        best_h, best_score = coordinate_descent(best_h, deadline, max_rounds=10, step_size=0.05)
    
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
