import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    deadline = time.time() + 140
    
    def compute_overlaps(h):
        h = np.asarray(h, dtype=float)
        H = np.fft.fft(h)
        autocorr = np.fft.ifft(H * np.conj(H)).real
        overlap = N / 2.0 - autocorr
        return overlap
    
    def objective_and_grad(h_flat):
        h = h_flat
        ov = compute_overlaps(h)
        beta = 100.0
        m = np.max(ov)
        if not np.isfinite(m) or m > 1e10:
            return m, None
        exp_terms = np.exp(beta * (ov - m))
        Z = np.sum(exp_terms)
        softmax = exp_terms / Z
        obj = (m + np.log(Z)) / beta
        # Gradient via automatic differentiation is complex; use None
        return obj, None
    
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_max_overlap = np.max(compute_overlaps(h0))
    
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0, 'jac': lambda h: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N
    
    start = time.time()
    iterations = 0
    while time.time() < deadline - 10:
        res = minimize(objective_and_grad, h0, method='SLSQP', bounds=bounds, 
                      constraints=constraints, options={'maxiter': 300, 'ftol': 1e-10})
        iterations += 1
        
        h_new = res.x
        h_new = np.clip(h_new, 0, 1)
        # Fix sum constraint
        diff = N/2.0 - np.sum(h_new)
        h_new += diff / N
        h_new = np.clip(h_new, 0, 1)
        diff = N/2.0 - np.sum(h_new)
        h_new += diff / N
        h_new = np.clip(h_new, 0, 1)
        
        ov = compute_overlaps(h_new)
        max_ov = np.max(ov)
        
        if max_ov < best_max_overlap - 1e-10:
            best_max_overlap = max_ov
            best_h = h_new.copy()
            h0 = best_h.copy()
        else:
            # Try perturbation
            if time.time() < deadline - 30:
                h0 = best_h + 0.1 * np.random.randn(N)
                h0 = np.clip(h0, 0, 1)
                diff = N/2.0 - np.sum(h0)
                h0 += diff / N
                h0 = np.clip(h0, 0, 1)
            else:
                break
        
        if time.time() > deadline - 15:
            break
    
    return best_h
