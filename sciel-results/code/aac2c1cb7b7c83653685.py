"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _normalize(h):
    """Enforce exact sum N/2 and bounds [0,1]."""
    h = np.array(h, dtype=float)
    for _ in range(20):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final exact correction
    s = np.sum(h)
    diff = N/2.0 - s
    if abs(diff) > 1e-15:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _get_basis(K):
    """
    Precompute the basis matrix for Fourier modes 1..K.
    """
    i = np.arange(N)
    B = np.zeros((N, 2*K))
    for k in range(1, K + 1):
        B[:, 2*(k-1)] = np.cos(2 * np.pi * k * i / N)
        B[:, 2*(k-1)+1] = np.sin(2 * np.pi * k * i / N)
    return B

def _project_to_coefficients(h, B):
    """
    Project h onto the column space of B (Fourier subspace).
    """
    c = np.zeros(B.shape[1])
    for j in range(B.shape[1]):
        c[j] = (2.0 / N) * np.dot(h, B[:, j])
    return c

def _pgd_fourier(h_init, K, max_iter=500, alpha=100.0):
    """
    Projected Gradient Descent on Fourier coefficients.
    """
    B = _get_basis(K)
    c = _project_to_coefficients(h_init, B)
    BT = B.T
    
    best_h = h_init.copy()
    best_score = _score(h_init)
    
    step_size = 0.01
    min_step_size = 1e-5
    
    for it in range(max_iter):
        h = 0.5 * np.ones(N) + B @ c
        
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        
        current_score = np.max(overlap)
        if current_score < best_score:
            best_score = current_score
            best_h = _normalize(h)
        
        # Gradient of smooth max w.r.t h
        exp_term = np.exp(alpha * (overlap - np.max(overlap)))
        weights = exp_term / np.sum(exp_term)
        
        grad_h = np.zeros(N)
        h_shifted_for_grad = np.zeros(N)
        h_shifted_for_grad[:-1] = h[1:]
        d_overlap_d_h_self = 1.0 - h_shifted_for_grad
        grad_h += weights * d_overlap_d_h_self
        
        if N > 1:
            grad_h[1:] += -weights[:-1] * h[:-1]
            
        grad_c = BT @ grad_h
        c_new = c - step_size * grad_c
        h_new = 0.5 * np.ones(N) + B @ c_new
        h_proj = _normalize(h_new)
        c = _project_to_coefficients(h_proj, B)
        
        score_proj = _score(h_proj)
        if score_proj < current_score:
            step_size = min(step_size * 1.1, 0.1)
        else:
            step_size *= 0.8
            if step_size < min_step_size:
                break
                
    return best_h, best_score

def _epigraph_polish(h_init, maxiter=1000, time_limit=30.0):
    """
    Refine h using SLSQP with an epigraph variable t.
    Minimize t + penalty * sum(max(0, overlap_i - t)^2)
    Subject to sum(h)=N/2, 0<=h<=1, t>=0.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return float('inf')
        
        h = params[:N]
        t = params[N]
        
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        
        # Penalty for violating t >= overlap_i
        # Using a smooth approximation or direct penalty
        # Direct penalty on (overlap - t) where overlap > t
        violations = np.maximum(0.0, overlap - t)
        penalty = 1000.0 * np.sum(violations**2)
        
        return t + penalty

    def constraint_sum(params):
        return np.sum(params[:N]) - N/2.0

    # We don't strictly need to enforce overlap <= t as a constraint if the penalty is strong enough,
    # but adding it as an inequality constraint helps SLSQP.
    # However, SLSQP handles many constraints slowly. 
    # Let's rely on the penalty for the epigraph and just keep sum/bounds.
    # If we add N inequality constraints, it might be slow.
    # Let's try just the penalty first. If it fails, we can add a few critical ones.
    
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)]
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    
    x0 = np.concatenate([h_init, [_score(h_init) + 0.01]])
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': False}
        )
        
        if result.fun < 1.0: # Sanity check
            h_opt = result.x[:N].copy()
            h_opt = _normalize(h_opt)
            return h_opt, _score(h_opt)
    except Exception:
        pass
        
    return None, float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 135.0 # Safety margin

    # 1. Generate diverse initial guesses
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    
    candidates = [h0]
    
    np.random.seed(42)
    for _ in range(5):
        # Random smooth candidates
        h_rand = 0.5 + 0.2 * np.sin(np.random.uniform(0, 2*np.pi) + np.arange(N)*0.1)
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = _normalize(h_rand)
        candidates.append(h_rand)
        
        # Random blocky candidates
        h_block = np.random.rand(N)
        h_block = np.round(h_block) # Binarize
        h_block = h_block * 0.5 + 0.25 # Scale to [0.25, 0.75]
        h_block = _normalize(h_block)
        candidates.append(h_block)

    best_h = h0.copy()
    best_score = _score(h0)

    # 2. PGD Phase: Refine candidates in Fourier space
    # This is fast and helps escape local minima
    for h_cand in candidates:
        if time.time() > deadline - 40.0:
            break
        try:
            # Try a few different K values
            for K in [5, 10]:
                if time.time() > deadline - 40.0:
                    break
                h_pgd, s_pgd = _pgd_fourier(h_cand, K, max_iter=200, alpha=50.0)
                if s_pgd < best_score:
                    best_h = h_pgd
                    best_score = s_pgd
        except Exception:
            pass

    # 3. Epigraph SLSQP Phase: High precision local polish
    # This is the structural change from Inspiration 1, combined with current's best candidate
    if time.time() < deadline - 5.0:
        try:
            h_polished, s_polished = _epigraph_polish(best_h, maxiter=500, time_limit=deadline - time.time() - 2.0)
            if s_polished < best_score:
                best_h = h_polished
                best_score = s_polished
        except Exception:
            pass

    # 4. If time remains, try polishing other good candidates
    if time.time() < deadline - 5.0:
        # Sort candidates by score after PGD? We didn't store them all.
        # Let's just try a few random restarts directly with Epigraph if time permits
        np.random.seed(43)
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            h_rand = 0.5 + 0.1 * np.random.randn(N)
            h_rand = np.clip(h_rand, 0, 1)
            h_rand = _normalize(h_rand)
            
            # Quick PGD
            h_pgd, s_pgd = _pgd_fourier(h_rand, 5, max_iter=50, alpha=50.0)
            
            # Epigraph Polish
            h_polished, s_polished = _epigraph_polish(h_pgd, maxiter=200, time_limit=5.0)
            
            if s_polished < best_score:
                best_h = h_polished
                best_score = s_polished

    return best_h
