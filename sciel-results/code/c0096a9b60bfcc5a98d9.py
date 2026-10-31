import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _score(h):
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _smooth_max_grad(x, alpha=100.0):
    x_max = np.max(x)
    e = np.exp(alpha * (x - x_max))
    p = e / np.sum(e)
    return p

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    h_best = np.full(N, 0.5)
    score_best = _score(h_best)

    def objective(params):
        h = params
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        return _smooth_max(overlap, alpha=200.0)

    def objective_grad(params):
        h = params
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        p = _smooth_max_grad(overlap, alpha=200.0)
        grad = np.zeros(N)
        # overlap[i] = h[i] * (1 - h[i+1])
        # d overlap[i] / d h[i] = 1 - h[i+1]
        # d overlap[i] / d h[i+1] = -h[i]
        # grad[j] = sum_i p[i] * d overlap[i] / d h[j]
        # d overlap[i] / d h[i] contributes to grad[i]
        # d overlap[i-1] / d h[i] contributes to grad[i]
        # grad[i] = p[i] * (1 - h[i+1]) + p[i-1] * (-h[i-1])
        # Note: h[N] is 0.
        # For i=0: grad[0] = p[0] * (1 - h[1])
        # For i=N-1: grad[N-1] = p[N-1] * (1 - 0) + p[N-2] * (-h[N-2])
        
        # Vectorized:
        # term1 = p * (1 - h[1:]) for indices 0..N-2?
        # Let's do it carefully.
        
        # grad[i] gets contribution from overlap[i] (via h[i]) and overlap[i-1] (via h[i+1] in overlap[i-1]? No, overlap[i-1] depends on h[i-1] and h[i].)
        # overlap[i] = h[i](1-h[i+1]).
        # d/dh[i] overlap[i] = 1 - h[i+1].
        # d/dh[i] overlap[i-1] = d/dh[i] (h[i-1](1-h[i])) = -h[i-1].
        
        # So grad[i] = p[i] * (1 - h[i+1]) - p[i-1] * h[i-1]
        # with h[N]=0 and p[-1]=0.
        
        h_next = np.roll(h, -1)
        h_next[-1] = 0.0
        h_prev = np.roll(h, 1)
        h_prev[0] = 0.0
        p_prev = np.roll(p, 1)
        p_prev[0] = 0.0
        
        grad = p * (1.0 - h_next) - p_prev * h_prev
        return grad

    def constraint_sum(params):
        return np.sum(params) - N/2.0

    def constraint_sum_grad(params):
        return np.ones(N)

    bounds = [(0.0, 1.0)] * N

    initializations = [
        np.full(N, 0.5),
        0.5 + 0.1 * np.sin(2 * np.pi * np.arange(N) / N),
        0.5 + 0.1 * np.cos(2 * np.pi * np.arange(N) / N),
        np.random.rand(N)
    ]
    
    for h0 in initializations:
        if time.time() > deadline - 10:
            break
        h0 = np.clip(h0, 0.0, 1.0)
        # Adjust sum
        diff = N/2.0 - np.sum(h0)
        h0 = h0 + diff/N
        h0 = np.clip(h0, 0.0, 1.0)
        
        try:
            res = minimize(
                objective,
                h0,
                jac=objective_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad}],
                options={'maxiter': 1000, 'ftol': 1e-16}
            )
            if res.success:
                sc = _score(res.x)
                if sc < score_best:
                    score_best = sc
                    h_best = res.x.copy()
        except Exception:
            pass

    return h_best
