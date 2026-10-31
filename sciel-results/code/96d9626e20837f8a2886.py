"""Erdos minimum overlap - optimize step heights to minimize max overlap."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def compute_overlap_max(h, n_eval=200):
    """Compute the maximum overlap of h with shifted 1-h."""
    # h is defined on [0, 2] with N steps, each of width 2/N
    # h[i] is the value on interval [2*i/N, 2*(i+1)/N)
    step_w = 2.0 / N
    # Evaluate overlap at many shift values
    shifts = np.linspace(-2, 2, n_eval)
    max_ov = 0.0
    for s in shifts:
        # overlap = integral of h(x) * (1 - h(x+s)) dx over x in [0,2] where x+s in [0,2]
        # For step functions, compute piecewise
        # x in [0,2], x+s in [0,2] means x in [max(0,-s), min(2, 2-s)]
        x_lo = max(0.0, -s)
        x_hi = min(2.0, 2.0 - s)
        if x_lo >= x_hi:
            continue
        # Integrate h(x)*(1-h(x+s)) over [x_lo, x_hi]
        # Sample at midpoints of sub-intervals aligned to step boundaries
        # For each step i, x in [2*i/N, 2*(i+1)/N)
        # h(x) = h[i], h(x+s) depends on where x+s falls
        total = 0.0
        for i in range(N):
            x0 = 2.0 * i / N
            x1 = 2.0 * (i + 1) / N
            # Intersection with [x_lo, x_hi]
            a = max(x0, x_lo)
            b = min(x1, x_hi)
            if a >= b:
                continue
            # h(x) = h[i] for x in [a,b]
            # h(x+s): for x in [a,b], x+s in [a+s, b+s]
            # This may span multiple steps of h
            # Approximate: use the value at the midpoint
            x_mid = (a + b) / 2.0
            xs_mid = x_mid + s
            if xs_mid < 0 or xs_mid >= 2.0:
                continue
            # Find which step xs_mid falls in
            j = int(xs_mid / step_w)
            if j >= N:
                j = N - 1
            if j < 0:
                j = 0
            total += h[i] * (1.0 - h[j]) * (b - a)
        ov = total
        if ov > max_ov:
            max_ov = ov
    return max_ov


def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s limit
    
    # Initial guess: constant 0.5
    h0 = np.full(N, 0.5)
    
    # Try a few different initializations and optimization strategies
    best_h = h0.copy()
    best_score = compute_overlap_max(h0)
    
    # Use scipy minimize with SLSQP or L-BFGS-B
    # Objective: maximize overlap -> minimize negative
    def objective(h_flat):
        h = np.clip(h_flat, 0, 1)
        # Enforce sum constraint: sum(h) * step_w = 1 => sum(h) = N/2
        # We'll handle this with a penalty or reparameterization
        # Reparameterize: h = sigmoid(x) * 2 to get range (0,2), then clip to [0,1]
        # Actually, let's use a different approach: optimize with constraint
        return compute_overlap_max(h)
    
    # Constraint: sum(h) = N/2
    # Use SLSQP with equality constraint
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
    bounds = [(0.0, 1.0)] * N
    
    # Try multiple random initializations
    np.random.seed(42)
    attempts = 0
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        if attempt == 0:
            h_init = h0.copy()
        else:
            # Random initialization satisfying sum constraint
            h_init = np.random.dirichlet(np.ones(N)) * N / 2.0
            h_init = np.clip(h_init, 0, 1)
            # Adjust to satisfy sum constraint
            diff = N/2.0 - np.sum(h_init)
            h_init += diff / N
            h_init = np.clip(h_init, 0, 1)
            # Final adjustment
            diff = N/2.0 - np.sum(h_init)
            mask = h_init > 0.01 if diff > 0 else h_init < 0.99
            if np.any(mask):
                h_init[mask] += diff / np.sum(mask)
                h_init = np.clip(h_init, 0, 1)
        
        try:
            result = minimize(
                objective, h_init, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            attempts += 1
        except Exception:
            continue
        
        h_opt = np.clip(result.x, 0, 1)
        # Ensure sum constraint
        diff = N/2.0 - np.sum(h_opt)
        h_opt += diff / N
        h_opt = np.clip(h_opt, 0, 1)
        
        score = compute_overlap_max(h_opt)
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Final: ensure exact sum
    diff = N/2.0 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
