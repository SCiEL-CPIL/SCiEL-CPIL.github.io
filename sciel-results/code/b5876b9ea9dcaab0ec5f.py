"""Erdos minimum overlap: minimize max(correlate(h, 1-h, full))/N*2 using Linear Programming."""
import numpy as np
import time
from scipy.optimize import linprog

N = 100
DEADLINE = 135.0  # safety margin below 150s kill
_START = time.time()


def remaining_time():
    return DEADLINE - (time.time() - _START)


def objective(h):
    """Compute the raw value (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def project(h):
    """Project h to satisfy: h in [0,1]^N, sum(h) ~ N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(100):
        s = h.sum()
        if abs(s - target) < 1e-12:
            break
        # Adjust by uniform shift, then re-clip
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    """
    Solve the problem as a Linear Program.
    
    We want to minimize t such that:
    1. 0 <= h_i <= 1 for all i
    2. sum(h_i) = N/2
    3. For all lags k, correlate(h, 1-h)[k] <= t * N / 2
    
    Note: The evaluator computes value = max(correlate(h, 1-h, full)) / N * 2.
    Let M = max(correlate(h, 1-h, full)). We want to minimize M/N * 2.
    Let t = M. We minimize t.
    Constraints:
    For each lag k in range(-N+1, N):
    sum_{i} h_{i+k} (1 - h_i) <= t  (with appropriate index bounds)
    
    The correlation is defined as:
    (h * (1-h))[k] = sum_{i} h[i] * (1-h)[i+k]
    Note: np.correlate(a, v, mode='full') computes sum a[i] v[i+k].
    Here a=h, v=1-h.
    So term[k] = sum_{i} h[i] (1 - h[i+k]).
    
    Wait, np.correlate(h, 1-h) means a=h, v=1-h.
    c[k] = sum_{i} h[i] * (1-h)[i+k]
    Let j = i+k. i = j-k.
    c[k] = sum_{j} h[j-k] (1-h[j])
    
    This is a quadratic form in h.
    c[k] = sum_{j} h[j-k] - sum_{j} h[j-k]h[j]
    
    This is NOT linear in h. The term h[j-k]h[j] is quadratic.
    Therefore, this is NOT a Linear Program. It is a Quadratic Program (QP).
    Specifically, minimizing the maximum of quadratic forms.
    
    However, we can re-formulate.
    Let's check the structure again.
    value = max_k c[k] / N * 2.
    
    If we cannot solve QP easily with standard scipy (which has minimize for general, but linprog is LP),
    we might need to use `minimize` with a specific method or reformulate.
    
    But wait, `scipy.optimize.minimize` with method='SLSQP' or 'trust-constr' can handle QP-like problems if we define the constraints properly.
    However, the "max of quadratics" is non-smooth.
    
    Alternative: Bisection on the value.
    For a given target value V, check if there exists h such that:
    1. 0 <= h <= 1
    2. sum(h) = N/2
    3. c[k] <= V * N / 2 for all k
    
    The constraints c[k] <= C are quadratic inequalities.
    This is a Quadratically Constrained Quadratic Program (QCQP) feasibility problem.
    
    Is there a simpler structural insight?
    The current program uses L-BFGS-B on the projected space.
    Inspiration 1 tried LinProg but scored lower (1.13 vs 1.28). Why?
    Perhaps the LinProg implementation in inspiration 1 was flawed or the relaxation was too loose, or it actually solved a different problem.
    Wait, if the problem is QP, LinProg is not applicable directly unless linearized.
    
    Let's look at the "Different Strategy" directive.
    Current: L-BFGS-B (Gradient based, smooth approximation of max via projection).
    
    New Strategy: **Coordinate Descent with Exact Local Optimization** or **Simulated Annealing**.
    Given the non-smoothness of `max(correlate)`, gradient methods can get stuck.
    Simulated Annealing (SA) or Random Restarts with Local Search (Hill Climbing) can escape local minima.
    
    Let's try **Simulated Annealing**.
    State: h (vector of 100 floats).
    Move: Pick a random index i, perturb h[i] by delta.
    Constraint Handling: Project to feasible set after each move.
    Energy: objective(h).
    
    This is a fundamentally different search paradigm (stochastic global search vs deterministic local gradient descent).
    """
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # Quick deterministic checks to ensure we don't start worse than trivial
    # 1. Cosine
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2 * np.pi * i / N)
    h_cos = project(h_cos)
    v_cos = objective(h_cos)
    if v_cos < best_val:
        best_val = v_cos
        best_h = h_cos.copy()
        
    # 2. Random restarts with Hill Climbing
    # This is a structural change: using stochastic local search with multiple restarts.
    
    rng = np.random.default_rng(12345)
    
    def local_search(h_init, max_iter=2000, step_size=0.1):
        """
        Perform coordinate ascent/descent with random restarts within the local search.
        """
        h = project(h_init)
        current_val = objective(h)
        
        # Adaptive step size
        curr_step = step_size
        
        for it in range(max_iter):
            if remaining_time() < 5.0:
                break
                
            # Pick a random coordinate
            idx = rng.integers(0, N)
            
            # Try to improve by moving in + or - direction
            # We will try a few random perturbations
            improved = False
            
            # Generate a few candidate moves
            for _ in range(10):
                delta = rng.uniform(-curr_step, curr_step)
                h_new = h.copy()
                h_new[idx] += delta
                h_new = project(h_new)
                
                # Ensure the move actually changed something significantly
                if np.allclose(h_new, h, atol=1e-9):
                    continue
                    
                new_val = objective(h_new)
                
                if new_val < current_val:
                    h = h_new
                    current_val = new_val
                    improved = True
                    # Decrease step size slightly if we found an improvement
                    curr_step *= 0.95
                    break
            else:
                # No improvement found in this batch, reduce step size
                curr_step *= 0.9
                if curr_step < 1e-4:
                    break
                    
        return h, current_val

    # Run multiple restarts
    num_restarts = 20
    for r in range(num_restarts):
        if remaining_time() < 10.0:
            break
            
        # Initialize with a random feasible point
        h_rand = rng.uniform(0, 1, N)
        h_rand = project(h_rand)
        
        h_opt, val_opt = local_search(h_rand)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
            
        # Also try refining the global best with a finer local search
        if r % 5 == 0 and remaining_time() > 15.0:
            h_ref, val_ref = local_search(best_h, max_iter=5000, step_size=0.01)
            if val_ref < best_val:
                best_val = val_ref
                best_h = h_ref.copy()

    # Final sanity check
    best_h = project(best_h)
    assert np.all(np.isfinite(best_h)), "Non-finite values in result"
    assert np.all(best_h >= -1e-10) and np.all(best_h <= 1.0 + 1e-10), "Out of bounds"
    assert abs(best_h.sum() - N / 2.0) < 1e-9 * N, "Sum constraint violated"
    
    return best_h


# EVOLVE-BLOCK-START
# (construct is already defined above)
# EVOLVE-BLOCK-END
