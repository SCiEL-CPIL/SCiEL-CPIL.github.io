"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def construct():
    """
    Construct 100 step heights h in [0,1] with sum(h)=50,
    minimizing max(np.correlate(h, 1-h, mode='full')) / N * 2.
    
    Strategy:
    - Model h as a sum of a few sinusoidal components with incommensurate frequencies
    - Use differential evolution to optimize amplitudes and phases
    - Polish with high-precision coordinate descent
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin
    
    def compute_value(h):
        """Compute the evaluator's value."""
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return np.inf
        h_clipped = np.clip(h, 0.0, 1.0)
        if abs(np.sum(h_clipped) - 50.0) > 1e-7 * 100:
            return np.inf
        corr = np.correlate(h_clipped, 1 - h_clipped, mode="full")
        return np.max(corr) / N * 2
    
    def project_to_feasible(h):
        """Project h to [0,1] and sum=50 using iterative bisection."""
        h = np.asarray(h, dtype=float).copy()
        target = 50.0
        
        # First clip to [0,1]
        h = np.clip(h, 0.0, 1.0)
        
        # Use bisection to find shift that makes sum = target
        lo = -1.0 - np.max(h)
        hi = 1.0 - np.min(h) + 1.0
        
        for _ in range(80):
            mid = (lo + hi) / 2.0
            h_shifted = np.clip(h + mid, 0.0, 1.0)
            s = np.sum(h_shifted)
            if s > target:
                hi = mid
            else:
                lo = mid
        
        h_proj = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
        
        # Fine-tune: adjust individual elements to hit sum exactly
        diff = target - np.sum(h_proj)
        if abs(diff) > 1e-12:
            if diff > 0:
                # Need to increase sum: adjust elements with room to increase
                idx = np.argsort(h_proj)[::-1]  # Start with elements closest to 1
                for i in idx:
                    room = 1.0 - h_proj[i]
                    if room <= 0:
                        continue
                    adj = min(diff, room)
                    h_proj[i] += adj
                    diff -= adj
                    if abs(diff) < 1e-14:
                        break
            else:
                # Need to decrease sum: adjust elements with room to decrease
                idx = np.argsort(h_proj)  # Start with elements closest to 0
                for i in idx:
                    room = h_proj[i]
                    if room <= 0:
                        continue
                    adj = max(diff, -room)
                    h_proj[i] += adj
                    diff -= adj
                    if abs(diff) < 1e-14:
                        break
        
        return h_proj
    
    def evaluate_params(params):
        """Evaluate a set of sinusoidal parameters."""
        # params: [A1, f1, p1, A2, f2, p2, A3, f3, p3, A4, f4, p4]
        A1, f1, p1 = params[0], params[1], params[2]
        A2, f2, p2 = params[3], params[4], params[5]
        A3, f3, p3 = params[6], params[7], params[8]
        A4, f4, p4 = params[9], params[10], params[11]
        
        i = np.arange(N, dtype=float)
        h = 0.5 + A1 * np.sin(2*np.pi*f1*i + p1) \
                    + A2 * np.sin(2*np.pi*f2*i + p2) \
                    + A3 * np.sin(2*np.pi*f3*i + p3) \
                    + A4 * np.sin(2*np.pi*f4*i + p4)
        
        h = project_to_feasible(h)
        return compute_value(h)
    
    # Differential evolution to optimize sinusoidal parameters
    # Bounds: amplitudes in [0, 0.3], frequencies in [0.01, 5.0], phases in [0, 2*pi]
    bounds = [(0.0, 0.3), (0.01, 5.0), (0.0, 2*np.pi)] * 4
    
    best_h = None
    best_val = np.inf
    
    # Run differential evolution with time budget
    if time.time() < deadline - 60.0:
        try:
            result = differential_evolution(
                evaluate_params,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False,
                callback=lambda x, convergence: None
            )
            
            if result.fun < best_val:
                best_val = result.fun
                # Reconstruct best_h from result.x
                A1, f1, p1 = result.x[0], result.x[1], result.x[2]
                A2, f2, p2 = result.x[3], result.x[4], result.x[5]
                A3, f3, p3 = result.x[6], result.x[7], result.x[8]
                A4, f4, p4 = result.x[9], result.x[10], result.x[11]
                
                i = np.arange(N, dtype=float)
                h = 0.5 + A1 * np.sin(2*np.pi*f1*i + p1) \
                        + A2 * np.sin(2*np.pi*f2*i + p2) \
                        + A3 * np.sin(2*np.pi*f3*i + p3) \
                        + A4 * np.sin(2*np.pi*f4*i + p4)
                best_h = project_to_feasible(h)
        except Exception:
            pass
    
    # If DE didn't find anything good, try a grid search as fallback
    if best_h is None or best_val > 1.2:
        # Quick grid search over single sinusoid
        for f in np.linspace(0.05, 3.0, 30):
            for A in np.linspace(0.1, 0.4, 15):
                for phase in np.linspace(0.0, 2*np.pi, 8):
                    i = np.arange(N, dtype=float)
                    h = 0.5 + A * np.sin(2*np.pi*f*i + phase)
                    h = project_to_feasible(h)
                    val = compute_value(h)
                    if val < best_val:
                        best_val = val
                        best_h = h.copy()
                    
                    if time.time() > deadline - 40.0:
                        break
                if time.time() > deadline - 40.0:
                    break
            if time.time() > deadline - 40.0:
                break
    
    # High-precision coordinate descent polish
    if best_h is not None and time.time() < deadline - 10.0:
        h = best_h.copy()
        current_val = compute_value(h)
        
        # Multiple passes with decreasing step sizes
        step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        
        for step in step_sizes:
            if time.time() > deadline - 10.0:
                break
            
            improved = True
            max_iters = 20
            for cd_iter in range(max_iters):
                if time.time() > deadline - 10.0:
                    break
                improved = False
                
                # Randomize order to avoid bias
                indices = np.random.permutation(N)
                
                for i in indices:
                    if time.time() > deadline - 10.0:
                        break
                    
                    h_new = h.copy()
                    old_val = h_new[i]
                    
                    # Try increasing
                    h_new[i] = min(1.0, old_val + step)
                    # Project to maintain sum
                    h_new = project_to_feasible(h_new)
                    val = compute_value(h_new)
                    if val < current_val - 1e-12:
                        current_val = val
                        h = h_new.copy()
                        improved = True
                        continue
                    
                    # Try decreasing
                    h_new = h.copy()
                    h_new[i] = max(0.0, old_val - step)
                    h_new = project_to_feasible(h_new)
                    val = compute_value(h_new)
                    if val < current_val - 1e-12:
                        current_val = val
                        h = h_new.copy()
                        improved = True
                        continue
                    
                    # Try both directions with smaller steps
                    for factor in [0.5, 0.25]:
                        s = step * factor
                        h_new = h.copy()
                        h_new[i] = min(1.0, old_val + s)
                        h_new = project_to_feasible(h_new)
                        val = compute_value(h_new)
                        if val < current_val - 1e-12:
                            current_val = val
                            h = h_new.copy()
                            improved = True
                            break
                        
                        h_new = h.copy()
                        h_new[i] = max(0.0, old_val - s)
                        h_new = project_to_feasible(h_new)
                        val = compute_value(h_new)
                        if val < current_val - 1e-12:
                            current_val = val
                            h = h_new.copy()
                            improved = True
                            break
                    
                    if improved:
                        break
                
                if not improved:
                    break
        
        best_h = h
        best_val = current_val
    
    # Final verification and correction
    if best_h is not None:
        best_h = project_to_feasible(best_h)
        final_val = compute_value(best_h)
        if final_val < best_val:
            best_val = final_val
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END
