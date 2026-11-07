"""Erdos minimum overlap: Linear Programming (LP) approach via epigraph reformulation."""
import time
import numpy as np
from scipy.optimize import linprog

N = 100
DEADLINE = 145.0  # Safety margin

def _score(h):
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _is_valid(h):
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True

def _fix_sum(h):
    """
    Adjust h to sum to N/2 while keeping values in [0, 1].
    This is a fallback for numerical errors in LP solvers.
    """
    target = N / 2.0
    h = h.copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(50):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-9
            if not np.any(mask):
                break
            room = 1.0 - h[mask]
            total_room = np.sum(room)
            if total_room < 1e-9:
                break
            add_factor = min(1.0, diff / total_room)
            h[mask] += room * add_factor
        else:
            mask = h > 1e-9
            if not np.any(mask):
                break
            val = h[mask]
            total_val = np.sum(val)
            if total_val < 1e-9:
                break
            sub_factor = min(1.0, -diff / total_val)
            h[mask] -= val * sub_factor
    
    # Final strict fix if needed
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.argmin(h)
            if h[idx] < 1.0:
                h[idx] += diff
        else:
            idx = np.argmax(h)
            if h[idx] > 0.0:
                h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    
    def deadline_check():
        return time.time() > start_time + DEADLINE
    
    # Initialize with seed
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # The objective is: min max_k corr_k
    # Let t be the upper bound on corr_k.
    # Minimize t subject to:
    # 1. sum(h) = N/2
    # 2. 0 <= h_i <= 1
    # 3. corr_k(h, 1-h) <= t for all k
    
    # Variables: x = [h_0, ..., h_{N-1}, t]
    # Total variables: N + 1
    
    # Objective: minimize t -> c = [0, ..., 0, 1]
    c = np.zeros(N + 1)
    c[-1] = 1.0
    
    # Equality constraints: sum(h) = N/2
    A_eq = np.zeros((1, N + 1))
    A_eq[0, :N] = 1.0
    b_eq = np.array([N / 2.0])
    
    # Inequality constraints:
    # 1. Box constraints on h: 0 <= h_i <= 1
    #    This is handled by bounds in linprog.
    # 2. Correlation constraints: corr_k <= t
    #    corr_k = sum_{i} h_{i+k} (1 - h_i)  (indices within range)
    #    corr_k = sum_{i} h_{i+k} - sum_{i} h_{i+k} h_i
    #    Wait, the correlation is linear in h?
    #    np.correlate(h, 1-h)
    #    Let g = 1-h.
    #    corr_k = sum_{i} h_{i+k} g_i = sum_{i} h_{i+k} (1 - h_i)
    #    corr_k = sum_{i} h_{i+k} - sum_{i} h_{i+k} h_i
    #    The term sum h_{i+k} h_i is quadratic (bilinear).
    #    So the standard LP formulation is NOT directly applicable because the objective/constraints are non-linear (quadratic).
    
    # RE-EVALUATION:
    # The current program uses scipy.optimize.minimize which handles non-linearities.
    # My "LP" idea fails because the correlation involves h*h terms.
    # 
    # However, we can use a different strategy: **Alternating Direction Method of Multipliers (ADMM)** 
    # or **Coordinate Descent with Analytic Solution**.
    # 
    # Let's look at the structure again.
    # value = max_k (sum_i h_{i+k} - h_{i+k}h_i) / (N/2)
    # 
    # If we fix the "active" set of correlations (those achieving the max), the problem becomes non-convex.
    # 
    # Let's try a different "Materially Different" strategy:
    # **Simulated Annealing with a Physics-Inspired Potential**.
    # The current program uses random restarts + SLSQP/NM.
    # Simulated Annealing (SA) is a global optimization heuristic that can escape local minima 
    # by accepting worse solutions with a probability that decreases over time.
    # Given the rugged landscape of "max correlation", SA is often more robust than 
    # local gradient methods starting from random points.
    
    # Strategy: Simulated Annealing
    # 1. Start with a good initial guess (e.g., the best from a quick random scan).
    # 2. Define a neighbor generation: perturb a few random indices.
    # 3. Accept/reject based on Metropolis criterion.
    # 4. Cool down slowly.
    
    rng = np.random.RandomState(42)
    
    # 1. Quick random scan to find a decent starting point
    # We reuse the logic from the current program's start generation but simplified
    starts = []
    
    # Sine waves
    for freq in [0.1, 0.2, 0.3, 0.5]:
        for amp in [0.3, 0.5]:
            for phase in [0, np.pi/2, np.pi]:
                x = np.arange(N)
                h = 0.5 + amp * np.sin(2 * np.pi * freq * x + phase)
                h = np.clip(h, 0, 1)
                h = _fix_sum(h)
                starts.append(h)
    
    # Binary blocks
    for block_size in [2, 3, 4, 5, 10]:
        for _ in range(5):
            h = np.zeros(N)
            for i in range(0, N, block_size):
                val = rng.uniform(0, 1)
                h[i:i+block_size] = val
            h = _fix_sum(h)
            starts.append(h)
            
    # Random smooth
    for _ in range(10):
        h = 0.5 + rng.normal(0, 0.2, N)
        h = np.clip(h, 0, 1)
        h = _fix_sum(h)
        starts.append(h)

    # Evaluate starts
    best_h_sa = starts[0].copy()
    best_score_sa = _score(best_h_sa)
    for h in starts[1:]:
        s = _score(h)
        if s < best_score_sa:
            best_score_sa = s
            best_h_sa = h.copy()
    
    if best_score_sa < best_score:
        best_score = best_score_sa
        best_h = best_h_sa.copy()

    # 2. Simulated Annealing
    current_h = best_h_sa.copy()
    current_score = best_score_sa
    best_h_sa = current_h.copy()
    best_score_sa = current_score
    
    T = 1.0  # Initial temperature
    cooling = 0.995  # Cooling rate
    min_T = 1e-5
    
    # We will run SA until deadline
    while T > min_T and not deadline_check():
        
        # Generate neighbor
        # Perturb 1-3 random indices
        n_perturb = rng.randint(1, 4)
        idxs = rng.choice(N, size=n_perturb, replace=False)
        
        new_h = current_h.copy()
        for idx in idxs:
            # Move towards 0 or 1 or random? 
            # Random step is better for exploration
            step = rng.uniform(-0.1, 0.1)
            new_h[idx] += step
        
        # Enforce constraints
        new_h = np.clip(new_h, 0, 1)
        new_h = _fix_sum(new_h)
        
        if not _is_valid(new_h):
            continue
            
        new_score = _score(new_h)
        
        # Metropolis criterion
        delta_score = new_score - current_score
        if delta_score < 0:
            # Better, always accept
            current_h = new_h
            current_score = new_score
            if new_score < best_score_sa:
                best_score_sa = new_score
                best_h_sa = new_h.copy()
                if new_score < best_score:
                    best_score = new_score
                    best_h = new_h.copy()
        else:
            # Worse, accept with probability exp(-delta_score / T)
            # Note: delta_score is positive (worse)
            prob = np.exp(-delta_score / T)
            if rng.random() < prob:
                current_h = new_h
                current_score = new_score
        
        # Cool down
        T *= cooling
        
        # If stuck for a long time, reheat slightly? 
        # For now, keep it simple.
        
    # 3. Final Local Refinement on the SA result
    # Use a few steps of coordinate descent to polish
    if not deadline_check():
        h = best_h_sa.copy()
        current_score = _score(h)
        
        # Coordinate descent
        for _ in range(100):
            if deadline_check():
                break
            improved = False
            for i in range(N):
                # Try small steps in + and -
                for step in [0.01, 0.05, 0.1]:
                    for direction in [-1, 1]:
                        h_new = h.copy()
                        h_new[i] += direction * step
                        h_new = np.clip(h_new, 0, 1)
                        h_new = _fix_sum(h_new)
                        
                        if _is_valid(h_new):
                            s_new = _score(h_new)
                            if s_new < current_score:
                                current_score = s_new
                                h = h_new
                                improved = True
                                break
                    if improved:
                        break
            if not improved:
                break
                
        if current_score < best_score:
            best_score = current_score
            best_h = h
    
    # Final validation
    if not _is_valid(best_h):
        best_h = np.full(N, 0.5)
        
    return best_h
