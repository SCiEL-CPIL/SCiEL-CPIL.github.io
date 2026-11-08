import numpy as np
import time
import random

N = 50
M = 2 * N - 1
W = 1.0 / (M + 1)
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
    """Exact evaluator metric for given heights."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    
    if np.all(h == 0):
        return 0.0
        
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    
    # l2 term: sum of (w/3) * (y_i^2 + y_i y_{i+1} + y_{i+1}^2)
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    
    # norm_1 = sum(abs(g)) / (m + 1)
    # Since h >= 0, g >= 0, so abs(g) = g
    norm_1 = np.sum(g) / (m + 1)
    
    # norm_inf = max(abs(g))
    norm_inf = np.max(g)
    
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = 0.0
    best_h = np.ones(N)
    best_h = best_h / np.max(best_h) # Normalize to 1
    
    # Initial diverse starts
    starts = [
        np.ones(N),
        np.linspace(1, 0.1, N),
        np.linspace(0.1, 1, N),
        np.array([1.0 if i % 2 == 0 else 0.5 for i in range(N)]),
        np.array([1.0 if i < N//2 else 0.1 for i in range(N)]),
        np.array([1.0 if i >= N//2 else 0.1 for i in range(N)]),
        np.exp(-np.linspace(0, 2, N)),
        np.exp(-np.linspace(2, 0, N)),
    ]
    
    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
            return True
        return False

    for s in starts:
        track(s)

    # State representation for Simulated Annealing:
    # A state is defined by K plateaus.
    # Each plateau has a start index, end index (exclusive), and height.
    # We represent this as a list of tuples: (start, end, height)
    # To make SA easier, we'll maintain a full vector h, but generate moves 
    # that respect the "plateau" structure or allow transitions between structures.
    
    # Actually, a simpler combinatorial state is just the vector h, 
    # but we bias the moves to create flat segments.
    # Let's use a hybrid: 
    # State: h (vector of 50)
    # Moves:
    # 1. Change height of a single index (local noise)
    # 2. Set a random contiguous block to a random height (plateau creation)
    # 3. Merge two adjacent plateaus of similar height (plateau smoothing)
    # 4. Split a plateau into two with different heights (plateau refinement)
    
    current_h = best_h.copy()
    current_val = best_val
    
    # SA Parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    steps_per_temp = 50
    
    rng = np.random.RandomState(42)
    
    while time.time() < deadline - 5.0:
        for _ in range(steps_per_temp):
            if time.time() > deadline - 2.0:
                break
                
            h_new = current_h.copy()
            move_type = rng.randint(0, 4)
            
            if move_type == 0:
                # Single point change
                idx = rng.randint(0, N)
                h_new[idx] = max(0.0, h_new[idx] + rng.normal(0, 0.2))
                
            elif move_type == 1:
                # Block assignment (Plateau creation)
                width = rng.randint(2, 10)
                start = rng.randint(0, N - width)
                height = max(0.0, rng.uniform(0.1, 2.0))
                h_new[start:start+width] = height
                
            elif move_type == 2:
                # Merge neighbors: Find adjacent indices with similar height and equalize
                # This reduces "noise" and creates larger plateaus
                idx = rng.randint(0, N-1)
                # If they are close, average them
                if abs(h_new[idx] - h_new[idx+1]) < 0.5:
                    avg = (h_new[idx] + h_new[idx+1]) / 2.0
                    h_new[idx] = avg
                    h_new[idx+1] = avg
                    # Extend merge probability
                    if rng.rand() < 0.5:
                        if idx > 0 and abs(h_new[idx-1] - avg) < 0.5:
                            h_new[idx-1] = avg
                        if idx+1 < N-1 and abs(h_new[idx+2] - avg) < 0.5:
                            h_new[idx+2] = avg

            elif move_type == 3:
                # Split plateau: Find a flat region and perturb the middle
                # Identify a run of equal values (approx)
                idx = rng.randint(1, N-1)
                if abs(h_new[idx] - h_new[idx-1]) < 0.1 and abs(h_new[idx] - h_new[idx+1]) < 0.1:
                    # Perturb the middle one
                    h_new[idx] = max(0.0, h_new[idx] + rng.normal(0, 0.5))
                    
            elif move_type == 4:
                # Random restart / Jumpy move
                # Copy a random segment from best_h or generate new
                if rng.rand() < 0.5:
                    width = rng.randint(5, 15)
                    start = rng.randint(0, N - width)
                    h_new[start:start+width] = best_h[start:start+width] * rng.uniform(0.5, 1.5)
                else:
                    # Large jump
                    h_new = np.maximum(0, h_new + rng.normal(0, 0.5, N))

            # Ensure non-negative
            h_new = np.maximum(0.0, h_new)
            
            # Evaluate
            val_new = compute_value(h_new)
            
            # Acceptance criterion
            if val_new > current_val:
                current_h = h_new
                current_val = val_new
                if track(h_new):
                    # Update best
                    pass
            else:
                # Metropolis criterion
                delta = current_val - val_new
                if T > 0:
                    prob = np.exp(-delta / T)
                    if rng.rand() < prob:
                        current_h = h_new
                        current_val = val_new

        # Cooling
        T *= cooling
        if T < T_min:
            # Reheat occasionally if stuck
            if rng.rand() < 0.2:
                T = 0.5
            else:
                T = T_min

    # Final polish: Local refinement on the best found structure
    # Use a simple gradient-free local search (Nelder-Mead) on the exact objective
    # restricted to small perturbations to fix boundary effects
    if time.time() < deadline - 1.0:
        try:
            from scipy.optimize import minimize
            
            def objective(x):
                h = np.maximum(0.0, x)
                return -compute_value(h)
                
            res = minimize(objective, best_h, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < -best_val:
                track(res.x)
        except Exception:
            pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    
    # One last check
    final_val = compute_value(best_h)
    if final_val > best_val:
        best_val = final_val

    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
