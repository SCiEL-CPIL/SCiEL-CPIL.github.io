"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _eval_value(h):
    """Compute the objective: max correlate(h, 1-h, full) / N * 2."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0

def _project(h):
    """Project h to [0,1] with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    h = h - diff / N
    h = np.clip(h, 0.0, 1.0)
    # Second pass for residual
    diff2 = np.sum(h) - N / 2.0
    h = h - diff2 / N
    h = np.clip(h, 0.0, 1.0)
    return h

def _coordinate_descent(h, max_iter=200, tol=1e-10):
    """Fast coordinate descent: adjust one element at a time to reduce max corr."""
    h = h.copy()
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            old_val = h[i]
            # Try moving h[i] in small steps
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1, 0.001, -0.001]:
                h[i] = old_val + delta
                h = _project(h)
                v = _eval_value(h)
                if v < best_val_global - tol:
                    improved = True
                    break
                h[i] = old_val
            h = _project(h)
        if not improved:
            break
    return h

best_val_global = 2.0  # Will be set in construct

def _simulated_annealing(h, rng, t_start=1.0, t_end=0.001, n_iter=5000):
    """SA on the height vector with sum-preserving moves."""
    h = h.copy()
    current_val = _eval_value(h)
    best_h = h.copy()
    best_val = current_val
    
    n_total = n_iter
    for it in range(n_total):
        if time.time() > deadline_global - 3.0:
            break
        # Cooling schedule
        t = t_start * (t_end / t_start) ** (it / max(n_total - 1, 1))
        
        # Pick two random indices and transfer mass
        i, j = rng.choice(N, size=2, replace=False)
        delta = rng.uniform(-0.05, 0.05)
        
        h[i] += delta
        h[j] -= delta
        h = _project(h)
        
        new_val = _eval_value(h)
        delta_val = new_val - current_val
        
        if delta_val < 0 or rng.random() < np.exp(-delta_val / max(t, 1e-10)):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Revert
            h[i] -= delta
            h[j] += delta
            h = _project(h)
    
    return best_h, best_val

def construct():
    global best_val_global, deadline_global
    start_time = time.time()
    deadline_global = start_time + 145.0  # Safety margin: 5s before 150s kill
    
    rng = np.random.RandomState(42)
    
    # Start from seed
    h = np.full(N, 0.5)
    best_val_global = _eval_value(h)  # Should be ~2.0 * (N/2 * N/2) / N * 2 ... let's compute
    # Actually for h=0.5: correlate(0.5, 0.5, full) max = sum(0.25) = 25
    # value = 25/100*2 = 0.5? No wait...
    # np.correlate([0.5]*100, [0.5]*100, "full") -> the max is at lag 0: 100*0.25 = 25
    # value = 25/100*2 = 0.5
    # But the task says seed gives value such that score = 1.0, meaning seed/value = 1.0
    # So value_seed = 1.0? Let me re-read...
    # "The search score is seed/value" where seed is the value of the seed program
    # So score = value_seed / value_child. If child = seed, score = 1.0.
    # So value_seed / value_seed = 1.0. That's consistent.
    
    best_h = h.copy()
    
    # Phase 1: Random restarts with coordinate descent
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() > deadline_global - 20.0:
            break
        
        # Generate diverse initial configurations
        if r == 0:
            h_cur = np.full(N, 0.5)
        elif r % 3 == 1:
            # Alternating blocks
            h_cur = np.zeros(N)
            h_cur[::2] = 0.8
            h_cur[1::2] = 0.2
            h_cur = _project(h_cur)
        elif r % 3 == 2:
            # Random with mean 0.5
            h_cur = rng.uniform(0.0, 1.0, size=N)
            h_cur = _project(h_cur)
        else:
            # Sine wave with random frequency
            i = np.arange(N)
            freq = rng.uniform(1, 5)
            phase = rng.uniform(0, 2 * np.pi)
            h_cur = 0.5 + 0.3 * np.sin(2 * np.pi * freq * i / N + phase)
            h_cur = _project(h_cur)
        
        # Coordinate descent
        h_cur = _coordinate_descent(h_cur, max_iter=100)
        v = _eval_value(h_cur)
        if v < best_val_global:
            best_val_global = v
            best_h = h_cur.copy()
    
    # Phase 2: Simulated annealing from best found so far
    if time.time() < deadline_global - 10.0:
        # Run SA with multiple restarts from perturbed best
        for sa_restart in range(5):
            if time.time() > deadline_global - 5.0:
                break
            
            # Perturb the best solution
            h_sa = best_h.copy()
            noise = rng.normal(0, 0.05, size=N)
            h_sa = _project(h_sa + noise)
            
            n_iter = 10000 if sa_restart == 0 else 5000
            h_sa, val_sa = _simulated_annealing(h_sa, rng, t_start=0.5, t_end=0.0005, n_iter=n_iter)
            
            if val_sa < best_val_global:
                best_val_global = val_sa
                best_h = h_sa.copy()
        
        # Phase 3: Fine-tune with coordinate descent from SA result
        if time.time() < deadline_global - 5.0:
            h_final = _coordinate_descent(best_h, max_iter=300)
            v_final = _eval_value(h_final)
            if v_final < best_val_global:
                best_val_global = v_final
                best_h = h_final.copy()
    
    # Final verification
    h_final = _project(best_h)
    sum_check = np.sum(h_final) - N / 2.0
    if abs(sum_check) > 1e-9 * N:
        return np.full(N, 0.5)
    
    if not np.all(np.isfinite(h_final)):
        return np.full(N, 0.5)
    
    return h_final
# EVOLVE-BLOCK-END
