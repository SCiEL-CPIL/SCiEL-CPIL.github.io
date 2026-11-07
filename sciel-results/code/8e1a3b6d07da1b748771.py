"""Erdos minimum overlap: Spectral optimization approach."""
import time
import numpy as np
from scipy.optimize import minimize

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

def _objective(x):
    """Optimization objective: minimize max correlation."""
    h = x
    if np.any(h < -1e-6) or np.any(h > 1 + 1e-6):
        return 100.0 + np.sum(np.abs(h - np.clip(h, 0, 1)))
    h_clipped = np.clip(h, 0, 1)
    corr = np.correlate(h_clipped, 1 - h_clipped, mode="full")
    max_corr = np.max(corr)
    # Penalty for sum constraint violation
    sum_penalty = np.abs(np.sum(h_clipped) - N / 2.0) * 100.0
    return max_corr / N * 2 + sum_penalty

def _objective_smooth(x):
    """Smoothed objective for better gradient estimation."""
    h = np.clip(x, 0, 1)
    corr = np.correlate(h, 1 - h, mode="full")
    # Use a smooth max approximation
    max_corr = np.max(corr)
    sum_penalty = np.abs(np.sum(h) - N / 2.0) * 50.0
    return max_corr / N * 2 + sum_penalty

def construct():
    start_time = time.time()
    
    def deadline_check():
        return time.time() > start_time + DEADLINE
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Strategy: Use scipy.optimize with multiple random starts
    # The key insight: the problem is a constrained minimization of a max-type objective
    # We can use SLSQP or Nelder-Mead with proper constraints
    
    rng = np.random.RandomState(12345)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Random smooth functions (sine-based)
    for freq in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        for amp in [0.3, 0.4, 0.5]:
            for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
                x = np.arange(N)
                h = 0.5 + amp * np.sin(2 * np.pi * freq * x + phase)
                h = np.clip(h, 0, 1)
                h = _fix_sum(h)
                starts.append(h.copy())
    
    # 2. Random binary-like patterns
    for _ in range(10):
        h = rng.choice([0, 1], size=N).astype(float)
        # Adjust to have sum 50
        num_ones = np.sum(h == 1)
        if num_ones > 50:
            # Flip some 1s to 0s
            ones_idx = np.where(h == 1)[0]
            to_flip = rng.choice(ones_idx, size=num_ones - 50, replace=False)
            h[to_flip] = 0
        elif num_ones < 50:
            zeros_idx = np.where(h == 0)[0]
            to_flip = rng.choice(zeros_idx, size=50 - num_ones, replace=False)
            h[to_flip] = 1
        starts.append(h.copy())
    
    # 3. Block patterns with varying block sizes
    for block_size in [2, 3, 4, 5, 7, 10]:
        num_blocks = N // block_size
        for _ in range(3):
            h = np.zeros(N)
            for i in range(num_blocks):
                val = rng.uniform(0, 1)
                h[i*block_size:(i+1)*block_size] = val
            if N % block_size != 0:
                h[-(N % block_size):] = rng.uniform(0, 1, N % block_size)
            h = _fix_sum(h)
            starts.append(h.copy())
    
    # 4. Smooth random walks
    for _ in range(10):
        h = np.zeros(N)
        h[0] = 0.5
        for i in range(1, N):
            h[i] = h[i-1] + rng.normal(0, 0.05)
        h = np.clip(h, 0, 1)
        h = _fix_sum(h)
        starts.append(h.copy())
    
    # Evaluate all starts and keep the best ones
    scored_starts = []
    for h in starts:
        if deadline_check():
            break
        s = _score(h)
        scored_starts.append((s, h))
    
    scored_starts.sort(key=lambda x: x[0])
    
    # Refine the top candidates using scipy.optimize
    top_k = min(8, len(scored_starts))
    for s, h in scored_starts[:top_k]:
        if deadline_check():
            break
        
        # Use SLSQP with bounds and constraint
        try:
            bounds = [(0, 1)] * N
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0, 'jac': lambda x: np.ones(N)}
            ]
            
            result = minimize(
                _objective_smooth,
                h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if result.success:
                h_opt = result.x
                h_opt = np.clip(h_opt, 0, 1)
                h_opt = _fix_sum(h_opt)
                
                if _is_valid(h_opt):
                    s_opt = _score(h_opt)
                    if s_opt < best_score:
                        best_score = s_opt
                        best_h = h_opt.copy()
        except Exception:
            pass
        
        # Also try Nelder-Mead on the best SLSQP result
        if deadline_check():
            break
        try:
            result2 = minimize(
                _objective_smooth,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 500, 'fatol': 1e-10, 'adaptive': True}
            )
            if result2.success:
                h_nm = np.clip(result2.x, 0, 1)
                h_nm = _fix_sum(h_nm)
                if _is_valid(h_nm):
                    s_nm = _score(h_nm)
                    if s_nm < best_score:
                        best_score = s_nm
                        best_h = h_nm.copy()
        except Exception:
            pass
    
    # Final local refinement with coordinate descent
    if not deadline_check():
        h = best_h.copy()
        current_score = _score(h)
        step = 0.02
        for _ in range(200):
            if deadline_check():
                break
            idx = rng.randint(N)
            h_new = h.copy()
            h_new[idx] += rng.uniform(-step, step)
            h_new = np.clip(h_new, 0, 1)
            h_new = _fix_sum(h_new)
            if _is_valid(h_new):
                s_new = _score(h_new)
                if s_new < current_score:
                    current_score = s_new
                    h = h_new
        
        if current_score < best_score:
            best_score = current_score
            best_h = h
    
    # Final validation
    if not _is_valid(best_h):
        best_h = np.full(N, 0.5)
    
    return best_h
