"""Erdos minimum overlap - find step heights minimizing max overlap with shifted complement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
WIDTH = 2.0 / N

def compute_max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h) over all shifts (circular)."""
    N = len(h)
    width = 2.0 / N
    max_overlap = 0.0
    for s in range(1, N):
        h_shifted = np.roll(h, -s)
        overlap = np.sum(h * (1.0 - h_shifted)) * width
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _overlaps_fast(h):
    """Return array of overlaps for shifts 1..N-1 using FFT-based correlation."""
    N = len(h)
    width = 2.0 / N
    h_ext = np.zeros(2 * N)
    h_ext[:N] = h
    H = np.fft.fft(h, n=2*N)
    corr = np.fft.ifft(H * np.conj(H)).real[:N]
    # corr[s] = sum_i h[i]*h[(i-s) mod N]
    # We need sum_i h[i]*h[(i+s) mod N] = corr[(N-s) mod N]
    overlaps = np.empty(N - 1)
    for s in range(1, N):
        idx = (N - s) % N
        overlaps[s-1] = (N/2.0 - corr[idx]) * width
    return overlaps

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s buffer for safety
    
    best_h = np.full(N, 0.5)
    best_score = compute_max_overlap(best_h)
    
    # Initial patterns
    initial_patterns = []
    # Block patterns
    for k in range(2, N//2 + 1):
        if N % k == 0:
            p = np.zeros(N)
            p[:k//2] = 1.0 # Sum k/2? No, sum must be N/2.
            # Let's make patterns that sum to N/2.
            # Pattern: alternating blocks of 1 and 0.
            if N % 2 == 0:
                p = np.tile(np.array([1.0, 0.0]), N // 2)
                if p not in initial_patterns:
                    initial_patterns.append(p)
            if N % 4 == 0:
                p = np.tile(np.array([1.0, 1.0, 0.0, 0.0]), N // 4)
                if p not in initial_patterns:
                    initial_patterns.append(p)
            if N % 3 == 0:
                # 1, 1, 0 doesn't sum to N/2 if N is multiple of 3? 
                # Sum per 3 is 2. Total sum = 2N/3. We need N/2.
                # So we can't use simple integer blocks unless we adjust.
                pass
    
    # Add random initializations
    rng = np.random.RandomState(42)
    for _ in range(10):
        h_init = rng.rand(N)
        # Project to sum N/2 and [0,1]
        h_init = (h_init - np.mean(h_init)) * 0.5 + 0.5
        h_init = np.clip(h_init, 0, 1)
        # Adjust sum
        diff = N/2 - np.sum(h_init)
        if diff > 0:
            room = 1.0 - h_init
            total_room = np.sum(room)
            if total_room > 1e-15:
                h_init += diff * room / total_room
        else:
            room = h_init
            total_room = np.sum(room)
            if total_room > 1e-15:
                h_init += diff * room / total_room
        h_init = np.clip(h_init, 0, 1)
        initial_patterns.append(h_init)

    # Evaluate initial patterns
    for p in initial_patterns:
        if abs(np.sum(p) - N/2) < 1e-6:
            s = compute_max_overlap(p)
            if s < best_score:
                best_score = s
                best_h = p.copy()

    # Optimization using SLSQP
    # Variables: h[0..N-1], T
    # Objective: T
    # Constraints:
    # 1. sum(h) - N/2 = 0
    # 2. O_s(h) - T <= 0 for all s
    
    def objective(vars):
        h = vars[:N]
        T = vars[N]
        return T

    def constraints(vars):
        h = vars[:N]
        T = vars[N]
        cons = []
        # Sum constraint
        cons.append({'type': 'eq', 'fun': lambda v: np.sum(v[:N]) - N/2})
        # Overlap constraints
        overlaps = _overlaps_fast(h)
        for i, ov in enumerate(overlaps):
            s = i + 1
            # We only need to enforce constraints for shifts that are currently close to max
            # Or all of them? SLSQP can handle many constraints, but it's slow.
            # Let's use all of them for robustness, N=100 is small.
            cons.append({'type': 'ineq', 'fun': lambda v, s=s: T - _overlaps_fast(v[:N])[s-1]})
        return cons

    # This lambda approach in constraints is slow because it recomputes overlaps.
    # Better to precompute or use a vectorized constraint function if possible.
    # SLSQP requires a list of constraint dicts or a single dict with 'fun' returning array.
    
    def constraint_fun(vars):
        h = vars[:N]
        T = vars[N]
        overlaps = _overlaps_fast(h)
        # Inequality: T - overlap_s >= 0
        return T - overlaps

    def constraint_jac(vars):
        # Jacobian is hard to derive analytically for all constraints at once easily
        # SLSQP will approximate it.
        return None

    cons = [
        {'type': 'eq', 'fun': lambda v: np.sum(v[:N]) - N/2},
        {'type': 'ineq', 'fun': constraint_fun}
    ]
    
    bounds = [(0, 1)] * N + [(0, 10)] # T is bounded loosely

    # Run SLSQP from multiple starts
    for start_idx, h_start in enumerate(initial_patterns):
        if time.time() > deadline - 30:
            break
            
        x0 = np.concatenate([h_start, [best_score * 1.5]])
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.success:
                h_opt = res.x[:N]
                # Project to ensure constraints are met strictly
                h_opt = np.clip(h_opt, 0, 1)
                diff = N/2 - np.sum(h_opt)
                if diff > 0:
                    room = 1.0 - h_opt
                    total_room = np.sum(room)
                    if total_room > 1e-15:
                        h_opt += diff * room / total_room
                else:
                    room = h_opt
                    total_room = np.sum(room)
                    if total_room > 1e-15:
                        h_opt += diff * room / total_room
                h_opt = np.clip(h_opt, 0, 1)
                
                score = compute_max_overlap(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
                    
        except Exception as e:
            pass

    # Local Search / Coordinate Descent Polish
    if time.time() < deadline - 10:
        h_cd = best_h.copy()
        improved = True
        max_iter = 50
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline - 5:
            improved = False
            iter_count += 1
            for i in range(N):
                # Try small perturbations
                for delta in [0.01, 0.05, 0.1]:
                    for sign in [1, -1]:
                        h_new = h_cd.copy()
                        h_new[i] += sign * delta
                        # Project to maintain sum and bounds
                        h_new = np.clip(h_new, 0, 1)
                        diff = N/2 - np.sum(h_new)
                        if diff > 0:
                            room = 1.0 - h_new
                            total_room = np.sum(room)
                            if total_room > 1e-15:
                                h_new += diff * room / total_room
                        else:
                            room = h_new
                            total_room = np.sum(room)
                            if total_room > 1e-15:
                                h_new += diff * room / total_room
                        h_new = np.clip(h_new, 0, 1)
                        
                        score = compute_max_overlap(h_new)
                        if score < best_score - 1e-10:
                            best_score = score
                            best_h = h_new.copy()
                            h_cd = h_new.copy()
                            improved = True

    # Final check
    if abs(np.sum(best_h) - N/2) > 1e-8:
        best_h = np.full(N, 0.5)

    return best_h
