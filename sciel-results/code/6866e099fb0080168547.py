"""Autocorrelation inequality: piecewise-constant plateau optimization with
simulated annealing over (K, start, length, levels) and continuous refinement.

Strategy:
  - Represent h as a piecewise-constant function with K distinct non-zero plateau
    levels.  The combinatorial part (K, start, length, which level goes where) is
    searched via simulated annealing; the continuous part (the K level values) is
    refined with L-BFGS-B and then Nelder-Mead on the exact objective.
  - This transforms the 50-dim continuous search into a structured combinatorial
    + low-dim continuous problem that can find non-uniform plateau profiles
    that a uniform plateau sweep would miss.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1            # len(g) = 99
W = 1.0 / (M + 1)        # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)


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
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _neg_exact_value(h):
    h = np.maximum(0.0, h)
    return -compute_value(h)


def _plateau(j, k, level=1.0):
    h = np.zeros(N)
    end = min(N, j + k)
    if end > j:
        h[j:end] = level
    return h


# ---------------------------------------------------------------------------
# Piecewise-constant plateau model
# ---------------------------------------------------------------------------

def _build_h_from_plateaus(start, length, levels, segments):
    """
    Build h from a piecewise-constant description.
    
    segments: list of (seg_start, seg_len, level_idx) tuples.
    levels: array of K non-negative level values.
    
    Returns h of length N.
    """
    h = np.zeros(N)
    for seg_start, seg_len, level_idx in segments:
        if level_idx < 0 or level_idx >= len(levels):
            continue
        level = levels[level_idx]
        if level <= 0:
            continue
        end = min(N, seg_start + seg_len)
        if end > seg_start:
            h[seg_start:end] = level
    return h


def _plateaus_to_h(params, K, start, length, level_indices):
    """
    Convert parameters to h.
    
    params: (K,) array of level values.
    K: number of distinct levels.
    start: int, start index of the plateau region.
    length: int, total length of the plateau region (<= N).
    level_indices: (length,) array of ints in [0, K-1], which level at each position.
    
    Returns h of length N.
    """
    h = np.zeros(N)
    for i in range(length):
        idx = start + i
        if idx >= N:
            break
        li = level_indices[i]
        if 0 <= li < K:
            h[idx] = params[li]
    return h


# ---------------------------------------------------------------------------
# Simulated annealing over plateau structure
# ---------------------------------------------------------------------------

def _sa_plateau_optimize(initial_h, deadline, rng):
    """
    Simulated annealing over:
      - K: number of distinct levels (2..8)
      - start: start index of plateau region
      - length: length of plateau region
      - level_indices: which level at each position in the region
      - levels: continuous values for each level (refined with L-BFGS-B)
    
    Returns best h found.
    """
    best_h = initial_h.copy()
    best_val = compute_value(initial_h)
    
    # Initialize structure
    K = 2
    start = 0
    length = N
    # Initialize with uniform levels
    levels = np.array([1.0, 1.0])
    level_indices = np.zeros(length, dtype=int)  # all use level 0
    
    def _current_val():
        h = _plateaus_to_h(levels, K, start, length, level_indices)
        h = np.maximum(0.0, h)
        return compute_value(h)
    
    cur_val = _current_val()
    
    T = 0.1
    T_min = 1e-8
    cooling = 0.9995
    steps = 0
    max_steps = 500000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Random move
        move_type = rng.randint(0, 5)
        
        if move_type == 0:
            # Change K
            K_new = rng.randint(2, 9)
            if K_new != K:
                old_K = K
                old_levels = levels.copy()
                old_li = level_indices.copy()
                
                K = K_new
                levels = np.ones(K) * max(1.0, np.max(old_levels))
                # Remap level indices
                level_indices = np.zeros(length, dtype=int)
                for i in range(length):
                    if old_li[i] < old_K:
                        level_indices[i] = min(old_li[i], K - 1)
        
        elif move_type == 1:
            # Change start
            delta = rng.randint(-5, 6)
            start_new = max(0, min(N - 1, start + delta))
            if start_new != start:
                start = start_new
        
        elif move_type == 2:
            # Change length
            delta = rng.randint(-5, 6)
            length_new = max(2, min(N, length + delta))
            if length_new != length:
                length = length_new
                level_indices = level_indices[:length] if length < len(level_indices) else \
                    np.pad(level_indices, (0, length - len(level_indices)), 'constant', constant_values=0)
        
        elif move_type == 3:
            # Change a level index
            pos = rng.randint(0, length)
            old_li = level_indices[pos]
            level_indices[pos] = rng.randint(0, K)
        
        elif move_type == 4:
            # Change a level value
            li = rng.randint(0, K)
            delta = rng.normal(0, 0.1)
            levels[li] = max(0.0, levels[li] + delta)
        
        # Evaluate
        h_new = _plateaus_to_h(levels, K, start, length, level_indices)
        h_new = np.maximum(0.0, h_new)
        val_new = compute_value(h_new)
        
        # Accept/reject
        if val_new > cur_val or rng.random() < np.exp((val_new - cur_val) / max(T, 1e-15)):
            cur_val = val_new
            if val_new > best_val:
                best_val = val_new
                best_h = h_new.copy()
        else:
            # Revert
            if move_type == 0:
                K = old_K
                levels = old_levels
                level_indices = old_li
            elif move_type == 1:
                start = start - delta
            elif move_type == 2:
                length = length - delta
                level_indices = level_indices[:length]
            elif move_type == 3:
                level_indices[pos] = old_li
            elif move_type == 4:
                levels[li] = max(0.0, levels[li] - delta)
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Reheat occasionally
            if steps % 5000 == 0:
                T = 0.05
    
    return best_h, best_val


# ---------------------------------------------------------------------------
# Main construction
# ---------------------------------------------------------------------------

def construct():
    """Construct 50 nonnegative step heights to maximize the value."""
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 15.0)  # safety margin
    
    best_val = 0.0
    best_h = np.ones(N)
    
    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        h = np.nan_to_num(h, nan=0.0, posinf=1e6, neginf=0.0)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
    
    # --- Phase 1: exact analytic sweep over plateau widths -----------------
    for k in range(1, N + 1):
        h = _plateau(0, k)
        track(h)
    
    # Also try some shifted plateaus for diversity
    for k in [5, 10, 15, 20, 25, 30, 35, 40, 45]:
        for j in range(0, N - k + 1, max(1, (N - k) // 10)):
            h = _plateau(j, k)
            track(h)
    
    # --- Phase 2: Simulated annealing over plateau structure ----------------
    if time.time() < deadline:
        # Use best plateau as initial
        sa_h, sa_val = _sa_plateau_optimize(best_h, deadline - 30, RNG)
        track(sa_h)
    
    # --- Phase 3: L-BFGS-B refinement from best ----------------------------
    if time.time() < deadline:
        for attempt in range(10):
            if time.time() > deadline:
                break
            pert = np.maximum(0.0, best_h + RNG.normal(0.0, 0.03, N))
            try:
                res = opt.minimize(
                    _neg_exact_value,
                    pert,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                track(res.x)
            except Exception:
                pass
    
    # --- Phase 4: Nelder-Mead polish on exact objective ---------------------
    if time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline:
                break
            pert = best_h + RNG.normal(0.0, 0.05, N)
            pert = np.maximum(0.0, pert)
            try:
                res = opt.minimize(
                    _neg_exact_value,
                    pert,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
                )
                track(res.x)
            except Exception:
                pass
    
    # --- Phase 5: High-precision final polish --------------------------------
    if time.time() < deadline - 5.0:
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            pert = best_h + RNG.normal(0.0, 0.01, N)
            pert = np.maximum(0.0, pert)
            try:
                res = opt.minimize(
                    _neg_exact_value,
                    pert,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                track(res.x)
            except Exception:
                pass
    
    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
