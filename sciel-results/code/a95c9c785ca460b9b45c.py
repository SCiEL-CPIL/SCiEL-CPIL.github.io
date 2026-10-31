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
    H = np.fft.fft(h, n=2 * N)
    corr = np.fft.ifft(H * np.conj(H)).real[:N]
    overlaps = np.empty(N - 1)
    for s in range(1, N):
        idx = (N - s) % N
        overlaps[s - 1] = (N / 2.0 - corr[idx]) * width
    return overlaps


def _smooth_max(x, tau):
    """Smooth max via log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    m = np.max(x)
    return m + tau * np.log(np.sum(np.exp((x - m) / tau)))


def _build_sawtooth(a, b, run_a, run_b, phase):
    """
    Build a two-level periodic pattern with values a and b.
    Pattern: run_a copies of a, then run_b copies of b, repeated.
    phase shifts the starting position within the period.
    Total length is N.
    """
    period = run_a + run_b
    if period < 2:
        period = 2
    if run_a < 0:
        run_a = 0
    if run_b < 0:
        run_b = 0
    if run_a + run_b == 0:
        run_a = 1
        run_b = 1

    # Build one period
    one_period = np.concatenate([np.full(run_a, a), np.full(run_b, b)])
    # Tile to fill N
    reps = int(np.ceil(N / period))
    h = np.tile(one_period, reps)[:N]
    # Apply phase shift
    if phase != 0:
        h = np.roll(h, -int(phase))
    return h


def _project_to_feasible(h, N_steps):
    """Project h to [0,1] with sum exactly N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N_steps / 2.0
    diff = target - np.sum(h)
    if abs(diff) < 1e-15:
        return h
    if diff > 0:
        room = 1.0 - h
        total_room = np.sum(room)
        if total_room < 1e-15:
            return h
        h = h + diff * room / total_room
    else:
        room = h
        total_room = np.sum(room)
        if total_room < 1e-15:
            return h
        h = h + diff * room / total_room
    return np.clip(h, 0.0, 1.0)


def _objective_smooth(params):
    """
    Objective: smooth max of overlaps for a sawtooth pattern.
    params = [a, b, run_a, run_b, phase]
    """
    a, b, run_a, run_b, phase = params
    # Clip to valid ranges
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    run_a = max(1, int(round(run_a)))
    run_b = max(1, int(round(run_b)))
    phase = int(phase) % (run_a + run_b)

    h = _build_sawtooth(a, b, run_a, run_b, phase)
    h = _project_to_feasible(h, N)
    ov = _overlaps_fast(h)
    return _smooth_max(ov, 0.01)


def _gradient_descent(params, max_iter=200, lr=0.05):
    """Simple gradient descent with finite differences."""
    x = np.array(params, dtype=np.float64)
    best_x = x.copy()
    best_f = _objective_smooth(x)
    
    for it in range(max_iter):
        # Compute gradient via central differences
        grad = np.zeros_like(x)
        eps = 0.01
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (_objective_smooth(x_plus) - _objective_smooth(x_minus)) / (2 * eps)
        
        # Update
        x_new = x - lr * grad
        f_new = _objective_smooth(x_new)
        
        if f_new < best_f:
            best_f = f_new
            best_x = x_new.copy()
            x = x_new
            # Decay learning rate
            lr *= 0.98
        else:
            lr *= 0.9
            if lr < 1e-6:
                break
    return best_x, best_f


def construct():
    start_time = time.time()
    deadline = start_time + 145

    best_h = np.full(N, 0.5)
    best_score = compute_max_overlap(best_h)

    # Try structured initial patterns
    initial_patterns = []
    if N % 4 == 0:
        p = np.tile(np.array([1.0, 1.0, 0.0, 0.0]), N // 4)
        initial_patterns.append(p)
    if N % 2 == 0:
        p = np.tile(np.array([1.0, 0.0]), N // 2)
        initial_patterns.append(p)
    p = np.zeros(N)
    p[:N // 2] = 1.0
    initial_patterns.append(p)
    if N % 3 == 0:
        p = np.tile(np.array([1.0, 1.0, 0.0]), N // 3)
        initial_patterns.append(p)
    if N % 5 == 0:
        p = np.tile(np.array([1.0, 1.0, 0.0, 0.0, 0.0]), N // 5)
        initial_patterns.append(p)

    for p in initial_patterns:
        if abs(np.sum(p) - N / 2) < 1e-10:
            s = compute_max_overlap(p)
            if s < best_score:
                best_score = s
                best_h = p.copy()

    # Phase 1: Sawtooth pattern search with smooth-max objective
    # Parameters: [a, b, run_a, run_b, phase]
    # We search over a grid of run lengths and optimize a, b via smooth objective

    run_a_vals = list(range(1, 15))
    run_b_vals = list(range(1, 15))

    for run_a in run_a_vals:
        for run_b in run_b_vals:
            if time.time() > deadline - 30:
                break

            # Try several initial (a, b) pairs
            initial_ab = [
                (1.0, 0.0),
                (0.8, 0.2),
                (0.7, 0.3),
                (0.6, 0.4),
                (0.5, 0.5),
                (0.4, 0.6),
                (0.3, 0.7),
                (0.2, 0.8),
            ]

            for a0, b0 in initial_ab:
                if time.time() > deadline - 25:
                    break

                # For each phase, try to optimize
                period = run_a + run_b
                for phase in range(0, min(period, 8)):
                    if time.time() > deadline - 20:
                        break

                    params = np.array([a0, b0, float(run_a), float(run_b), float(phase)])

                    # Use scipy minimize with bounds
                    bounds = [
                        (0.0, 1.0),  # a
                        (0.0, 1.0),  # b
                        (float(run_a), float(run_a)),  # run_a fixed
                        (float(run_b), float(run_b)),  # run_b fixed
                        (0.0, float(period)),  # phase
                    ]

                    try:
                        res = minimize(_objective_smooth, params, method='L-BFGS-B',
                                       bounds=bounds,
                                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                        h_opt = _build_sawtooth(res.x[0], res.x[1], int(round(res.x[2])),
                                                int(round(res.x[3])), int(round(res.x[4])))
                        h_opt = _project_to_feasible(h_opt, N)
                        score = compute_max_overlap(h_opt)
                        if score < best_score:
                            best_score = score
                            best_h = h_opt.copy()
                    except Exception:
                        pass

    # Phase 2: Local refinement via coordinate descent on the best pattern found
    if time.time() < deadline - 15:
        h_cd = best_h.copy()
        for cd_iter in range(50):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(0, N - 1, 2):
                j = i + 1
                h_old_i = h_cd[i]
                h_old_j = h_cd[j]
                for delta in [0.05, 0.1, 0.2, 0.3, 0.5]:
                    h_new_i = np.clip(h_old_i + delta, 0, 1)
                    h_new_j = np.clip(h_old_j - delta, 0, 1)
                    h_cd[i] = h_new_i
                    h_cd[j] = h_new_j
                    score = compute_max_overlap(h_cd)
                    if score < best_score - 1e-12:
                        best_score = score
                        best_h = h_cd.copy()
                        improved = True
                    h_cd[i] = h_old_i
                    h_cd[j] = h_old_j
                for delta in [0.05, 0.1, 0.2, 0.3, 0.5]:
                    h_new_i = np.clip(h_old_i - delta, 0, 1)
                    h_new_j = np.clip(h_old_j + delta, 0, 1)
                    h_cd[i] = h_new_i
                    h_cd[j] = h_new_j
                    score = compute_max_overlap(h_cd)
                    if score < best_score - 1e-12:
                        best_score = score
                        best_h = h_cd.copy()
                        improved = True
                    h_cd[i] = h_old_i
                    h_cd[j] = h_old_j
            if not improved:
                break

    # Ensure sum is exactly N/2
    if abs(np.sum(best_h) - N / 2) > 1e-10:
        diff = N / 2 - np.sum(best_h)
        if diff > 0:
            room = 1.0 - best_h
            total_room = np.sum(room)
            if total_room > 1e-15:
                best_h += diff * room / total_room
        else:
            room = best_h
            total_room = np.sum(room)
            if total_room > 1e-15:
                best_h += diff * room / total_room
        best_h = np.clip(best_h, 0.0, 1.0)

    if abs(np.sum(best_h) - N / 2) > 1e-8:
        best_h = np.full(N, 0.5)

    return best_h
