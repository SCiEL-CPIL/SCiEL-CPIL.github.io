"""Erdos minimum overlap: Gradient-based optimization using SLSQP and smooth max."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# Theoretical optimum for continuous case: h=a on [0,1], h=b=1-a on [1,2]
# minimize max(a, (1-a)^2) -> a = (3 - sqrt(5))/2
A = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.381966
B = 1.0 - A                        # ~0.618034


def _compute_overlaps(h):
    """
    Compute the overlap of h with shifted (1-h) for all shifts k.
    overlap[k] = sum_i h[i] * (1 - h[(i+k) % N])
    """
    # autocorr[m] = sum_i h[i]*h[(i-m)%N]
    # sum_i h[i]*h[(i+k)%N] = autocorr[k] (since autocorr is even for real signals? 
    # Actually, circular autocorrelation R[k] = sum h[i]h[i+k]. R[k] = R[-k].
    # So sum_i h[i]h[(i+k)%N] is the circular autocorrelation at lag k.
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(np.abs(H) ** 2).real
    # autocorr[k] corresponds to sum_i h[i] h[(i-k)%N] in standard FFT definition?
    # Let's verify.
    # ifft(|fft(h)|^2)[k] = sum_m h[m] h[m-k] (circular).
    # We want sum_i h[i] h[(i+k)%N].
    # Let j = i+k -> i = j-k. sum_j h[j-k] h[j]. This is cross-correlation.
    # For real signals, autocorrelation is even, so R[k] = R[-k].
    # So autocorr[k] from ifft(|H|^2) is sum_i h[i] h[i-k].
    # We need sum_i h[i] h[i+k]. This is autocorr[-k] = autocorr[k].
    # So yes, autocorr[k] is the term.
    
    sum_h = np.sum(h)
    # overlap[k] = sum_i h[i] (1 - h[(i+k)%N]) = sum_h - sum_i h[i] h[(i+k)%N]
    overlaps = sum_h - autocorr
    return overlaps


def _smooth_max(overlaps, beta):
    """
    Smooth approximation of the maximum using log-sum-exp.
    """
    # Numerical stability: shift by max
    m = np.max(overlaps)
    if m > 100: # Avoid overflow in exp
        return m + (1.0/beta) * np.log(np.sum(np.exp(beta * (overlaps - m))))
    else:
        return (1.0/beta) * np.log(np.sum(np.exp(beta * overlaps)))


def _objective(h, beta):
    """
    Objective function to minimize.
    """
    overlaps = _compute_overlaps(h)
    return _smooth_max(overlaps, beta)


def _enforce_sum_constraint(h, target_sum):
    """Clip to [0,1] and adjust to make sum exactly target_sum."""
    h = np.clip(h, 0.0, 1.0)
    diff = target_sum - np.sum(h)
    if abs(diff) < 1e-14:
        return h
    if diff > 0:
        room = 1.0 - h
        total_room = np.sum(room)
        if total_room > 0:
            h += diff * room / total_room
    else:
        room = h
        total_room = np.sum(room)
        if total_room > 0:
            h += diff * room / total_room
    h = np.clip(h, 0.0, 1.0)
    # Final fine adjustment
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.argmax(1.0 - h)
            add = min(diff, 1.0 - h[idx])
            h[idx] += add
            remaining = diff - add
            if abs(remaining) > 1e-12:
                for i in range(N):
                    if abs(remaining) < 1e-12:
                        break
                    room = 1.0 - h[i]
                    if room > 0:
                        adj = min(remaining, room)
                        h[i] += adj
                        remaining -= adj
        else:
            idx = np.argmin(h)
            sub = min(-diff, h[idx])
            h[idx] -= sub
            remaining = -diff - sub
            if abs(remaining) > 1e-12:
                for i in range(N):
                    if abs(remaining) < 1e-12:
                        break
                    room = h[i]
                    if room > 0:
                        adj = min(remaining, room)
                        h[i] -= adj
                        remaining -= adj
    return np.clip(h, 0.0, 1.0)


def construct():
    """
    Construct step heights h in [0,1] with sum = N/2 that minimize
    the maximum overlap of h with shifted (1-h).
    
    Strategy:
    1. Start with the theoretical two-block construction.
    2. Use SLSQP to minimize the smooth max of overlaps subject to sum constraint.
    3. Refine with higher beta if time permits.
    """
    deadline = time.time() + 110.0  # Safety margin
    target_sum = N / 2.0
    
    # Initial guess: Theoretical two-block
    half = N // 2
    h0 = np.zeros(N)
    h0[:half] = A
    h0[half:] = B
    h0 = _enforce_sum_constraint(h0, target_sum)
    
    best_h = h0.copy()
    best_raw_score = _compute_max_overlap_val(best_h)
    
    # Define constraint
    cons = ({'type': 'eq', 'fun': lambda h: np.sum(h) - target_sum},)
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Phase 1: SLSQP with moderate beta
    beta = 20.0
    if time.time() < deadline:
        res = minimize(
            _objective, 
            h0, 
            args=(beta,),
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
        )
        if res.success or res.fun < best_raw_score:
            h_res = _enforce_sum_constraint(res.x, target_sum)
            score_res = _compute_max_overlap_val(h_res)
            if score_res < best_raw_score:
                best_raw_score = score_res
                best_h = h_res.copy()
    
    # Phase 2: SLSQP with higher beta (sharper max approximation)
    if time.time() < deadline - 30:
        beta = 50.0
        res2 = minimize(
            _objective, 
            best_h, 
            args=(beta,),
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
        )
        if res2.success or res2.fun < best_raw_score:
            h_res2 = _enforce_sum_constraint(res2.x, target_sum)
            score_res2 = _compute_max_overlap_val(h_res2)
            if score_res2 < best_raw_score:
                best_raw_score = score_res2
                best_h = h_res2.copy()

    # Phase 3: Coordinate descent refinement on the best solution found
    # This helps with the "exactly sum" constraint and fine-tuning
    if time.time() < deadline - 10:
        h_work = best_h.copy()
        step_sizes = [0.01, 0.005, 0.001]
        for step in step_sizes:
            if time.time() > deadline:
                break
            improved = True
            max_passes = 5
            for _pass in range(max_passes):
                if time.time() > deadline:
                    break
                if not improved:
                    break
                improved = False
                for i in range(N):
                    if time.time() > deadline:
                        break
                    old_val = h_work[i]
                    # Try increasing
                    new_val = min(1.0, old_val + step)
                    if abs(new_val - old_val) > 1e-10:
                        h_work[i] = new_val
                        h_work = _enforce_sum_constraint(h_work, target_sum)
                        new_score = _compute_max_overlap_val(h_work)
                        if new_score < best_raw_score - 1e-10:
                            best_raw_score = new_score
                            best_h = h_work.copy()
                            improved = True
                        else:
                            h_work[i] = old_val
                            h_work = _enforce_sum_constraint(h_work, target_sum)
                    
                    # Try decreasing
                    new_val = max(0.0, old_val - step)
                    if abs(new_val - old_val) > 1e-10:
                        h_work[i] = new_val
                        h_work = _enforce_sum_constraint(h_work, target_sum)
                        new_score = _compute_max_overlap_val(h_work)
                        if new_score < best_raw_score - 1e-10:
                            best_raw_score = new_score
                            best_h = h_work.copy()
                            improved = True
                        else:
                            h_work[i] = old_val
                            h_work = _enforce_sum_constraint(h_work, target_sum)

    # Final enforcement
    best_h = _enforce_sum_constraint(best_h, target_sum)
    
    # Final verification
    assert np.all(best_h >= -1e-10) and np.all(best_h <= 1.0 + 1e-10), "Heights out of bounds"
    assert abs(np.sum(best_h) - target_sum) < 1e-8, f"Sum constraint violated: {np.sum(best_h)} vs {target_sum}"
    
    return best_h

def _compute_max_overlap_val(h):
    """Helper to compute raw score for comparison."""
    overlaps = _compute_overlaps(h)
    return np.max(overlaps)
