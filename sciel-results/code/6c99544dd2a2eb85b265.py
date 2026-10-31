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
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(np.abs(H) ** 2).real
    # autocorr[k] = sum_i h[i] h[(i-k)%N] = sum_i h[i] h[(i+k)%N]
    sum_h = np.sum(h)
    overlaps = sum_h - autocorr
    return overlaps


def _compute_max_overlap_val(h):
    """Helper to compute raw score for comparison."""
    overlaps = _compute_overlaps(h)
    return np.max(overlaps)


def construct():
    """
    Construct step heights h in [0,1] with sum = N/2 that minimize
    the maximum overlap of h with shifted (1-h).
    """
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    target_sum = N / 2.0
    
    # Initial guess: Theoretical two-block
    half = N // 2
    h0 = np.zeros(N)
    h0[:half] = A
    h0[half:] = B
    # Ensure exact sum
    diff = target_sum - np.sum(h0)
    h0 += diff / N
    
    # We optimize over h and t, where t is the upper bound on overlap.
    # Variables: x = [h_0, ..., h_{N-1}, t]
    # Objective: minimize t
    # Constraints:
    # 1. sum(h) - target_sum = 0
    # 2. h_i >= 0, h_i <= 1 (handled by bounds)
    # 3. overlap_k - t <= 0 for all k
    
    def objective(x):
        h = x[:N]
        t = x[N]
        return t

    def constraint_sum(x):
        h = x[:N]
        return np.sum(h) - target_sum

    # Inequality constraints: overlap_k - t <= 0  =>  t - overlap_k >= 0
    # scipy expects fun(x) >= 0 for inequality constraints
    def make_overlap_constraint(k):
        def cons(x):
            h = x[:N]
            t = x[N]
            # Compute overlap for specific k
            # overlap_k = sum_i h[i] * (1 - h[(i+k) % N])
            # For efficiency in the loop, we could precompute, but N=100 is small.
            # However, SLSQP will call this many times.
            # Let's compute all overlaps once if possible? No, SLSQP calls constraint functions separately.
            # We can vectorize the constraint function to return an array?
            # SLSQP supports 'ineq' constraints as a single function returning an array of values >= 0.
            pass
        return cons

    # Better: Define a single inequality constraint function that returns an array
    def constraint_overlaps(x):
        h = x[:N]
        t = x[N]
        overlaps = _compute_overlaps(h)
        return t - overlaps  # We want t - overlap_k >= 0

    # Initial t estimate
    t0 = _compute_max_overlap_val(h0) + 0.1
    
    x0 = np.concatenate([h0, [t0]])
    
    # Bounds for h are [0, 1]. Bound for t is [0, 2] (max possible overlap is 1*1*100? No.
    # Max overlap: if h is 1 everywhere, sum=100 (invalid).
    # If h is 0.5 everywhere, overlap = 0.5*0.5*100 = 25.
    # If h is 1 on half, 0 on half. Overlap at shift 0: 50*1*0 + 50*0*1 = 0?
    # Wait. overlap[k] = sum h[i](1-h[i+k]).
    # If h is 1 on [0,50) and 0 on [50,100).
    # k=0: sum h[i](1-h[i]) = 0.
    # k=1: sum h[i](1-h[i+1]).
    # i=0..48: h[i]=1, h[i+1]=1 -> 0.
    # i=49: h[49]=1, h[50]=0 -> 1.
    # i=50..98: h[i]=0 -> 0.
    # i=99: h[99]=0, h[0]=1 -> 0.
    # So overlap[1] = 1.
    # Max overlap is 1?
    # Let's check k=50.
    # i=0..49: h[i]=1, h[i+50]=0 -> 1. Sum = 50.
    # i=50..99: h[i]=0 -> 0.
    # So overlap[50] = 50.
    # So t can be up to 50.
    # Theoretical min is ~0.382 * 100? No.
    # The score is the "raw value".
    # The problem statement says: "score is the upper bound on C5".
    # "Interface: construct() -> sequence of step heights ... whose sum is exactly len/2; score is the upper bound on C5"
    # "The evaluator measures the raw value (lower is better)."
    # In the current code, `_compute_max_overlap_val` returns `np.max(overlaps)`.
    # For the two-block solution, what is the max overlap?
    # h = [A]*50 + [B]*50.
    # overlap[k] = 50 - autocorr[k].
    # autocorr[k] for two blocks:
    # For k=0: 50*A^2 + 50*B^2.
    # For k=50: 50*A*B + 50*B*A = 100*AB.
    # A = 0.382, B = 0.618.
    # A^2 = 0.1459, B^2 = 0.3819.
    # autocorr[0] = 50*(0.1459 + 0.3819) = 50*0.5278 = 26.39.
    # overlap[0] = 50 - 26.39 = 23.61.
    # autocorr[50] = 100 * 0.382 * 0.618 = 100 * 0.236 = 23.6.
    # overlap[50] = 50 - 23.6 = 26.4.
    # So the max overlap is around 26.4.
    # Wait, the inspiration 1 score was 1.22. The base score is 1.0.
    # The search score is seed/value.
    # If base score is 1.0, and value is V_base.
    # If new score is 1.22, value is V_new = V_base / 1.22.
    # So we need to reduce the raw overlap value.
    
    # Let's re-read carefully.
    # "score is the upper bound on C5"
    # "The evaluator measures the raw value (lower is better)."
    # "search score is seed/value"
    # So if we minimize the raw value, the score increases.
    
    # The current program returns `best_h`.
    # The evaluator calculates the score from `best_h`.
    
    # So we just need to find `h` that minimizes `max_k overlap_k`.
    
    # Let's use SLSQP with the auxiliary variable t.
    
    cons = (
        {'type': 'eq', 'fun': constraint_sum},
        {'type': 'ineq', 'fun': constraint_overlaps}
    )
    
    bounds = [(0.0, 1.0)] * N + [(0.0, 50.0)]  # t bound
    
    best_h = h0.copy()
    best_raw_score = _compute_max_overlap_val(best_h)
    
    # Phase 1: SLSQP
    if time.time() < deadline - 5:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
        )
        
        if res.success or res.fun < best_raw_score:
            h_res = res.x[:N]
            # Enforce sum and bounds strictly
            h_res = np.clip(h_res, 0.0, 1.0)
            diff = target_sum - np.sum(h_res)
            # Simple adjustment: distribute diff to elements with most room
            if diff > 0:
                room = 1.0 - h_res
                total_room = np.sum(room)
                if total_room > 0:
                    h_res += diff * room / total_room
            else:
                room = h_res
                total_room = np.sum(room)
                if total_room > 0:
                    h_res += diff * room / total_room
            
            h_res = np.clip(h_res, 0.0, 1.0)
            # Final fine tune sum
            diff = target_sum - np.sum(h_res)
            if abs(diff) > 1e-10:
                idx = np.argmax(1.0 - h_res) if diff > 0 else np.argmin(h_res)
                adj = diff if diff > 0 else diff
                if diff > 0:
                    h_res[idx] += min(diff, 1.0 - h_res[idx])
                else:
                    h_res[idx] += max(diff, -h_res[idx])
                
            score_res = _compute_max_overlap_val(h_res)
            if score_res < best_raw_score:
                best_raw_score = score_res
                best_h = h_res.copy()
                x0 = np.concatenate([best_h, [best_raw_score + 0.01]])

    # Phase 2: Refine with higher precision / more iterations if time allows
    if time.time() < deadline - 10:
        res2 = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
        )
        
        if res2.success or res2.fun < best_raw_score:
            h_res2 = res2.x[:N]
            h_res2 = np.clip(h_res2, 0.0, 1.0)
            diff = target_sum - np.sum(h_res2)
            if diff > 0:
                room = 1.0 - h_res2
                total_room = np.sum(room)
                if total_room > 0:
                    h_res2 += diff * room / total_room
            else:
                room = h_res2
                total_room = np.sum(room)
                if total_room > 0:
                    h_res2 += diff * room / total_room
            
            h_res2 = np.clip(h_res2, 0.0, 1.0)
            diff = target_sum - np.sum(h_res2)
            if abs(diff) > 1e-10:
                idx = np.argmax(1.0 - h_res2) if diff > 0 else np.argmin(h_res2)
                if diff > 0:
                    h_res2[idx] += min(diff, 1.0 - h_res2[idx])
                else:
                    h_res2[idx] += max(diff, -h_res2[idx])

            score_res2 = _compute_max_overlap_val(h_res2)
            if score_res2 < best_raw_score:
                best_raw_score = score_res2
                best_h = h_res2.copy()

    # Final enforcement
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = target_sum - np.sum(best_h)
    if abs(diff) > 1e-10:
        # Distribute evenly to all non-boundary elements?
        # Just add to one element if simple, or distribute.
        # To be safe, distribute to all elements proportional to their distance from bounds.
        if diff > 0:
            room = 1.0 - best_h
            total_room = np.sum(room)
            if total_room > 0:
                best_h += diff * room / total_room
        else:
            room = best_h
            total_room = np.sum(room)
            if total_room > 0:
                best_h += diff * room / total_room
        
    best_h = np.clip(best_h, 0.0, 1.0)
    
    # Verify
    assert np.all(best_h >= -1e-10) and np.all(best_h <= 1.0 + 1e-10), "Heights out of bounds"
    assert abs(np.sum(best_h) - target_sum) < 1e-8, f"Sum constraint violated: {np.sum(best_h)} vs {target_sum}"
    
    return best_h
