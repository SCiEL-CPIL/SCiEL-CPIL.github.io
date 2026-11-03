import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Minimize the maximum overlap of h with shifted (1-h).

    Uses a smooth log-sum-exp surrogate for the max overlap, multi-start
    L-BFGS-B with structured initializations, and a coordinate-descent
    polish that preserves the sum and bounds.
    """
    deadline = time.time() + 145  # Safety margin

    N = 100

    # Precompute shift indices for a fast overlap via np.roll
    # For shift k, overlap = sum_i h[i] * (1 - h[(i+k) % N])
    #                       = sum(h) - sum_i h[i]*h[(i+k)%N]
    # We compute all shifts at once using FFT autocorrelation (fast path)
    # and also keep a direct roll-based path for verification.

    def autocorr_fft(h):
        """Circular autocorrelation via FFT."""
        F = np.fft.fft(h)
        ac = np.real(np.fft.ifft(F * np.conj(F)))
        return ac

    def compute_C(h):
        """
        C(k) = sum_i h[i]*(1 - h[(i+k)%N]) for k=1..N-1.
        = N*0.5 - autocorr(k)  when sum(h)=N/2.
        More generally = sum(h) - autocorr(k).
        """
        s = h.sum()
        ac = autocorr_fft(h)
        # ac[k] = sum_i h[i]*h[(i-k)%N] = sum_i h[i]*h[(i+k)%N] (circular, symmetric)
        C_vals = s - ac[1:]
        return C_vals

    def max_overlap(h):
        C_vals = compute_C(h)
        return float(np.max(C_vals))

    def smooth_max(h, beta):
        """Smooth log-sum-exp surrogate for max(C(k))."""
        C_vals = compute_C(h)
        C_max = np.max(C_vals)
        if C_max < -1e6:
            return 0.0
        exp_C = np.exp(beta * (C_vals - C_max))
        return C_max + np.log(np.sum(exp_C)) / beta

    def project_feasible(h):
        """Project onto {0<=h<=1, sum(h)=N/2} via iterative clipping + mean fix."""
        h = h.copy()
        for _ in range(20):
            h = np.clip(h, 0.0, 1.0)
            diff = N / 2.0 - h.sum()
            if abs(diff) < 1e-14:
                break
            h += diff / N
        # Final exact sum fix (tiny residual)
        h = np.clip(h, 0.0, 1.0)
        diff = N / 2.0 - h.sum()
        if abs(diff) > 0:
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
        return h

    def make_objective(beta):
        def obj(params):
            return smooth_max(params, beta)
        return obj

    # --- Initial candidates ---
    candidates = []

    # 1. Constant 0.5
    candidates.append(np.full(N, 0.5))

    # 2. Alternating 0,1 pattern (sum = N/2 = 50)
    alt = np.zeros(N)
    alt[1::2] = 1.0
    candidates.append(alt)

    # 3. Alternating with amplitude a: h[i] = 0.5 + a*(-1)^i, sum=50
    #    For even N, sum = N*0.5 = 50 regardless of a.
    for a in [0.3, 0.5, 0.7, 0.9]:
        h_alt = 0.5 + a * np.array([1 if i % 2 == 0 else -1 for i in range(N)])
        h_alt = np.clip(h_alt, 0.0, 1.0)
        candidates.append(h_alt)

    # 4. Period-3 pattern: values cycling [v1, v2, v3] with mean 0.5
    #    Try a few structured patterns
    for (v1, v2, v3) in [(0.0, 1.0, 0.5), (0.2, 0.8, 0.5), (0.0, 0.5, 1.0),
                          (0.1, 0.9, 0.4), (0.3, 0.3, 0.8)]:
        h_p3 = np.tile([v1, v2, v3], N // 3 + 1)[:N]
        h_p3 = np.clip(h_p3, 0.0, 1.0)
        h_p3 = project_feasible(h_p3)
        candidates.append(h_p3)

    # 5. Random binary-ish patterns
    rng_init = np.random.default_rng(12345)
    for _ in range(8):
        h_rand = rng_init.random(N)
        # Push toward extremes
        h_rand = (h_rand > 0.5).astype(float)
        h_rand = project_feasible(h_rand)
        candidates.append(h_rand)

    # 6. Smooth sinusoidal perturbation of 0.5
    for freq in [1, 2, 3, 5]:
        t = np.arange(N)
        h_sin = 0.5 + 0.3 * np.sin(2 * np.pi * freq * t / N)
        h_sin = np.clip(h_sin, 0.0, 1.0)
        h_sin = project_feasible(h_sin)
        candidates.append(h_sin)

    # Evaluate all candidates
    best_h = candidates[0].copy()
    best_val = max_overlap(best_h)
    for h_c in candidates:
        val = max_overlap(h_c)
        if val < best_val:
            best_val = val
            best_h = h_c.copy()

    bounds = [(0.0, 1.0)] * N

    # --- Phase 1: Coarse L-BFGS-B from best candidate and a few others ---
    starts = [best_h]
    # Add top-3 distinct candidates by value
    scored = sorted([(max_overlap(h), h) for h in candidates], key=lambda x: x[0])
    for _, h_s in scored[:4]:
        if not np.allclose(h_s, best_h, atol=1e-8):
            starts.append(h_s)

    for h_start in starts:
        if time.time() > deadline:
            break

        for beta in [15.0, 40.0]:
            if time.time() > deadline:
                break
            try:
                result = minimize(
                    make_objective(beta),
                    h_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                h_opt = project_feasible(result.x)
                val = max_overlap(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
            except Exception:
                pass

    # --- Phase 2: Fine L-BFGS-B with higher beta from best ---
    for beta in [80.0, 150.0]:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                make_objective(beta),
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = project_feasible(result.x)
            val = max_overlap(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass

    # --- Phase 3: Coordinate descent polish ---
    # For each coordinate i, find the best value in [0,1] that minimizes max_overlap,
    # adjusting one other coordinate to preserve the sum.
    rng = np.random.default_rng(42)
    h = best_h.copy()

    # Precompute indices
    idx = np.arange(N)

    polish_steps = 0
    max_polish_steps = 3000
    while polish_steps < max_polish_steps:
        if time.time() > deadline:
            break
        polish_steps += 1

        # Cycle through coordinates in a shuffled order
        order = idx.copy()
        rng.shuffle(order)

        for i in order:
            if time.time() > deadline:
                break

            # Choose partner j != i to adjust for sum preservation
            j = (i + 1) % N
            if j == i:
                j = (i + N // 2) % N

            h_i = h[i]
            h_j = h[j]

            # h_i can vary in [max(0, h_i + h_j - 1), min(1, h_i + h_j)]
            # because h_j = h_i_old + h_j_old - h_i_new must be in [0,1]
            lo = max(0.0, h_i + h_j - 1.0)
            hi = min(1.0, h_i + h_j)

            if hi - lo < 1e-12:
                continue

            # Try a few candidate values for h_i
            # Use ternary-like search: evaluate at a few points
            # Since max_overlap is piecewise-linear-ish, try quantiles
            for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
                h_i_new = lo + frac * (hi - lo)
                h_j_new = h_i + h_j - h_i_new
                if h_i_new < -1e-12 or h_i_new > 1 + 1e-12:
                    continue
                if h_j_new < -1e-12 or h_j_new > 1 + 1e-12:
                    continue

                h[i] = h_i_new
                h[j] = h_j_new
                val = max_overlap(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
                else:
                    # Revert
                    h[i] = h_i
                    h[j] = h_j

            # Also try the midpoint of the current best interval
            mid = (lo + hi) / 2.0
            h_j_mid = h_i + h_j - mid
            if 0 <= mid <= 1 and 0 <= h_j_mid <= 1:
                h[i] = mid
                h[j] = h_j_mid
                val = max_overlap(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
                else:
                    h[i] = h_i
                    h[j] = h_j

        # Occasionally do a random pairwise perturbation
        if polish_steps % 50 == 0:
            for _ in range(10):
                if time.time() > deadline:
                    break
                i, j = rng.integers(0, N, size=2)
                if i == j:
                    continue
                delta = rng.uniform(-0.05, 0.05)
                if 0 <= h[i] + delta <= 1 and 0 <= h[j] - delta <= 1:
                    h[i] += delta
                    h[j] -= delta
                    val = max_overlap(h)
                    if val < best_val:
                        best_val = val
                        best_h = h.copy()
                    else:
                        h[i] -= delta
                        h[j] += delta

    # Final validation
    best_h = project_feasible(best_h)
    # Ensure sum is as close to N/2 as possible
    diff = N / 2.0 - best_h.sum()
    if abs(diff) > 0:
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)

    return best_h
