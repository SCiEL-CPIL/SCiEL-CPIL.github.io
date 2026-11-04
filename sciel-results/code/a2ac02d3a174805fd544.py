"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START

def construct():
    """
    Minimize the maximum overlap of h with shifted (1-h) using smooth
    log-sum-exp approximation and L-BFGS-B with constraint projection.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def get_correlations(h):
        """Compute overlaps for all non-zero circular shifts."""
        # For circular shifts, use FFT-based circular correlation
        # Overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
        # = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
        # = N/2 - circular_autocorr(h, k)
        # So minimizing max overlap = minimizing max_k (N/2 - autocorr_k)
        # = N/2 - min_k autocorr_k
        # We want to maximize min_k autocorr_k
        # Equivalent: minimize max_k (N/2 - autocorr_k)
        
        # Compute circular autocorrelation via FFT
        H = np.fft.fft(h)
        autocorr = np.fft.ifft(H * np.conj(H)).real
        # autocorr[k] = sum_i h[i] * h[(i+k) % N]
        # For shift k, overlap = N/2 - autocorr[k]
        # We exclude k=0 (no shift) since that's trivially N/2 - sum(h^2)
        overlaps = N / 2.0 - autocorr
        # Exclude k=0
        overlaps_nonzero = np.concatenate([overlaps[1:], overlaps[:0]])
        # Actually, for N bins, shifts are 1..N-1
        # autocorr has N elements, index 0 is k=0
        overlaps_nonzero = overlaps[1:]
        return overlaps_nonzero

    def lse_objective(h, beta):
        """Log-sum-exp approximation of max overlap."""
        overlaps = get_correlations(h)
        max_ov = np.max(overlaps)
        if max_ov > 500:
            max_ov = 500.0
        lse = max_ov + np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) / beta
        return lse

    def project_to_feasible(h):
        """Project h onto {0 <= h_i <= 1, sum(h) = N/2}."""
        h = np.clip(h, 0.0, 1.0)
        for _ in range(100):
            current_sum = np.sum(h)
            diff = N / 2.0 - current_sum
            if abs(diff) < 1e-15:
                break
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
        # Final exact correction
        residual = N / 2.0 - np.sum(h)
        if abs(residual) > 1e-15:
            interior = (h > 1e-12) & (h < 1.0 - 1e-12)
            if np.sum(interior) > 0:
                h[interior] += residual / np.sum(interior)
                h = np.clip(h, 0.0, 1.0)
        return h

    def true_score(h):
        """Calculate the true max overlap score."""
        overlaps = get_correlations(h)
        return np.max(overlaps)

    best_h = np.full(N, 0.5)
    best_score = true_score(best_h)  # Should be ~25.0

    rng = np.random.default_rng(42)

    # Generate diverse initial candidates
    candidates = []

    # 1. Uniform baseline
    candidates.append(np.full(N, 0.5))

    # 2. Sinusoidal patterns with various frequencies and phases
    for freq in range(1, 15):
        for phase in [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            x = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
            h = 0.5 + 0.45 * np.sin(x + phase)
            h = project_to_feasible(h)
            candidates.append(h)

    # 3. Triangle wave patterns
    for freq in range(1, 8):
        x = np.linspace(0, 2 * np.pi * freq, N, endpoint=False)
        h = 0.5 + 0.45 * (2 / np.pi) * np.arcsin(np.sin(x))
        h = project_to_feasible(h)
        candidates.append(h)

    # 4. Random perturbations of uniform
    for _ in range(30):
        h = 0.5 + 0.3 * rng.standard_normal(N)
        h = project_to_feasible(h)
        candidates.append(h)

    # 5. Block patterns (alternating blocks)
    for block_size in [2, 3, 4, 5, 7, 10]:
        pattern = np.zeros(N)
        for i in range(N):
            pattern[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        h = 0.5 * pattern + 0.5 * (1 - pattern) * 0.5  # Mix
        h = project_to_feasible(h)
        candidates.append(h)

    # 6. Random smooth patterns
    for _ in range(20):
        # Low-frequency random signal
        coeffs = rng.standard_normal(N // 4 + 1)
        freqs = np.arange(N // 4 + 1)
        signal = np.zeros(N)
        for f in range(1, min(5, N // 4 + 1)):
            signal += coeffs[f] * np.sin(2 * np.pi * f * np.arange(N) / N)
        h = 0.5 + 0.3 * signal / (np.max(np.abs(signal)) + 1e-10)
        h = project_to_feasible(h)
        candidates.append(h)

    # Evaluate all candidates and select top ones for optimization
    scored_candidates = []
    for h in candidates:
        if time.time() > deadline - 20:
            break
        score = true_score(h)
        scored_candidates.append((score, h))

    scored_candidates.sort(key=lambda x: x[0])

    # Optimize from top candidates using L-BFGS-B with increasing beta
    top_candidates = [h for _, h in scored_candidates[:min(10, len(scored_candidates))]]

    for h0 in top_candidates:
        if time.time() > deadline - 15:
            break

        # Multi-scale beta optimization
        current_h = h0.copy()
        for beta in [5.0, 20.0, 50.0, 100.0, 200.0]:
            if time.time() > deadline - 10:
                break

            bounds = [(0, 1)] * N

            def obj(h):
                h_proj = project_to_feasible(h)
                return lse_objective(h_proj, beta)

            try:
                res = minimize(
                    obj,
                    current_h,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current_h = project_to_feasible(res.x)
            except Exception:
                pass

        score = true_score(current_h)
        if score < best_score:
            best_score = score
            best_h = current_h.copy()

    # Final polish: coordinate descent on the best solution
    if time.time() < deadline - 5:
        h = best_h.copy()
        for _ in range(200):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                old_val = h[i]
                # Try a few perturbations
                for delta in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                    new_val = old_val + delta
                    if 0 <= new_val <= 1:
                        h[i] = new_val
                        # Adjust another element to maintain sum
                        # Find element with most room
                        residual = N / 2.0 - np.sum(h)
                        if abs(residual) < 1e-10:
                            continue
                        # Distribute residual to a random interior element
                        j = (i + 1) % N
                        while j == i:
                            j = (j + 1) % N
                        h[j] = h[j] + residual
                        if 0 <= h[j] <= 1:
                            score = true_score(h)
                            if score < best_score:
                                best_score = score
                                best_h = h.copy()
                                improved = True
                        h[j] = h[j] - residual
                h[i] = old_val
            if not improved:
                break

    return best_h

# EVOLVE-BLOCK-END
