import numpy as np
import time

def _autoconv(f):
    """Compute autoconvolution (convolution f*f) using FFT.
    Returns array of length 2n-1."""
    n = len(f)
    F_full = np.fft.fft(f, n=2*n)
    conv_full = np.fft.ifft(F_full * F_full)[:2*n-1].real
    return conv_full

def _compute_score(f):
    """Compute the normalized max of autoconvolution."""
    n = len(f)
    f = np.asarray(f, dtype=np.float64)
    if np.all(f == 0):
        return 0.0
    conv = _autoconv(f)
    max_conv = np.max(conv)
    sum_f = np.sum(f)
    norm = sum_f * sum_f
    if norm == 0:
        return 0.0
    return max_conv / norm

def construct():
    n = 600
    start_time = time.time()
    deadline = start_time + 145.0

    k = np.arange(n)
    amp = np.exp(-k**2 / (n/4)**2)
    amp[0] = 1.0
    phase = np.zeros(n)

    def amp_to_f(amp, phase):
        F = amp * np.exp(1j * phase)
        f_raw = np.fft.ifft(F).real
        f = np.maximum(f_raw, 0.0)
        return f

    def compute_from_amp(amp, phase):
        f = amp_to_f(amp, phase)
        score = _compute_score(f)
        return f, score

    f, best_score = compute_from_amp(amp, phase)
    best_amp = amp.copy()
    best_phase = phase.copy()

    rng = np.random.default_rng(42)

    # Soft-max parameter for smooth approximation of max
    beta = 50.0

    def smooth_max(x, beta=50.0):
        """Soft approximation of max using log-sum-exp."""
        m = np.max(x)
        if m < -1e10:
            return 0.0
        return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

    def smooth_score(amp, phase, beta=50.0):
        """Smooth surrogate for the normalized max autoconvolution."""
        f = amp_to_f(amp, phase)
        if np.all(f == 0):
            return 0.0
        conv = _autoconv(f)
        sum_f = np.sum(f)
        norm = sum_f * sum_f
        if norm == 0:
            return 0.0
        return smooth_max(conv, beta) / norm

    # Vectorized coordinate descent: perturb ALL coordinates at once
    def batch_perturb(best_amp, best_phase, step_sizes, beta):
        """Try perturbing all amplitude coordinates simultaneously."""
        improved = False
        for delta in step_sizes:
            test_amp = best_amp + delta
            test_amp = np.maximum(test_amp, 0.0)
            f_test, score_test = compute_from_amp(test_amp, best_phase)
            if score_test < best_score:
                best_amp = test_amp.copy()
                best_score = score_test
                improved = True
        return best_amp, best_score, improved

    # Main optimization loop with adaptive step sizes
    amp_step = 0.1
    phase_step = 0.1
    no_improve_count = 0
    max_no_improve = 30

    while time.time() < deadline - 10.0:
        improved = False

        # Amplitude coordinate descent (vectorized)
        for delta in [amp_step, -amp_step, amp_step * 0.5, -amp_step * 0.5]:
            if time.time() > deadline - 10.0:
                break
            test_amp = best_amp + delta
            test_amp = np.maximum(test_amp, 0.0)
            f_test, score_test = compute_from_amp(test_amp, best_phase)
            if score_test < best_score - 1e-12:
                best_amp = test_amp
                best_score = score_test
                improved = True
                no_improve_count = 0

        # Phase coordinate descent (vectorized)
        for delta in [phase_step, -phase_step, phase_step * 0.5, -phase_step * 0.5]:
            if time.time() > deadline - 10.0:
                break
            test_phase = best_phase + delta
            f_test, score_test = compute_from_amp(best_amp, test_phase)
            if score_test < best_score - 1e-12:
                best_phase = test_phase
                best_score = score_test
                improved = True
                no_improve_count = 0

        if improved:
            # Adaptive step: shrink if improving
            amp_step *= 0.95
            phase_step *= 0.95
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Try random restarts
                for _ in range(20):
                    if time.time() > deadline - 10.0:
                        break
                    noise_scale = 0.05
                    test_amp = best_amp + rng.normal(0, noise_scale, n)
                    test_amp = np.maximum(test_amp, 0)
                    test_phase = best_phase + rng.normal(0, noise_scale, n)
                    f_test, score_test = compute_from_amp(test_amp, test_phase)
                    if score_test < best_score:
                        best_amp = test_amp
                        best_phase = test_phase
                        best_score = score_test
                        improved = True
                        no_improve_count = 0
                        amp_step = 0.1
                        phase_step = 0.1
                if not improved:
                    break
            # Shrink step more aggressively when stuck
            amp_step *= 0.7
            phase_step *= 0.7
            amp_step = max(amp_step, 1e-4)
            phase_step = max(phase_step, 1e-4)

    # Final polish: use smooth surrogate for more precise local optimization
    if time.time() < deadline - 5.0:
        beta = 100.0
        polish_amp = best_amp.copy()
        polish_phase = best_phase.copy()
        polish_score = best_score
        polish_step = 1e-3

        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            improved_polish = False
            for delta in [polish_step, -polish_step]:
                test_amp = np.maximum(polish_amp + delta, 0.0)
                f_test, score_test = compute_from_amp(test_amp, polish_phase)
                if score_test < polish_score - 1e-14:
                    polish_amp = test_amp
                    polish_score = score_test
                    improved_polish = True
            for delta in [polish_step, -polish_step]:
                test_phase = polish_phase + delta
                f_test, score_test = compute_from_amp(polish_amp, test_phase)
                if score_test < polish_score - 1e-14:
                    polish_phase = test_phase
                    polish_score = score_test
                    improved_polish = True
            if not improved_polish:
                polish_step *= 0.5
                if polish_step < 1e-6:
                    break

        if polish_score < best_score:
            best_amp = polish_amp
            best_phase = polish_phase
            best_score = polish_score

    f = amp_to_f(best_amp, best_phase)
    return list(f)
