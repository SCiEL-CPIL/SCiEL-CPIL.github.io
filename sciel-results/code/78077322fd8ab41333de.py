import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s margin for safety

    N = 24
    k = np.arange(N)

    def sup_norm(c, n_samples=4096, n_refine=8, refine_iters=25):
        """Compute sup norm of P(e^{iθ}) where P(z) = sum c_k z^k"""
        # Fast evaluation using FFT
        # P(e^{iθ}) = sum c_k e^{ikθ}
        # At θ_j = 2πj/n_samples, this is DFT of c (padded)
        n_fft = 1 << (n_samples - 1).bit_length()  # next power of 2
        c_padded = np.zeros(n_fft, dtype=complex)
        c_padded[:N] = c.astype(complex)
        # DFT: X_j = sum c_k e^{-2πijk/n}, but we want e^{+ikθ}
        # So we need inverse DFT convention or conjugate
        vals = np.fft.ifft(c_padded) * n_fft  # gives sum c_k e^{2πijk/n}
        abs_vals = np.abs(vals[:n_samples])

        # Find top peaks
        idxs = np.argsort(abs_vals)[-n_refine:]

        refined_max = np.max(abs_vals)
        dtheta = 2 * np.pi / n_samples

        for idx in idxs:
            theta0 = idx * dtheta
            a = theta0 - dtheta
            b = theta0 + dtheta

            # Ternary search for max of |P(e^{iθ})|
            for _ in range(refine_iters):
                m1 = a + (b - a) / 3
                m2 = b - (b - a) / 3
                f1 = np.abs(np.sum(c * np.exp(1j * m1 * k)))
                f2 = np.abs(np.sum(c * np.exp(1j * m2 * k)))
                if f1 < f2:
                    a = m1
                else:
                    b = m2
            mid = (a + b) / 2
            f_mid = np.abs(np.sum(c * np.exp(1j * mid * k)))
            if f_mid > refined_max:
                refined_max = f_mid

        return refined_max

    # Generate diverse starting candidates
    def golay_rs():
        p, q = [1], [1]
        while len(p) < N:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:N], dtype=float)

    def thue_morse():
        # Thue-Morse: t(n) = popcount(n) mod 2, mapped to ±1
        tm = [1 - 2 * (bin(i).count('1') % 2) for i in range(N)]
        return np.array(tm, dtype=float)

    def random_seed():
        return np.array([random.choice([-1.0, 1.0]) for _ in range(N)], dtype=float)

    def block_pattern():
        # Alternating blocks of 3
        pat = []
        sign = 1.0
        for i in range(N):
            block = i // 3
            if block % 2 == 0:
                pat.append(sign)
            else:
                pat.append(-sign)
        return np.array(pat, dtype=float)

    seeds = [golay_rs(), thue_morse(), random_seed(), block_pattern(), random_seed()]

    best_coeffs = seeds[0].copy()
    best_norm = sup_norm(best_coeffs)

    # Simulated annealing
    T_start = 0.5
    T_end = 0.001
    cooling_rate = 0.995
    T = T_start

    while time.time() < deadline:
        improved = False

        # Try single-bit flips
        for i in range(N):
            new_coeffs = best_coeffs.copy()
            new_coeffs[i] *= -1
            new_norm = sup_norm(new_coeffs)
            delta = new_norm - best_norm

            if delta < -1e-9:
                best_coeffs = new_coeffs
                best_norm = new_norm
                improved = True
            elif random.random() < np.exp(-delta / T):
                best_coeffs = new_coeffs
                best_norm = new_norm
                improved = True

        # Try segment flips (length 2-4)
        for seg_len in [2, 3, 4]:
            for i in range(N - seg_len + 1):
                new_coeffs = best_coeffs.copy()
                new_coeffs[i:i+seg_len] *= -1
                new_norm = sup_norm(new_coeffs)
                delta = new_norm - best_norm

                if delta < -1e-9:
                    best_coeffs = new_coeffs
                    best_norm = new_norm
                    improved = True
                elif random.random() < np.exp(-delta / T):
                    best_coeffs = new_coeffs
                    best_norm = new_norm
                    improved = True

        T *= cooling_rate
        if T < T_end:
            T = T_end

    return [int(c) for c in best_coeffs]
