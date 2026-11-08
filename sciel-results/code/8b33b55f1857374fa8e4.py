import numpy as np
import time

N = 100

def construct():
    deadline = time.time() + 140

    def max_overlap(h):
        H = np.fft.fft(h)
        autocorr = np.real(np.fft.ifft(H * np.conj(H)))
        return N / 2.0 - np.min(autocorr)

    def project(h):
        h = np.clip(h, 0.0, 1.0)
        s = np.sum(h)
        diff = N / 2.0 - s
        if abs(diff) < 1e-12:
            return h
        if diff > 0:
            room = 1.0 - h
            order = np.argsort(-room)
            remaining = diff
            for i in order:
                add = min(remaining, room[i])
                h[i] += add
                remaining -= add
                if remaining <= 1e-12:
                    break
        else:
            order = np.argsort(h)
            remaining = -diff
            for i in order:
                sub = min(remaining, h[i])
                h[i] -= sub
                remaining -= sub
                if remaining <= 1e-12:
                    break
        return h

    h = np.full(N, 0.5)
    best_h = h.copy()
    best_val = max_overlap(h)

    np.random.seed(123)
    step_size = 0.15
    no_improve = 0
    max_no_improve = 300

    while time.time() < deadline:
        delta = np.random.randn(N) * step_size
        new_h = project(h + delta)
        val = max_overlap(new_h)

        if val < best_val - 1e-9:
            best_h = new_h.copy()
            best_val = val
            h = new_h.copy()
            no_improve = 0
            step_size = min(step_size * 1.02, 0.5)
        else:
            no_improve += 1
            step_size = max(step_size * 0.98, 0.005)
            if no_improve >= max_no_improve:
                h = best_h.copy()
                step_size = 0.2
                no_improve = 0

    return best_h
