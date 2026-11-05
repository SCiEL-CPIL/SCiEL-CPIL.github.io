"""Optimized step-height search for third autocorrelation inequality."""
import numpy as np
import time

N = 400


def _value(h):
    s = h.sum()
    if s == 0:
        return float("inf")
    c = np.convolve(h, h)
    return abs(2.0 * N * c.max() / (s * s))


def construct():
    t0 = time.time()
    deadline = t0 + 120.0
    rng = np.random.default_rng(2024)

    # --- Phase 1: quick analytical exploration (fast, no convolve) ---
    # For binary ±1 sequences: sum(h) can be 0 (invalid) or small.
    # Try structured sequences: quadratic residues, Thue-Morse, etc.
    candidates = []

    # Quadratic residue sequence mod p for several primes near 400
    for p in (397, 401, 409, 389, 383, 419, 379, 431):
        # Legendre symbol: (a|p) = a^((p-1)/2) mod p; 1 if QR, -1 if non-QR, 0 if p|a
        seq = np.array([int(pow(int(i % p), (p - 1) // 2, p)) for i in range(p)])
        # Convert: 1 -> +1, (p-1) -> -1, 0 -> +1
        seq = np.where(seq == p - 1, -1, 1).astype(float)
        # Take first N, also try shifted versions
        for shift in range(0, min(p, 20)):
            h = np.roll(seq, -shift)[:N] if p >= N else np.concatenate([seq, seq])[:N]
            s = h.sum()
            if abs(s) > 1e-6:
                candidates.append(h)

    # Thue-Morse sequence
    tm = np.array([bin(i).count("1") & 1 for i in range(2048)], dtype=float)
    tm = 1 - 2 * tm  # +1 for even parity, -1 for odd
    for shift in range(0, 40):
        h = np.roll(tm, -shift)[:N]
        s = h.sum()
        if abs(s) > 1e-6:
            candidates.append(h)

    # Barker-like: try ±1 with balanced sum
    for trial in range(200):
        h = rng.choice([-1.0, 1.0], size=N)
        if abs(h.sum()) > 5.0:  # avoid near-zero sum
            candidates.append(h)

    # Evaluate all candidates
    best_h = None
    best_val = float("inf")
    for h in candidates:
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    if best_h is None:
        best_h = np.ones(N)
        best_val = 2.0

    # --- Phase 2: local search from best candidate ---
    if best_h is not None and best_val < float("inf"):
        current = best_h.copy()
        current_val = best_val
        step_size = 1.0
        stall = 0
        max_stall = 300

        while time.time() < deadline - 5.0:
            # Random coordinate perturbation
            idx = rng.integers(0, N)
            old_val = current[idx]
            # Mix of small and large perturbations
            if rng.random() < 0.3:
                current[idx] = -old_val  # sign flip
            else:
                current[idx] = old_val + rng.normal(0, step_size * max(1.0, abs(old_val)))

            new_val = _value(current)

            if new_val < current_val:
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = current.copy()
                stall = 0
                step_size = max(0.1, step_size * 1.05)  # grow step on success
            else:
                current[idx] = old_val
                stall += 1
                if stall % 50 == 0:
                    step_size = max(0.1, step_size * 0.9)  # shrink on stall

                if stall > max_stall:
                    # Restart from best with fresh noise
                    current = best_h + rng.normal(0, 0.5, size=N)
                    current_val = _value(current)
                    if not np.isfinite(current_val):
                        current = best_h.copy()
                        current_val = best_val
                    stall = 0
                    max_stall = int(max_stall * 1.5)

    # --- Phase 3: fine-tune sign flips on best_h ---
    if time.time() < deadline - 3.0:
        h = best_h.copy()
        val = best_val
        for _ in range(500):
            if time.time() > deadline - 3.0:
                break
            # Flip a random subset of 1-5 indices
            k = rng.integers(1, 6)
            idxs = rng.choice(N, size=k, replace=False)
            old = h[idxs].copy()
            h[idxs] = -h[idxs]
            v = _value(h)
            if v < val:
                val = v
                best_val = val
                best_h = h.copy()
            else:
                h[idxs] = old

    # --- Final safety: ensure validity ---
    if not np.all(np.isfinite(best_h)) or abs(best_h.sum()) < 1e-10:
        return np.ones(N)

    return best_h
