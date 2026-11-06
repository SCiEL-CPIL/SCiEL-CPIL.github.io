"""Structural redesign: fast convolution + simulated annealing on exponent structure."""
import time
import math
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    def elapsed_ok():
        return time.time() < deadline

    def square_nonzero_fast(exponents, coefficients):
        """Fast square using numpy convolution. Returns nonzero count."""
        max_deg = max(exponents)
        arr = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exponents, coefficients):
            arr[e] = c
        sq = np.convolve(arr, arr)
        return int(np.count_nonzero(sq))

    # DFT-based spectral surrogate: L1 norm of |P(w_k)^2| for 2^13-th roots of unity
    N_DFT = 8192
    omega = 2.0 * math.pi / N_DFT
    # Precompute roots of unity
    roots = np.exp(1j * omega * np.arange(N_DFT))

    def spectral_surrogate(exponents, coefficients):
        """Compute L1 norm of |P(w_k)|^2 over DFT points.
        This is a smooth surrogate that upper-bounds the time-domain L1 norm."""
        # Evaluate P at all DFT points
        # P(w_k) = sum_j c_j * w_k^{e_j}
        # Use matrix multiplication for speed
        exp_matrix = np.zeros((N_DFT, 12), dtype=np.complex128)
        for j in range(12):
            exp_matrix[:, j] = roots ** exponents[j]
        p_vals = exp_matrix @ np.array(coefficients, dtype=np.complex128)
        # |P(w_k)|^2 = |p_vals|^2
        mag_sq = np.abs(p_vals) ** 2
        return float(np.sum(mag_sq))

    def spectral_gradient(exponents, coefficients):
        """Compute gradient of spectral surrogate w.r.t. coefficients.
        d/dc_j sum_k |P(w_k)|^2 = 2 * Re(sum_k P(w_k) * conj(w_k^{e_j}))
        """
        exp_matrix = np.zeros((N_DFT, 12), dtype=np.complex128)
        for j in range(12):
            exp_matrix[:, j] = roots ** exponents[j]
        p_vals = exp_matrix @ np.array(coefficients, dtype=np.complex128)
        # Gradient: d/dc_j = 2 * Re(sum_k P(w_k) * conj(w_k^{e_j}))
        # = 2 * Re(sum_k p_vals * conj(exp_matrix[:, j]))
        grad = 2.0 * np.real(exp_matrix.T.conj() @ p_vals)
        return grad

    def square_dict(exponents, coefficients):
        """Return dict of exponent->coeff for P^2 (for detailed analysis)."""
        sq = defaultdict(int)
        n = len(exponents)
        for i in range(n):
            ci = coefficients[i]
            ei = exponents[i]
            for j in range(i, n):
                e = ei + exponents[j]
                if i == j:
                    sq[e] += ci * ci
                else:
                    sq[e] += 2 * ci * coefficients[j]
        return {e: c for e, c in sq.items() if c != 0}

    def make_poly(exponents, coefficients):
        max_deg = max(exponents) if exponents else 0
        coeffs = [0] * (max_deg + 1)
        for e, c in zip(exponents, coefficients):
            coeffs[e] += c
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs

    def is_valid(coeffs):
        if len(coeffs) > 41:
            return False
        if not coeffs or coeffs[-1] == 0:
            return False
        nz = sum(1 for c in coeffs if c != 0)
        if nz != 12:
            return False
        for c in coeffs:
            if c < -64 or c > 64:
                return False
        return True

    best_score = 10**9
    best_exponents = None
    best_coefficients = None

    def consider(exponents, coefficients):
        nonlocal best_score, best_exponents, best_coefficients
        if len(exponents) != 12:
            return
        if len(set(exponents)) != 12:
            return
        if max(exponents) > 40:
            return
        for c in coefficients:
            if c == 0 or c < -64 or c > 64:
                return
        nz = square_nonzero_fast(exponents, coefficients)
        if nz < best_score:
            best_score = nz
            best_exponents = list(exponents)
            best_coefficients = list(coefficients)

    # Simulated annealing with structural moves + spectral guidance
    def simulated_annealing(init_exponents, init_coefficients, max_iters=5000, t0=1.0, t_end=0.01):
        exponents = list(init_exponents)
        coefficients = list(init_coefficients)
        current_nz = square_nonzero_fast(exponents, coefficients)
        best_local_nz = current_nz
        best_local_e = list(exponents)
        best_local_c = list(coefficients)

        rng = random.Random(hash(tuple(exponents)) % (2**31))

        for it in range(max_iters):
            if not elapsed_ok():
                break
            t = t0 * (t_end / t0) ** (it / max_iters)

            # Choose a move type
            move_type = rng.random()
            i = rng.randint(0, 11)

            if move_type < 0.4:
                # Exponent move: pick new exponent
                delta = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, -8, 8, -12, 12])
                new_e = exponents[i] + delta
                if new_e < 0 or new_e > 40:
                    continue
                others = set(exponents[:i] + exponents[i+1:])
                if new_e in others:
                    continue
                exponents[i] = new_e
                new_nz = square_nonzero_fast(exponents, coefficients)
                delta_e = new_nz - current_nz
                if delta_e < 0 or rng.random() < math.exp(-delta_e / max(t, 1e-10)):
                    current_nz = new_nz
                    if new_nz < best_local_nz:
                        best_local_nz = new_nz
                        best_local_e = list(exponents)
                        best_local_c = list(coefficients)
                else:
                    exponents[i] -= delta
            elif move_type < 0.75:
                # Coefficient move with spectral guidance
                old_c = coefficients[i]
                # Use spectral gradient to bias coefficient choices
                grad = spectral_gradient(exponents, coefficients)
                # Choose new coefficient: bias towards reducing spectral norm
                if grad[i] > 0:
                    # Positive gradient: decrease coefficient
                    new_c = rng.choice([-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64])
                    if new_c >= old_c:
                        new_c = -abs(new_c) if rng.random() < 0.5 else min(new_c, 64)
                elif grad[i] < 0:
                    # Negative gradient: increase coefficient
                    new_c = rng.choice([-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64])
                    if new_c <= old_c:
                        new_c = abs(new_c) if rng.random() < 0.5 else max(new_c, -64)
                else:
                    new_c = rng.choice([-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64])
                
                if new_c == old_c or new_c == 0:
                    continue
                coefficients[i] = new_c
                new_nz = square_nonzero_fast(exponents, coefficients)
                delta_c = new_nz - current_nz
                if delta_c < 0 or rng.random() < math.exp(-delta_c / max(t, 1e-10)):
                    current_nz = new_nz
                    if new_nz < best_local_nz:
                        best_local_nz = new_nz
                        best_local_e = list(exponents)
                        best_local_c = list(coefficients)
                else:
                    coefficients[i] = old_c
            elif move_type < 0.85:
                # Swap two exponents (structural)
                j = rng.randint(0, 11)
                if j == i:
                    continue
                exponents[i], exponents[j] = exponents[j], exponents[i]
                new_nz = square_nonzero_fast(exponents, coefficients)
                delta_s = new_nz - current_nz
                if delta_s < 0 or rng.random() < math.exp(-delta_s / max(t, 1e-10)):
                    current_nz = new_nz
                    if new_nz < best_local_nz:
                        best_local_nz = new_nz
                        best_local_e = list(exponents)
                        best_local_c = list(coefficients)
                else:
                    exponents[i], exponents[j] = exponents[j], exponents[i]
            else:
                # Spectral gradient descent step on coefficients (continuous relaxation)
                grad = spectral_gradient(exponents, coefficients)
                lr = 0.5 * t
                new_coeffs = []
                for j in range(12):
                    c_new = coefficients[j] - lr * grad[j]
                    # Round to nearest integer and clamp
                    c_rounded = int(round(c_new))
                    c_clamped = max(-64, min(64, c_rounded))
                    if c_clamped == 0:
                        c_clamped = 1 if c_new >= 0 else -1
                    new_coeffs.append(c_clamped)
                new_nz = square_nonzero_fast(exponents, new_coeffs)
                delta_g = new_nz - current_nz
                if delta_g < 0 or rng.random() < math.exp(-delta_g / max(t, 1e-10)):
                    coefficients = new_coeffs
                    current_nz = new_nz
                    if new_nz < best_local_nz:
                        best_local_nz = new_nz
                        best_local_e = list(exponents)
                        best_local_c = list(coefficients)

        return best_local_e, best_local_c, best_local_nz

    # Final polish: local descent on coefficients with exact evaluation
    def polish_coefficients(exponents, max_iters=2000):
        """Refine coefficients using local descent with exact square evaluation."""
        # Initialize with small integers
        coefficients = [1 if i % 2 == 0 else -1 for i in range(12)]
        current_nz = square_nonzero_fast(exponents, coefficients)
        best_nz = current_nz
        best_c = list(coefficients)
        no_improve = 0
        rng = random.Random(hash(tuple(exponents)) % (2**31))

        for it in range(max_iters):
            if not elapsed_ok():
                break
            if no_improve > 200:
                break

            i = rng.randint(0, 11)
            old_c = coefficients[i]
            # Try neighboring values
            neighbors = [c for c in range(max(-64, old_c - 8), min(65, old_c + 9)) if c != 0 and c != old_c]
            if not neighbors:
                neighbors = [1, -1, 2, -2]
            new_c = rng.choice(neighbors)
            coefficients[i] = new_c
            new_nz = square_nonzero_fast(exponents, coefficients)

            if new_nz < current_nz:
                current_nz = new_nz
                no_improve = 0
                if new_nz < best_nz:
                    best_nz = new_nz
                    best_c = list(coefficients)
            else:
                coefficients[i] = old_c
                no_improve += 1

        return best_c, best_nz

    # Generate diverse initial candidates
    candidates = []

    # Arithmetic progression bases with various signs
    for base in range(0, 25):
        for step in [1, 2, 3, 4, 5]:
            exps = [base + i * step for i in range(12)]
            if max(exps) > 40:
                continue
            for mask in range(64):
                signs = [1 if (mask >> (i % 6)) & 1 else -1 for i in range(12)]
                candidates.append((exps, signs))

    # Two-group structure: 6 terms in group A, 6 in group B, far apart
    for gap in range(20, 41):
        for step_a in [1, 2, 3]:
            for step_b in [1, 2, 3]:
                exps_a = [i * step_a for i in range(6)]
                exps_b = [gap + i * step_b for i in range(6)]
                exps = sorted(exps_a + exps_b)
                if len(set(exps)) != 12 or max(exps) > 40:
                    continue
                for mask in range(16):
                    signs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
                    candidates.append((exps, signs))

    # Geometric-like: exponents from small set of base positions
    for base_set in [[0,1,2,3,4,5], [0,1,2,4,5,7], [0,2,4,6,8,10], [0,1,3,5,7,9], [0,2,4,6,8,12]]:
        for mult in [1, 2, 3]:
            exps = [e * mult for e in base_set]
            if max(exps) > 40 or len(set(exps)) != 6:
                continue
            # Duplicate with offset to get 12 terms
            offset = max(exps) + 1
            exps2 = exps + [e + offset for e in base_set[:6]]
            exps2 = sorted(exps2)
            if len(set(exps2)) != 12 or max(exps2) > 40:
                continue
            for mask in range(32):
                signs = [1 if (mask >> (i % 5)) & 1 else -1 for i in range(12)]
                candidates.append((exps2, signs))

    # Random
    rng = random.Random(42)
    for _ in range(2000):
        exps = sorted(rng.sample(range(0, 41), 12))
        coeffs = [rng.choice([-2, -1, 1, 2]) for _ in range(12)]
        candidates.append((exps, coeffs))

    # Evaluate candidates quickly, sort by score
    scored = []
    for exps, coeffs in candidates:
        if not elapsed_ok():
            break
        if len(exps) != 12 or len(set(exps)) != 12:
            continue
        if max(exps) > 40:
            continue
        if any(c == 0 or c < -64 or c > 64 for c in coeffs):
            continue
        nz = square_nonzero_fast(exps, coeffs)
        scored.append((nz, exps, coeffs))

    scored.sort(key=lambda x: x[0])

    # Run SA on top candidates
    for nz, exps, coeffs in scored[:150]:
        if not elapsed_ok():
            break
        new_e, new_c, new_nz = simulated_annealing(exps, coeffs, max_iters=3000)
        consider(new_e, new_c)

    # More SA iterations on the current best
    if best_exponents is not None and elapsed_ok():
        for _ in range(5):
            if not elapsed_ok():
                break
            new_e, new_c, new_nz = simulated_annealing(best_exponents, best_coefficients, max_iters=8000)
            consider(new_e, new_c)

    # Final random restarts
    if elapsed_ok():
        for _ in range(300):
            if not elapsed_ok():
                break
            exps = sorted(rng.sample(range(0, 41), 12))
            coeffs = [rng.choice([-2, -1, 1, 2]) for _ in range(12)]
            new_e, new_c, new_nz = simulated_annealing(exps, coeffs, max_iters=1500)
            consider(new_e, new_c)

    # Polish pass on the best exponent structure found
    if best_exponents is not None and elapsed_ok():
        polished_c, polished_nz = polish_coefficients(best_exponents, max_iters=5000)
        consider(best_exponents, polished_c)

    # Build final polynomial
    if best_exponents is None:
        return [1] * 12
    poly = make_poly(best_exponents, best_coefficients)
    if is_valid(poly):
        return poly
    return [1] * 12
# EVOLVE-BLOCK-END
