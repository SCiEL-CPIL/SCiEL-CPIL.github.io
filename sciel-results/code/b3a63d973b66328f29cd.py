"""Broadband antireflection coating design via simulated annealing + per-layer polish."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s wall-clock kill


def _mean_reflectance(materials, thicknesses, n_amb, n_sub, band, n_wl=401):
    """Exact characteristic-matrix mean reflectance (fraction)."""
    nL = materials["L"]
    nH = materials["H"]
    n_layers = len(thicknesses)
    if n_layers == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return r * r
    wl = np.linspace(band[0], band[1], n_wl)
    k = 2.0 * np.pi / wl  # rad / nm
    # Build per-wavelength stack matrices
    # Precompute per-layer index arrays
    idx_arr = np.empty(n_layers, dtype=float)
    for i, m in enumerate(materials):
        idx_arr[i] = nL if m == "L" else nH
    th = np.asarray(thicknesses, dtype=float)
    # For each wavelength, compute product of matrices
    # Use vectorized complex arithmetic
    # M_i = [[cos delta_i, i sin delta_i / n_i], [i n_i sin delta_i, cos delta_i]]
    # We'll iterate layers for each wavelength (n_layers * n_wl small)
    # Represent matrix as 4 complex numbers: a,b,c,d
    # Start with identity
    A = np.ones(n_wl, dtype=complex)
    B = np.zeros(n_wl, dtype=complex)
    C = np.zeros(n_wl, dtype=complex)
    D = np.ones(n_wl, dtype=complex)
    for i in range(n_layers):
        n_i = idx_arr[i]
        delta = k * n_i * th[i]  # shape (n_wl,)
        c = np.cos(delta)
        s = np.sin(delta)
        # M_i
        m_a = c
        m_b = 1j * s / n_i
        m_c = 1j * n_i * s
        m_d = c
        # New = M_i @ (A,B;C,D)
        nA = m_a * A + m_b * C
        nB = m_a * B + m_b * D
        nC = m_c * A + m_d * C
        nD = m_c * B + m_d * D
        A, B, C, D = nA, nB, nC, nD
    # Y = (C + n_sub D) / (A + n_sub B)
    Y = (C + n_sub * D) / (A + n_sub * B)
    r = (n_amb - Y) / (n_amb + Y)
    R = np.abs(r) ** 2
    return float(np.mean(R))


def _score(materials, thicknesses, problem):
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    band = problem["band_nm"]
    return _mean_reflectance(materials, thicknesses, n_amb, n_sub, band) * 100.0


def _polish(materials, thicknesses, problem, n_amb, n_sub, band, max_iter=200):
    """Coordinate-descent / Nelder-Mead style polish on thicknesses."""
    best_th = np.array(thicknesses, dtype=float)
    best_s = _score(materials, best_th, problem)
    max_layer = problem.get("max_layer_nm", 400.0)
    max_total = problem.get("max_total_nm", 1000.0)
    n = len(best_th)
    # Simple gradient-free coordinate descent with adaptive step
    step = 10.0
    for it in range(max_iter):
        if time.time() > _DEADLINE - 2.0:
            break
        improved = False
        for i in range(n):
            for direction in (1.0, -1.0):
                trial = best_th.copy()
                trial[i] += direction * step
                if trial[i] < 0 or trial[i] > max_layer:
                    continue
                if np.sum(trial) > max_total:
                    continue
                s = _score(materials, trial, problem)
                if s < best_s - 1e-9:
                    best_th = trial
                    best_s = s
                    improved = True
        if not improved:
            step *= 0.7
            if step < 0.01:
                break
    return best_th, best_s


def _random_sequence(n_layers, rng):
    """Generate a random alternating-ish sequence starting and ending with L for AR."""
    seq = []
    for i in range(n_layers):
        # Bias toward alternating, but allow some randomness
        if i == 0:
            seq.append("L")
        elif i == n_layers - 1:
            seq.append("L")
        else:
            # Alternate with some probability
            if rng.random() < 0.85:
                seq.append("H" if seq[-1] == "L" else "L")
            else:
                seq.append("L" if rng.random() < 0.5 else "H")
    return seq


def _initial_thicknesses(seq, problem, rng):
    """Initialize with quarter-wave at center wavelength."""
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    th = []
    for m in seq:
        n = nL if m == "L" else nH
        t = centre / (4.0 * n)
        # Add small random perturbation
        t *= (1.0 + 0.1 * (rng.random() - 0.5))
        t = max(0.0, min(t, problem.get("max_layer_nm", 400.0)))
        th.append(t)
    return th


def design(problem):
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    band = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total = problem.get("max_total_nm", 1000.0)
    max_layer = problem.get("max_layer_nm", 400.0)

    rng = random.Random(42)
    np_rng = np.random.RandomState(42)

    # Start with a known good design: quarter-wave L at center
    centre = 0.5 * (band[0] + band[1])
    best_seq = ["L"]
    best_th = [centre / (4.0 * problem["materials"]["L"])]
    best_score = _score(best_seq, best_th, problem)

    # Also try a few hand-crafted starting points
    candidates = []
    # Quarter-wave stack L-H-L
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    for n_layers in [3, 5, 7, 9]:
        seq = []
        for i in range(n_layers):
            seq.append("L" if i % 2 == 0 else "H")
        if seq[-1] != "L":
            seq[-1] = "L"
        th = []
        for m in seq:
            n = nL if m == "L" else nH
            th.append(centre / (4.0 * n))
        if sum(th) <= max_total:
            candidates.append((seq, th))

    for seq, th in candidates:
        s = _score(seq, th, problem)
        if s < best_score:
            best_score = s
            best_seq = seq[:]
            best_th = th[:]

    # Simulated annealing over sequences and thicknesses
    T = 2.0  # temperature in percent
    T_min = 0.001
    cooling = 0.995

    current_seq = best_seq[:]
    current_th = np.array(best_th, dtype=float)
    current_score = best_score

    n_moves = 0
    while time.time() < _DEADLINE - 10.0 and T > T_min:
        n_moves += 1

        # Choose a move
        move_type = rng.random()
        if move_type < 0.3 and len(current_seq) > 1:
            # Remove a layer (not first or last)
            idx = rng.randint(1, len(current_seq) - 2)
            trial_seq = current_seq[:idx] + current_seq[idx+1:]
            trial_th = np.delete(current_th, idx)
        elif move_type < 0.5 and len(current_seq) < max_layers:
            # Add a layer
            idx = rng.randint(1, len(current_seq))
            m = "H" if current_seq[idx-1] == "L" else "L"
            # Sometimes random
            if rng.random() < 0.2:
                m = "L" if rng.random() < 0.5 else "H"
            trial_seq = current_seq[:idx] + [m] + current_seq[idx:]
            n = nL if m == "L" else nH
            t = centre / (4.0 * n) * (1.0 + 0.2 * (rng.random() - 0.5))
            t = max(0.0, min(t, max_layer))
            trial_th = np.insert(current_th, idx, t)
        elif move_type < 0.7:
            # Swap a layer's index
            idx = rng.randint(0, len(current_seq) - 1)
            trial_seq = current_seq[:]
            trial_seq[idx] = "H" if trial_seq[idx] == "L" else "L"
            trial_th = current_th.copy()
        else:
            # Perturb thicknesses
            trial_seq = current_seq[:]
            trial_th = current_th.copy()
            idx = rng.randint(0, len(trial_th) - 1)
            trial_th[idx] += (rng.random() - 0.5) * 20.0
            trial_th[idx] = max(0.0, min(trial_th[idx], max_layer))

        # Check constraints
        if len(trial_seq) != len(trial_th):
            continue
        if np.any(trial_th < 0) or np.any(trial_th > max_layer):
            continue
        if np.sum(trial_th) > max_total:
            continue
        if len(trial_seq) > max_layers:
            continue

        trial_score = _score(trial_seq, trial_th.tolist(), problem)

        # Acceptance
        if trial_score < current_score:
            current_seq = trial_seq
            current_th = trial_th
            current_score = trial_score
            if current_score < best_score:
                best_score = current_score
                best_seq = current_seq[:]
                best_th = current_th.copy()
        else:
            delta = trial_score - current_score
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-6)):
                current_seq = trial_seq
                current_th = trial_th
                current_score = trial_score

        T *= cooling

        # Periodic polish
        if n_moves % 200 == 0 and time.time() < _DEADLINE - 5.0:
            polished_th, polished_s = _polish(
                current_seq, current_th.tolist(), problem,
                n_amb, n_sub, band, max_iter=50
            )
            if polished_s < current_score:
                current_th = polished_th
                current_score = polished_s
                if current_score < best_score:
                    best_score = current_score
                    best_seq = current_seq[:]
                    best_th = polished_th.copy()

    # Final polish on the best design
    if time.time() < _DEADLINE - 3.0:
        polished_th, polished_s = _polish(
            best_seq, best_th.tolist(), problem,
            n_amb, n_sub, band, max_iter=300
        )
        if polished_s < best_score:
            best_score = polished_s
            best_th = polished_th

    # Ensure constraints are met
    best_th = np.clip(best_th, 0.0, max_layer)
    if np.sum(best_th) > max_total:
        scale = max_total / np.sum(best_th)
        best_th *= scale

    # Return as lists
    return best_seq, [float(t) for t in best_th]
# EVOLVE-BLOCK-END
