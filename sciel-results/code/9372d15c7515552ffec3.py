"""Broadband AR coating: DE over signed continuous 20-D + L-BFGS polish."""
import time
import numpy as np
from scipy.optimize import minimize

def _mean_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm):
    """Exact characteristic-matrix mean reflectance (%) over 401 wavelengths."""
    lam_min, lam_max = band_nm
    wavelengths = np.linspace(lam_min, lam_max, 401)
    n_wl = len(wavelengths)

    if len(labels) == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return float(np.abs(r) ** 2 * 100.0)

    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    beta = 2.0 * np.pi * n_mat * d  # 2*pi*n*d per layer

    total_r = 0.0
    chunk = 50
    for start in range(0, n_wl, chunk):
        end = min(start + chunk, n_wl)
        lams = wavelengths[start:end]
        nl = len(lams)

        M00 = np.ones(nl, dtype=complex)
        M01 = np.zeros(nl, dtype=complex)
        M10 = np.zeros(nl, dtype=complex)
        M11 = np.ones(nl, dtype=complex)

        for i in range(len(labels)):
            delta = beta[i] / lams
            c = np.cos(delta)
            s = np.sin(delta)
            a = c
            b = 1j * s / n_mat[i]
            cc = 1j * n_mat[i] * s
            dd = c

            new_M00 = M00 * a + M01 * cc
            new_M01 = M00 * b + M01 * dd
            new_M10 = M10 * a + M11 * cc
            new_M11 = M10 * b + M11 * dd
            M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11

        Y = M10 / M00
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2 * 100.0
        total_r += np.sum(R)

    return total_r / n_wl


def design(problem):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    centre = 0.5 * (band_nm[0] + band_nm[1])

    # ---------- Fast evaluation for a decoded (labels, thicknesses) ----------
    def _eval(labels, thicknesses):
        return _mean_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, band_nm)

    # ---------- Decode continuous 20-D vector -> (labels, thicknesses) ----------
    # x[i] > 0 => layer i is 'L', x[i] < 0 => 'H'; |x[i]| = thickness (clipped)
    # Layers with |x[i]| < eps are dropped (treated as zero-thickness, i.e., absent)
    EPS = 1.0  # nm threshold for "present"

    def _decode(x):
        x = np.clip(x, -max_layer_nm, max_layer_nm)
        labels = []
        thicknesses = []
        for i in range(len(x)):
            t = abs(x[i])
            if t < EPS:
                continue
            lab = 'L' if x[i] > 0 else 'H'
            labels.append(lab)
            thicknesses.append(float(t))
        # Enforce total thickness constraint by scaling if needed
        if sum(thicknesses) > max_total_nm:
            scale = max_total_nm / sum(thicknesses)
            thicknesses = [t * scale for t in thicknesses]
        # Enforce max layer
        thicknesses = [min(t, max_layer_nm) for t in thicknesses]
        return labels, thicknesses

    def _objective(x):
        labels, thicknesses = _decode(x)
        if len(labels) == 0:
            return 100.0  # bare glass penalty (actually ~4.26 but we want to avoid 0 layers)
        if len(labels) > max_layers:
            return 100.0
        return _eval(labels, thicknesses)

    # ---------- Differential Evolution (manual, lightweight) ----------
    def _differential_evolution(obj, bounds, pop_size=60, max_iter=80, F=0.8, CR=0.9, seed=42):
        rng = np.random.RandomState(seed)
        n = len(bounds)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Initial population
        pop = np.array([lo + rng.rand(n) * (hi - lo) for _ in range(pop_size)])
        fpop = np.array([obj(ind) for ind in pop])

        best_idx = int(np.argmin(fpop))
        best_x = pop[best_idx].copy()
        best_f = fpop[best_idx]

        for it in range(max_iter):
            if time.time() > deadline - 20.0:
                break
            for i in range(pop_size):
                # Select three distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                # Differential variation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                # Crossover
                trial = pop[i].copy()
                jrand = rng.randint(n)
                for j in range(n):
                    if rng.rand() < CR or j == jrand:
                        trial[j] = v[j]
                # Bound
                trial = np.clip(trial, lo, hi)
                ft = obj(trial)
                if ft <= fpop[i]:
                    pop[i] = trial
                    fpop[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
                        best_idx = i

        return best_x, best_f

    # ---------- L-BFGS polish on a fixed label sequence ----------
    def _polish(labels, x0):
        """Polish thicknesses for a fixed label sequence using L-BFGS-B."""
        n = len(labels)
        if n == 0:
            return None, float('inf')

        n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)

        def _obj_thick(th):
            th = np.clip(th, 0.0, max_layer_nm)
            total = np.sum(th)
            if total > max_total_nm:
                penalty = 50.0 * (total - max_total_nm) / max_total_nm
            else:
                penalty = 0.0
            return _eval(labels, list(th)) + penalty

        # Initial point
        th0 = np.array(x0, dtype=float)
        th0 = np.clip(th0, 0.0, max_layer_nm)
        if np.sum(th0) > max_total_nm:
            th0 = th0 * (max_total_nm / np.sum(th0))

        bounds = [(0.0, max_layer_nm)] * n
        res = minimize(_obj_thick, th0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})

        th_opt = np.clip(res.x, 0.0, max_layer_nm)
        if np.sum(th_opt) > max_total_nm:
            th_opt = th_opt * (max_total_nm / np.sum(th_opt))
            th_opt = np.clip(th_opt, 0.0, max_layer_nm)

        score = _eval(labels, list(th_opt))
        if np.all(th_opt >= 0) and np.all(th_opt <= max_layer_nm) and np.sum(th_opt) <= max_total_nm + 1e-9:
            return list(th_opt), score
        return None, float('inf')

    # ---------- Phase 0: Quick initial candidates ----------
    best_labels = None
    best_thicknesses = None
    best_score = float('inf')

    initial_configs = [
        (["L"], [centre / (4.0 * n_L)]),
        (["L", "H"], [centre / (4.0 * n_L), centre / (4.0 * n_H)]),
        (["H", "L"], [centre / (4.0 * n_H), centre / (4.0 * n_L)]),
        (["L", "H", "L"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]),
        (["H", "L", "H"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]),
        (["L", "H", "L", "H"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]),
        (["H", "L", "H", "L"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]),
        (["L", "H", "L", "H", "L"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]),
    ]

    for labels, thicknesses in initial_configs:
        if len(labels) > max_layers:
            continue
        if sum(thicknesses) > max_total_nm:
            continue
        if any(t < 0 or t > max_layer_nm for t in thicknesses):
            continue
        score = _eval(labels, thicknesses)
        if score < best_score:
            best_score = score
            best_labels = labels[:]
            best_thicknesses = thicknesses[:]

    # ---------- Phase 1: Differential Evolution over 20-D signed space ----------
    N_DIM = min(max_layers, 20)
    bounds = [(-max_layer_nm, max_layer_nm)] * N_DIM

    if time.time() < deadline - 40.0:
        de_x, de_f = _differential_evolution(_objective, bounds, pop_size=50, max_iter=60, seed=42)
        de_labels, de_thick = _decode(de_x)
        if de_labels and len(de_labels) <= max_layers and sum(de_thick) <= max_total_nm:
            de_score = _eval(de_labels, de_thick)
            if de_score < best_score:
                best_score = de_score
                best_labels = de_labels[:]
                best_thicknesses = de_thick[:]

    # ---------- Phase 2: L-BFGS polish on best and a few alternative sequences ----------
    polish_sequences = []
    if best_labels:
        polish_sequences.append(best_labels[:])

    # Add some structured alternatives
    for n_layers in range(3, min(12, max_layers + 1)):
        seq_L = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        seq_H = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
        polish_sequences.append(seq_L)
        polish_sequences.append(seq_H)

    # Deduplicate
    seen = set()
    unique_seqs = []
    for seq in polish_sequences:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            unique_seqs.append(seq)

    for labels in unique_seqs:
        if time.time() > deadline - 15.0:
            break
        if len(labels) > max_layers:
            continue

        # Initial thickness guess: quarter-wave at center
        th0 = [centre / (4.0 * (n_L if l == 'L' else n_H)) for l in labels]
        if sum(th0) > max_total_nm:
            scale = max_total_nm / sum(th0)
            th0 = [t * scale for t in th0]

        opt_th, opt_score = _polish(labels, th0)
        if opt_th is not None and opt_score < best_score:
            best_score = opt_score
            best_labels = labels[:]
            best_thicknesses = opt_th[:]

    # ---------- Phase 3: Second DE run with different seed if time permits ----------
    if time.time() < deadline - 30.0:
        de_x2, de_f2 = _differential_evolution(_objective, bounds, pop_size=40, max_iter=40, seed=123)
        de_labels2, de_thick2 = _decode(de_x2)
        if de_labels2 and len(de_labels2) <= max_layers and sum(de_thick2) <= max_total_nm:
            de_score2 = _eval(de_labels2, de_thick2)
            if de_score2 < best_score:
                best_score = de_score2
                best_labels = de_labels2[:]
                best_thicknesses = de_thick2[:]
                # Polish it
                if time.time() < deadline - 10.0:
                    opt_th, opt_score = _polish(best_labels, best_thicknesses)
                    if opt_th is not None and opt_score < best_score:
                        best_score = opt_score
                        best_thicknesses = opt_th[:]

    # ---------- Final constraint enforcement ----------
    if best_labels is None:
        best_labels = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]

    best_thicknesses = [max(0.0, min(t, max_layer_nm)) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]

    return best_labels, best_thicknesses
