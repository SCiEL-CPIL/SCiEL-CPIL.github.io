"""Broadband antireflection coating design using characteristic-matrix method."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers):
    """Compute reflectance (0-1) for a stack using the characteristic matrix method."""
    n = len(n_layers)
    if n == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wavelengths), np.abs(r) ** 2)

    M11 = np.ones(len(wavelengths), dtype=complex)
    M12 = np.zeros(len(wavelengths), dtype=complex)
    M21 = np.zeros(len(wavelengths), dtype=complex)
    M22 = np.ones(len(wavelengths), dtype=complex)

    for i in range(n):
        ni = n_layers[i]
        di = d_layers[i]
        if di < 1e-10:
            continue
        delta = 2.0 * np.pi * ni * di / wavelengths
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        L11 = cos_d
        L12 = 1j * sin_d / ni
        L21 = 1j * ni * sin_d
        L22 = cos_d
        new_M11 = M11 * L11 + M12 * L21
        new_M12 = M11 * L12 + M12 * L22
        new_M21 = M21 * L11 + M22 * L21
        new_M22 = M21 * L12 + M22 * L22
        M11 = new_M11; M12 = new_M12; M21 = new_M21; M22 = new_M22

    Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    r = (n_ambient - Y) / (n_ambient + Y)
    return np.abs(r) ** 2


def _mean_reflectance_percent(wavelengths, n_ambient, n_substrate, n_layers, d_layers):
    R = _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers)
    return float(np.mean(R) * 100.0)


def _is_valid(materials, thicknesses, problem):
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if len(materials) == 0:
        return True
    total = sum(thicknesses)
    if total > max_total_nm + 1e-9:
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False
    for m in materials:
        if m not in problem["materials"]:
            return False
    return True


def _evaluate_design(materials, thicknesses, problem, wavelengths):
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_layers = np.array([problem["materials"][m] for m in materials], dtype=float)
    d_layers = np.array(thicknesses, dtype=float)
    return _mean_reflectance_percent(wavelengths, n_ambient, n_substrate, n_layers, d_layers)


def _nelder_mead(mats, x0, problem, wavelengths, deadline, max_iter=3000):
    """Nelder-Mead optimization with constraint handling."""
    n = len(mats)
    if n == 0:
        return list(x0), _evaluate_design(mats, list(x0), problem, wavelengths)
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    n_layers = np.array([problem["materials"][m] for m in mats], dtype=float)
    d_layers = np.zeros(n, dtype=float)

    def objective(x):
        x = np.clip(x, 0.0, max_layer_nm)
        penalty = 0.0
        total = x.sum()
        if total > max_total_nm:
            penalty += (total - max_total_nm) * 0.1
        d_layers[:] = x
        R = _char_matrix_method(wavelengths, problem["ambient_index"], problem["substrate_index"], n_layers, d_layers)
        return float(np.mean(R) * 100.0) + penalty

    x0 = np.clip(np.asarray(x0, dtype=float), 0.0, max_layer_nm)
    step = 20.0
    simplex = [x0.copy() for _ in range(n + 1)]
    for i in range(n):
        simplex[i + 1] = x0.copy()
        simplex[i + 1][i] = min(x0[i] + step, max_layer_nm)
    fvals = [objective(s) for s in simplex]

    alpha = 1.0; gamma = 2.0; rho = 0.5; sigma = 0.5
    for it in range(max_iter):
        if time.time() > deadline:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if max(fvals) - min(fvals) < 1e-12:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe; fvals[-1] = fe
            else:
                simplex[-1] = xr; fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr; fvals[-1] = fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc; fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

    best_x = np.clip(simplex[0], 0.0, max_layer_nm)
    if best_x.sum() > max_total_nm:
        scale = max_total_nm / best_x.sum()
        best_x *= scale
    best_f = _evaluate_design(mats, list(best_x), problem, wavelengths)
    return list(best_x), best_f


def _polish_thicknesses(mats, thicks, problem, wavelengths, deadline, max_iter=3000):
    """High-precision coordinate descent polish."""
    current_thicks = list(thicks)
    current_score = _evaluate_design(mats, current_thicks, problem, wavelengths)
    n = len(mats)
    if n == 0:
        return mats, current_thicks, current_score
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    step_sizes = [1.0] * n
    min_step = 0.001
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            best_local_score = current_score
            best_local_thick = current_thicks[i]
            for factor in [1.0, 0.5, 0.25, 0.1]:
                step = step_sizes[i] * factor
                for sign in (1, -1):
                    new_thick = current_thicks[i] + sign * step
                    if new_thick < 0:
                        new_thick = 0.0
                    if new_thick > max_layer_nm:
                        new_thick = max_layer_nm
                    new_thicks = list(current_thicks)
                    new_thicks[i] = new_thick
                    if _is_valid(mats, new_thicks, problem):
                        new_score = _evaluate_design(mats, new_thicks, problem, wavelengths)
                        if new_score < best_local_score:
                            best_local_score = new_score
                            best_local_thick = new_thick
            if best_local_score < current_score - 1e-10:
                current_thicks[i] = best_local_thick
                current_score = best_local_score
                improved = True
        if not improved:
            step_sizes = [s * 0.5 for s in step_sizes]
            if min(step_sizes) < min_step:
                break
        else:
            step_sizes = [min(s * 1.2, 5.0) for s in step_sizes]
    return mats, current_thicks, current_score


def _simulated_annealing(problem, wavelengths, deadline, max_iter=50000):
    """Simulated annealing that can change both material sequence and thicknesses."""
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])

    def qw(m):
        return centre / (4.0 * materials_dict[m])

    # Initialize with a random design
    n_layers_init = random.randint(4, min(12, max_layers))
    current_mats = [random.choice(['L', 'H']) for _ in range(n_layers_init)]
    current_thicks = [qw(m) for m in current_mats]
    current_score = _evaluate_design(current_mats, current_thicks, problem, wavelengths)

    best_mats = list(current_mats)
    best_thicks = list(current_thicks)
    best_score = current_score

    T_start = 5.0
    T_end = 0.001
    cooling = (T_end / T_start) ** (1.0 / max_iter)
    T = T_start

    rng = random.Random(42)

    for it in range(max_iter):
        if time.time() > deadline:
            break

        T *= cooling

        neighbor_mats = list(current_mats)
        neighbor_thicks = list(current_thicks)

        move_type = rng.random()

        if move_type < 0.3:
            idx = rng.randint(0, len(neighbor_thicks) - 1)
            delta = rng.gauss(0, 30)
            neighbor_thicks[idx] = max(0, min(max_layer_nm, neighbor_thicks[idx] + delta))
        elif move_type < 0.5:
            idx = rng.randint(0, len(neighbor_mats) - 1)
            new_mat = 'L' if neighbor_mats[idx] == 'H' else 'H'
            neighbor_mats[idx] = new_mat
            neighbor_thicks[idx] = qw(new_mat)
        elif move_type < 0.6 and len(neighbor_mats) < max_layers:
            pos = rng.randint(0, len(neighbor_mats))
            new_mat = rng.choice(['L', 'H'])
            neighbor_mats.insert(pos, new_mat)
            neighbor_thicks.insert(pos, qw(new_mat))
        elif move_type < 0.7 and len(neighbor_mats) > 1:
            pos = rng.randint(0, len(neighbor_mats) - 1)
            neighbor_mats.pop(pos)
            neighbor_thicks.pop(pos)
        elif move_type < 0.8:
            if len(neighbor_mats) >= 2:
                i, j = rng.sample(range(len(neighbor_mats)), 2)
                neighbor_mats[i], neighbor_mats[j] = neighbor_mats[j], neighbor_mats[i]
                neighbor_thicks[i], neighbor_thicks[j] = neighbor_thicks[j], neighbor_thicks[i]
        else:
            for _ in range(rng.randint(1, 3)):
                idx = rng.randint(0, len(neighbor_thicks) - 1)
                delta = rng.gauss(0, 20)
                neighbor_thicks[idx] = max(0, min(max_layer_nm, neighbor_thicks[idx] + delta))

        if not _is_valid(neighbor_mats, neighbor_thicks, problem):
            continue

        neighbor_score = _evaluate_design(neighbor_mats, neighbor_thicks, problem, wavelengths)
        delta = neighbor_score - current_score

        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_mats = neighbor_mats
            current_thicks = neighbor_thicks
            current_score = neighbor_score

            if current_score < best_score:
                best_score = current_score
                best_mats = list(current_mats)
                best_thicks = list(current_thicks)

    return best_mats, best_thicks, best_score


def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0

    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    centre = 0.5 * (band_nm[0] + band_nm[1])

    def qw(m):
        return centre / (4.0 * materials_dict[m])

    # --- Build diverse initial seeds ---
    initial_designs = []
    for n in range(1, min(15, max_layers + 1)):
        seq_h = ["H" if i % 2 == 0 else "L" for i in range(n)]
        initial_designs.append((seq_h, [qw(m) for m in seq_h]))
        seq_l = ["L" if i % 2 == 0 else "H" for i in range(n)]
        initial_designs.append((seq_l, [qw(m) for m in seq_l]))

    initial_designs.append((["H", "L", "H", "L", "H", "L", "H"], [qw("H"), qw("L"), qw("H"), qw("L"), qw("H"), qw("L"), qw("H")]))
    inner = ["L", "H", "L", "H", "L", "H", "L", "H", "L"]
    initial_designs.append((["H"] + inner + ["H"], [qw("H")] + [qw(m) for m in inner] + [qw("H")]))

    initial_designs.append((["H", "L", "L", "H", "L", "L", "H"], [qw("H"), qw("L"), qw("L"), qw("H"), qw("L"), qw("L"), qw("H")]))
    initial_designs.append((["L", "H", "H", "L", "H", "H", "L"], [qw("L"), qw("H"), qw("H"), qw("L"), qw("H"), qw("H"), qw("L")]))

    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    for mats, thicks in initial_designs:
        if not _is_valid(mats, thicks, problem):
            continue
        score = _evaluate_design(mats, thicks, problem, wavelengths)
        if score < best_score:
            best_score = score
            best_materials = list(mats)
            best_thicknesses = list(thicks)

    if best_materials is None:
        best_materials = ["L"]
        best_thicknesses = [qw("L")]
        best_score = _evaluate_design(best_materials, best_thicknesses, problem, wavelengths)

    # --- Simulated annealing for structural exploration ---
    if time.time() < deadline - 60.0:
        sa_mats, sa_thicks, sa_score = _simulated_annealing(problem, wavelengths, deadline - 40.0, max_iter=30000)
        if sa_score < best_score:
            best_score = sa_score
            best_materials = sa_mats
            best_thicknesses = sa_thicks

    # --- Optimize the best design with Nelder-Mead ---
    if time.time() < deadline - 20.0:
        best_thicknesses, best_score = _nelder_mead(best_materials, best_thicknesses, problem, wavelengths, deadline - 10.0, max_iter=2000)

    # --- Polish ---
    if time.time() < deadline - 5.0:
        _, best_thicknesses, best_score = _polish_thicknesses(best_materials, best_thicknesses, problem, wavelengths, deadline - 1.0, max_iter=3000)

    # --- Try more sequences if time permits ---
    if time.time() < deadline - 30.0:
        candidate_sequences = []
        for n_layers in range(3, min(15, max_layers + 1)):
            seq = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
            candidate_sequences.append(seq)
            seq = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
            candidate_sequences.append(seq)
        candidate_sequences.append(["H", "L", "L", "H", "L", "L", "H"])
        candidate_sequences.append(["L", "H", "H", "L", "H", "H", "L"])
        candidate_sequences.append(["H", "L", "H", "L", "L", "H", "L", "H"])
        candidate_sequences.append(["L", "H", "L", "H", "H", "L", "H", "L"])

        seen = set()
        unique_candidates = []
        for seq in candidate_sequences:
            if len(seq) > max_layers:
                continue
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(seq)

        for seq in unique_candidates:
            if time.time() > deadline - 15.0:
                break
            if seq == best_materials:
                continue
            thicks = [qw(m) for m in seq]
            if not _is_valid(seq, thicks, problem):
                continue
            score = _evaluate_design(seq, thicks, problem, wavelengths)
            if score < best_score + 2.0:
                thicks, score = _nelder_mead(seq, thicks, problem, wavelengths, deadline - 8.0, max_iter=1000)
                if time.time() < deadline - 5.0:
                    _, thicks, score = _polish_thicknesses(seq, thicks, problem, wavelengths, deadline - 3.0, max_iter=1000)
                if score < best_score:
                    best_score = score
                    best_materials = list(seq)
                    best_thicknesses = list(thicks)

    # --- Final refinement ---
    if time.time() < deadline - 8.0:
        best_thicknesses, best_score = _nelder_mead(best_materials, best_thicknesses, problem, wavelengths, deadline - 4.0, max_iter=1500)
    if time.time() < deadline - 2.0:
        _, best_thicknesses, best_score = _polish_thicknesses(best_materials, best_thicknesses, problem, wavelengths, deadline - 0.5, max_iter=2000)

    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ["L"]
        best_thicknesses = [qw("L")]

    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
