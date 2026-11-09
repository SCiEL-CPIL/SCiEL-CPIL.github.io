"""Antireflection coating design using simulated annealing on layer thicknesses and pattern."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _compute_reflectance(materials, thicknesses, n_ambient, n_substrate, band_nm, n_points=401):
    """Compute mean normal-incidence reflectance (percent) using characteristic-matrix method."""
    L = len(materials)
    if L == 0:
        # Bare substrate
        n0 = n_ambient
        ns = n_substrate
        R = ((n0 - ns) / (n0 + ns)) ** 2
        return 100.0 * R

    wls = np.linspace(band_nm[0], band_nm[1], n_points)
    n_mat = np.array([materials[m] for m in materials])
    d = np.array(thicknesses, dtype=float)

    # Precompute for speed
    total_R = 0.0
    for wl in wls:
        k = 2.0 * np.pi / wl  # 1/nm
        # Build characteristic matrix
        M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        for i in range(L):
            n_i = n_mat[i]
            phase = k * n_i * d[i]
            # For non-absorbing materials:
            # M_i = [[cos(phase), (i/n_i)*sin(phase)], [i*n_i*sin(phase), cos(phase)]]
            c = np.cos(phase)
            s = np.sin(phase)
            Mi = np.array([[c, 1j * s / n_i],
                           [1j * n_i * s, c]], dtype=complex)
            M = M @ Mi

        # Admittance at air side
        Y = (M[0, 1]) / (M[1, 1])  # Actually Y = M12 / M22? Let me be careful.
        # Characteristic matrix M = [[M11, M12], [M21, M22]]
        # The optical admittance at the front is Y = M22 / M12? No.
        # Standard: Y = M22 / M12 is not right.
        # Let's use: The field at the front is related by:
        # [E]   = M [E_sub]
        # [H]       [H_sub]
        # So E0 = M11*Es + M12*Hs, H0 = M21*Es + M22*Hs
        # Admittance Y = H0/E0 = (M21*Es + M22*Hs)/(M11*Es + M12*Hs)
        # At substrate: Hs = Es / n_sub (for TE? No, normal incidence, H = E/n)
        # Actually for normal incidence: H = E / n (in units where impedance is 1/n)
        # So Hs = Es / n_sub
        # Y = (M21/n_sub + M22) / (M11/n_sub + M12)
        # R = |(Y - n_ambient) / (Y + n_ambient)|^2

        Y = (M[1, 0] / n_substrate + M[1, 1]) / (M[0, 0] / n_substrate + M[0, 1])
        R = np.abs((Y - n_ambient) / (Y + n_ambient)) ** 2
        total_R += R
    return 100.0 * total_R / n_points


def _is_valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > problem["max_layers"]:
        return False
    if sum(thicknesses) > problem["max_total_nm"] + 1e-9:
        return False
    for t in thicknesses:
        if t < -1e-9 or t > problem["max_layer_nm"] + 1e-9:
            return False
    for m in materials:
        if m not in problem["materials"]:
            return False
    return True


def _evaluate(materials, thicknesses, problem):
    """Evaluate the design, returning reflectance or infinity if invalid."""
    if not _is_valid(materials, thicknesses, problem):
        return float('inf')
    return _compute_reflectance(materials, thicknesses,
                                 problem["ambient_index"],
                                 problem["substrate_index"],
                                 problem["band_nm"])


def _random_stack(problem, n_layers):
    """Generate a random valid stack."""
    materials = ['L' if np.random.rand() < 0.5 else 'H' for _ in range(n_layers)]
    # Random thicknesses, ensuring sum <= max_total_nm
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    # Generate random thicknesses that sum to at most max_total
    # Use Dirichlet-like distribution
    raw = np.random.rand(n_layers)
    total = raw.sum()
    thicknesses = (raw / total) * min(max_total, max_layer * n_layers)
    # Clip to max_layer
    thicknesses = np.clip(thicknesses, 0, max_layer)
    # If sum exceeds max_total, scale down
    s = thicknesses.sum()
    if s > max_total:
        thicknesses = thicknesses * (max_total / s)
    return materials, thicknesses.tolist()


def _perturb(materials, thicknesses, problem, rng):
    """Perturb the current stack."""
    n_layers = len(materials)
    new_materials = materials[:]
    new_thicknesses = thicknesses[:]

    # Randomly choose what to perturb
    action = rng.integers(0, 4)

    if action == 0:
        # Perturb a random thickness
        i = rng.integers(0, n_layers)
        delta = rng.normal(0, 20)
        new_thicknesses[i] = max(0, min(problem["max_layer_nm"], new_thicknesses[i] + delta))
    elif action == 1:
        # Swap material at a random position
        i = rng.integers(0, n_layers)
        new_materials[i] = 'H' if new_materials[i] == 'L' else 'L'
    elif action == 2:
        # Swap two adjacent layers
        if n_layers >= 2:
            i = rng.integers(0, n_layers - 1)
            new_materials[i], new_materials[i+1] = new_materials[i+1], new_materials[i]
            new_thicknesses[i], new_thicknesses[i+1] = new_thicknesses[i+1], new_thicknesses[i]
    else:
        # Add or remove a layer
        if n_layers < problem["max_layers"] and rng.rand() < 0.5:
            # Add a layer
            pos = rng.integers(0, n_layers + 1)
            m = 'L' if rng.rand() < 0.5 else 'H'
            t = rng.uniform(10, problem["max_layer_nm"])
            new_materials.insert(pos, m)
            new_thicknesses.insert(pos, t)
        elif n_layers > 1:
            # Remove a layer
            pos = rng.integers(0, n_layers)
            new_materials.pop(pos)
            new_thicknesses.pop(pos)

    # Check validity, fix if needed
    if sum(new_thicknesses) > problem["max_total_nm"]:
        scale = problem["max_total_nm"] / sum(new_thicknesses)
        new_thicknesses = [t * scale for t in new_thicknesses]
    new_thicknesses = [max(0, min(problem["max_layer_nm"], t)) for t in new_thicknesses]

    return new_materials, new_thicknesses


def design(problem):
    """Design an antireflection coating stack using simulated annealing."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s wall-clock limit

    rng = np.random.default_rng(42)

    # Initialize with a few random starts
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')

    n_layers_options = [3, 4, 5, 6, 7, 8]

    # Quick initial evaluations to get a starting point
    for n_layers in n_layers_options:
        for trial in range(3):
            if time.time() > deadline:
                break
            materials, thicknesses = _random_stack(problem, n_layers)
            score = _evaluate(materials, thicknesses, problem)
            if score < best_score:
                best_score = score
                best_materials = materials[:]
                best_thicknesses = thicknesses[:]

    if best_materials is None:
        # Fallback: single quarter-wave L layer
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        return ["L"], [centre / (4.0 * problem["materials"]["L"])]

    # Simulated annealing
    current_materials = best_materials[:]
    current_thicknesses = best_thicknesses[:]
    current_score = best_score

    T_start = 0.5
    T_end = 0.01
    n_iter = 5000
    alpha = (T_end / T_start) ** (1.0 / n_iter)

    T = T_start
    for i in range(n_iter):
        if time.time() > deadline:
            break

        new_materials, new_thicknesses = _perturb(current_materials, current_thicknesses, problem, rng)
        new_score = _evaluate(new_materials, new_thicknesses, problem)

        if new_score < current_score:
            current_materials = new_materials
            current_thicknesses = new_thicknesses
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_materials = current_materials[:]
                best_thicknesses = current_thicknesses[:]
        else:
            # Accept with probability exp(-delta/T)
            delta = new_score - current_score
            if delta < 0:
                accept = True
            else:
                accept_prob = np.exp(-delta / T)
                accept = rng.rand() < accept_prob
            if accept:
                current_materials = new_materials
                current_thicknesses = new_thicknesses
                current_score = new_score

        T *= alpha

    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
