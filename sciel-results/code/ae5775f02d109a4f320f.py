"""Broadband antireflection coating design using characteristic-matrix method."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers):
    """
    Compute reflectance (0-1) for a stack using the characteristic matrix method.
    
    wavelengths: array of wavelengths in nm
    n_ambient: refractive index of ambient
    n_substrate: refractive index of substrate
    n_layers: array of refractive indices of layers
    d_layers: array of physical thicknesses in nm
    
    Returns: array of reflectance values (0-1)
    """
    n = len(n_layers)
    if n == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.abs(r) ** 2
    
    # Initialize identity matrix for each wavelength
    M11 = np.ones(len(wavelengths), dtype=complex)
    M12 = np.zeros(len(wavelengths), dtype=complex)
    M21 = np.zeros(len(wavelengths), dtype=complex)
    M22 = np.ones(len(wavelengths), dtype=complex)
    
    for i in range(n):
        ni = n_layers[i]
        di = d_layers[i]
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
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
    
    Y = (M11 + M12 * n_substrate) / (M21 + M22 * n_substrate)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    
    return R


def _mean_reflectance_percent(wavelengths, n_ambient, n_substrate, n_layers, d_layers):
    """Compute mean reflectance in percent."""
    R = _char_matrix_method(wavelengths, n_ambient, n_substrate, n_layers, d_layers)
    return np.mean(R) * 100.0


def _is_valid(materials, thicknesses, problem):
    """Check if the design satisfies all constraints."""
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
    if total > max_total_nm:
        return False
    
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    
    for m in materials:
        if m not in problem["materials"]:
            return False
    
    return True


def _evaluate_design(materials, thicknesses, problem, wavelengths):
    """Evaluate a design and return mean reflectance in percent."""
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_layers = np.array([problem["materials"][m] for m in materials], dtype=float)
    d_layers = np.array(thicknesses, dtype=float)
    
    return _mean_reflectance_percent(wavelengths, n_ambient, n_substrate, n_layers, d_layers)


def _nelder_mead(objective, x0, bounds, deadline, max_iter=5000, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """
    Nelder-Mead simplex optimization.
    
    objective: function to minimize
    x0: initial point (list or array)
    bounds: list of (min, max) for each dimension
    deadline: time limit
    """
    n = len(x0)
    x0 = np.array(x0, dtype=float)
    
    # Create initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 if bounds[i][1] - bounds[i][0] > 0 else 0.01
        xi[i] += step
        xi[i] = np.clip(xi[i], bounds[i][0], bounds[i][1])
        simplex.append(xi)
    
    # Evaluate initial simplex
    values = [objective(x) for x in simplex]
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Sort simplex by objective value
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[-1] - values[0]) < 1e-10:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        xr = np.clip(xr, [b[0] for b in bounds], [b[1] for b in bounds])
        fr = objective(xr)
        
        if fr < values[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            xe = np.clip(xe, [b[0] for b in bounds], [b[1] for b in bounds])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            xc = np.clip(xc, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = objective(xc)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    values[i] = objective(simplex[i])
    
    # Return best point
    best_idx = np.argmin(values)
    return simplex[best_idx], values[best_idx]


def _simulated_annealing(mats, thicks, problem, wavelengths, deadline, T0=50.0, cooling=0.995, min_T=0.01):
    """
    Simulated annealing over variable-length layer sequences.
    Moves: perturb thickness, swap layers, add layer, remove layer, flip material.
    """
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    
    current_mats = list(mats)
    current_thicks = list(thicks)
    current_score = _evaluate_design(current_mats, current_thicks, problem, wavelengths)
    
    best_mats = list(current_mats)
    best_thicks = list(current_thicks)
    best_score = current_score
    
    T = T0
    n_evals = 0
    max_evals = 50000
    
    while time.time() < deadline and T > min_T and n_evals < max_evals:
        # Generate a random move
        move_type = random.random()
        n_layers = len(current_mats)
        
        if n_layers == 0:
            # Must add a layer
            current_mats.append(random.choice(["L", "H"]))
            current_thicks.append(random.uniform(50, 200))
            n_evals += 1
            if _is_valid(current_mats, current_thicks, problem):
                new_score = _evaluate_design(current_mats, current_thicks, problem, wavelengths)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_mats = list(current_mats)
                        best_thicks = list(current_thicks)
                else:
                    current_mats.pop()
                    current_thicks.pop()
            else:
                current_mats.pop()
                current_thicks.pop()
            T *= cooling
            continue
        
        # Choose move type
        if move_type < 0.5:
            # Perturb a random thickness
            idx = random.randint(0, n_layers - 1)
            delta = random.gauss(0, 20)
            new_thick = current_thicks[idx] + delta
            new_thick = np.clip(new_thick, 0, max_layer_nm)
            
            saved = current_thicks[idx]
            current_thicks[idx] = new_thick
            n_evals += 1
            
            if _is_valid(current_mats, current_thicks, problem):
                new_score = _evaluate_design(current_mats, current_thicks, problem, wavelengths)
                delta_score = new_score - current_score
                if delta_score < 0 or random.random() < np.exp(-delta_score / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_mats = list(current_mats)
                        best_thicks = list(current_thicks)
                else:
                    current_thicks[idx] = saved
            else:
                current_thicks[idx] = saved
                
        elif move_type < 0.7:
            # Swap two adjacent layers
            if n_layers >= 2:
                idx = random.randint(0, n_layers - 2)
                current_mats[idx], current_mats[idx + 1] = current_mats[idx + 1], current_mats[idx]
                current_thicks[idx], current_thicks[idx + 1] = current_thicks[idx + 1], current_thicks[idx]
                n_evals += 1
                
                if _is_valid(current_mats, current_thicks, problem):
                    new_score = _evaluate_design(current_mats, current_thicks, problem, wavelengths)
                    delta_score = new_score - current_score
                    if delta_score < 0 or random.random() < np.exp(-delta_score / T):
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_mats = list(current_mats)
                            best_thicks = list(current_thicks)
                    else:
                        current_mats[idx], current_mats[idx + 1] = current_mats[idx + 1], current_mats[idx]
                        current_thicks[idx], current_thicks[idx + 1] = current_thicks[idx + 1], current_thicks[idx]
                else:
                    current_mats[idx], current_mats[idx + 1] = current_mats[idx + 1], current_mats[idx]
                    current_thicks[idx], current_thicks[idx + 1] = current_thicks[idx + 1], current_thicks[idx]
                    
        elif move_type < 0.85 and n_layers < max_layers:
            # Add a layer at random position
            pos = random.randint(0, n_layers)
            mat = random.choice(["L", "H"])
            thick = random.uniform(30, 150)
            
            # Check total thickness constraint
            if sum(current_thicks) + thick > max_total_nm:
                T *= cooling
                continue
                
            current_mats.insert(pos, mat)
            current_thicks.insert(pos, thick)
            n_evals += 1
            
            if _is_valid(current_mats, current_thicks, problem):
                new_score = _evaluate_design(current_mats, current_thicks, problem, wavelengths)
                delta_score = new_score - current_score
                if delta_score < 0 or random.random() < np.exp(-delta_score / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_mats = list(current_mats)
                        best_thicks = list(current_thicks)
                else:
                    current_mats.pop(pos)
                    current_thicks.pop(pos)
            else:
                current_mats.pop(pos)
                current_thicks.pop(pos)
                
        elif move_type < 0.95 and n_layers > 1:
            # Remove a random layer
            idx = random.randint(0, n_layers - 1)
            saved_mat = current_mats[idx]
            saved_thick = current_thicks[idx]
            current_mats.pop(idx)
            current_thicks.pop(idx)
            n_evals += 1
            
            new_score = _evaluate_design(current_mats, current_thicks, problem, wavelengths)
            delta_score = new_score - current_score
            if delta_score < 0 or random.random() < np.exp(-delta_score / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_mats = list(current_mats)
                    best_thicks = list(current_thicks)
            else:
                current_mats.insert(idx, saved_mat)
                current_thicks.insert(idx, saved_thick)
                
        else:
            # Flip material of a random layer
            idx = random.randint(0, n_layers - 1)
            saved_mat = current_mats[idx]
            current_mats[idx] = "H" if saved_mat == "L" else "L"
            n_evals += 1
            
            if _is_valid(current_mats, current_thicks, problem):
                new_score = _evaluate_design(current_mats, current_thicks, problem, wavelengths)
                delta_score = new_score - current_score
                if delta_score < 0 or random.random() < np.exp(-delta_score / T):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_mats = list(current_mats)
                        best_thicks = list(current_thicks)
                else:
                    current_mats[idx] = saved_mat
            else:
                current_mats[idx] = saved_mat
        
        T *= cooling
    
    return best_mats, best_thicks, best_score


def _optimize_with_nm(mats, thicks, problem, wavelengths, deadline, max_iter=3000):
    """Optimize thicknesses using Nelder-Mead."""
    n = len(mats)
    if n == 0:
        return mats, thicks, _evaluate_design(mats, thicks, problem, wavelengths)
    
    # Setup bounds
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    def objective(x):
        # Check constraints
        if np.any(x < 0) or np.any(x > max_layer_nm):
            return 100.0
        if np.sum(x) > max_total_nm:
            return 100.0
        return _evaluate_design(mats, list(x), problem, wavelengths)
    
    bounds = [(0.0, max_layer_nm)] * n
    x0 = np.array(thicks, dtype=float)
    
    x_opt, f_opt = _nelder_mead(objective, x0, bounds, deadline, max_iter=max_iter)
    
    # Clip to valid range
    x_opt = np.clip(x_opt, 0.0, max_layer_nm)
    
    return mats, list(x_opt), f_opt


def _polish_thicknesses(mats, thicks, problem, wavelengths, deadline, max_iter=5000):
    """High-precision polish using finer step sizes."""
    current_thicks = list(thicks)
    current_score = _evaluate_design(mats, current_thicks, problem, wavelengths)
    
    n = len(mats)
    if n == 0:
        return mats, current_thicks, current_score
    
    step_sizes = [1.0] * n
    min_step = 0.01
    max_step = 5.0
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            best_local_score = current_score
            best_local_thick = current_thicks[i]
            
            for factor in [1.0, 0.5, 0.25, 0.1, 2.0]:
                step = step_sizes[i] * factor
                
                for sign in [1.0, -1.0]:
                    new_thick = current_thicks[i] + sign * step
                    new_thick = np.clip(new_thick, 0.0, 400.0)
                    
                    new_thicks = list(current_thicks)
                    new_thicks[i] = new_thick
                    
                    if _is_valid(mats, new_thicks, problem):
                        new_score = _evaluate_design(mats, new_thicks, problem, wavelengths)
                        if new_score < best_local_score:
                            best_local_score = new_score
                            best_local_thick = new_thick
            
            if best_local_score < current_score:
                current_thicks[i] = best_local_thick
                current_score = best_local_score
                improved = True
        
        if not improved:
            step_sizes = [s * 0.7 for s in step_sizes]
            if min(step_sizes) < min_step:
                break
        else:
            step_sizes = [min(s * 1.3, max_step) for s in step_sizes]
    
    return mats, current_thicks, current_score


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s wall-clock limit
    
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
    
    # Generate diverse initial designs
    initial_designs = []
    
    # Quarter-wave thicknesses
    qw_L = centre / (4.0 * n_L)
    qw_H = centre / (4.0 * n_H)
    
    # Single layer
    initial_designs.append((["L"], [qw_L]))
    initial_designs.append((["H"], [qw_H]))
    
    # Two-layer
    initial_designs.append((["L", "H"], [qw_L, qw_H]))
    initial_designs.append((["H", "L"], [qw_H, qw_L]))
    
    # Three-layer
    initial_designs.append((["H", "L", "H"], [qw_H, qw_L, qw_H]))
    initial_designs.append((["L", "H", "L"], [qw_L, qw_H, qw_L]))
    
    # Four-layer
    initial_designs.append((["H", "L", "H", "L"], [qw_H, qw_L, qw_H, qw_L]))
    initial_designs.append((["L", "H", "L", "H"], [qw_L, qw_H, qw_L, qw_H]))
    
    # Five-layer
    initial_designs.append((["H", "L", "H", "L", "H"], [qw_H, qw_L, qw_H, qw_L, qw_H]))
    initial_designs.append((["L", "H", "L", "H", "L"], [qw_L, qw_H, qw_L, qw_H, qw_L]))
    
    # Six-layer
    initial_designs.append((["H", "L", "H", "L", "H", "L"], [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]))
    initial_designs.append((["L", "H", "L", "H", "L", "H"], [qw_L, qw_H, qw_L, qw_H, qw_L, qw_H]))
    
    # Seven-layer
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H"], 
                           [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H]))
    
    # Eight-layer
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L"],
                           [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]))
    
    # Nine-layer
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H"],
                           [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H]))
    
    # Ten-layer
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L"],
                           [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]))
    
    # Non-standard: HLLH
    initial_designs.append((["H", "L", "L", "H"], [qw_H, qw_L, qw_L, qw_H]))
    initial_designs.append((["L", "H", "H", "L"], [qw_L, qw_H, qw_H, qw_L]))
    
    # Non-standard: HLLLH
    initial_designs.append((["H", "L", "L", "L", "H"], [qw_H, qw_L, qw_L, qw_L, qw_H]))
    
    # Non-standard: HLHLHL
    initial_designs.append((["H", "L", "H", "L", "H", "L"], [qw_H, qw_L, qw_H, qw_L, qw_H, qw_L]))
    
    # Evaluate all initial designs and keep the best
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
    
    # Fallback if no valid initial design
    if best_materials is None:
        best_materials = ["L"]
        best_thicknesses = [qw_L]
        best_score = _evaluate_design(best_materials, best_thicknesses, problem, wavelengths)
    
    # Phase 1: Nelder-Mead on best initial design (quick convergence)
    nm_deadline = start_time + 30.0
    best_materials, best_thicknesses, best_score = _optimize_with_nm(
        best_materials, best_thicknesses, problem, wavelengths, nm_deadline, max_iter=2000
    )
    
    # Phase 2: Simulated annealing over variable-length sequences
    sa_deadline = start_time + 100.0
    sa_mats, sa_thicks, sa_score = _simulated_annealing(
        best_materials, best_thicknesses, problem, wavelengths, sa_deadline
    )
    if sa_score < best_score:
        best_score = sa_score
        best_materials = list(sa_mats)
        best_thicknesses = list(sa_thicks)
    
    # Phase 3: Nelder-Mead polish on the best design found
    if time.time() < deadline - 15.0:
        best_materials, best_thicknesses, best_score = _optimize_with_nm(
            best_materials, best_thicknesses, problem, wavelengths, deadline - 10.0, max_iter=2000
        )
    
    # Phase 4: High-precision coordinate descent polish
    if time.time() < deadline - 3.0:
        best_materials, best_thicknesses, best_score = _polish_thicknesses(
            best_materials, best_thicknesses, problem, wavelengths, deadline, max_iter=5000
        )
    
    # Final validation
    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ["L"]
        best_thicknesses = [qw_L]
    
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
