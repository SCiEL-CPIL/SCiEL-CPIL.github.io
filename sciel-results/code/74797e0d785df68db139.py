"""Broadband antireflection coating design via multi-start local optimization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 150.0
_MARGIN = 10.0  # seconds of safety margin

def _char_matrix(n, d, wl):
    """Characteristic matrix for a single layer at normal incidence."""
    beta = 2.0 * math.pi * n * d / wl
    cb = math.cos(beta)
    sb = math.sin(beta)
    return np.array([[cb, 1j * sb / n],
                     [1j * n * sb, cb]], dtype=complex)

def _fast_reflectance(materials, thicknesses, wl, n0, ns):
    """Reflectance (0..1) at wavelength wl using characteristic matrix method."""
    n_layers = len(materials)
    if n_layers == 0:
        r = (n0 - ns) / (n0 + ns)
        return r * r
    # Build total matrix
    M = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    for i in range(n_layers):
        Mi = _char_matrix(materials[i], thicknesses[i], wl)
        M = M @ Mi
    # Admittance
    Y = (M[1, 0] + M[1, 1] * ns) / (M[0, 0] + M[0, 1] * ns)
    r = (n0 - Y) / (n0 + Y)
    return float(np.real(r * np.conj(r)))

def _mean_reflectance_pct(materials, thicknesses, band, n0, ns, n_wl=401):
    """Mean reflectance in percent over the band."""
    wls = np.linspace(band[0], band[1], n_wl)
    total = 0.0
    for wl in wls:
        total += _fast_reflectance(materials, thicknesses, float(wl), n0, ns)
    return 100.0 * total / n_wl

def _is_valid(materials, thicknesses, max_layers, max_total, max_layer):
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return False
    if sum(thicknesses) > max_total + 1e-6:
        return False
    return True

def _evaluate(materials, thicknesses, problem, n_wl=401):
    """Return mean reflectance in percent, or inf if invalid."""
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    if not _is_valid(materials, thicknesses, max_layers, max_total, max_layer):
        return float('inf')
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    return _mean_reflectance_pct(materials, thicknesses, band, n0, ns, n_wl)

def _local_optimize(materials, thicknesses, problem, max_iters=200, n_wl_eval=50):
    """Adaptive coordinate descent with layer add/remove."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    # Use fewer wavelengths for inner loop speed
    wls_coarse = np.linspace(band[0], band[1], n_wl_eval)
    
    def eval_coarse(mats, thks):
        if not _is_valid(mats, thks, max_layers, max_total, max_layer):
            return float('inf')
        total = 0.0
        for wl in wls_coarse:
            total += _fast_reflectance(mats, thks, float(wl), n0, ns)
        return 100.0 * total / n_wl_eval
    
    current_score = eval_coarse(materials, thicknesses)
    if current_score == float('inf'):
        return materials, thicknesses
    
    step_sizes = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2]
    step_idx = 0
    no_improve = 0
    max_no_improve = 15
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - _MARGIN:
            break
        improved = False
        
        # Coordinate descent on thicknesses
        for i in range(len(thicknesses)):
            if time.time() > _DEADLINE - _MARGIN:
                break
            old_t = thicknesses[i]
            best_t = old_t
            best_s = current_score
            for delta in [step_sizes[step_idx], -step_sizes[step_idx],
                         step_sizes[step_idx] * 0.5, -step_sizes[step_idx] * 0.5]:
                new_t = old_t + delta
                if new_t < 0 or new_t > max_layer:
                    continue
                new_thks = list(thicknesses)
                new_thks[i] = new_t
                s = eval_coarse(materials, new_thks)
                if s < best_s - 1e-6:
                    best_s = s
                    best_t = new_t
            if best_t != old_t:
                thicknesses[i] = best_t
                current_score = best_s
                improved = True
                no_improve = 0
            else:
                no_improve += 1
        
        # Try adding a layer
        if len(materials) < max_layers and not improved:
            if time.time() > _DEADLINE - _MARGIN:
                break
            current_total = sum(thicknesses)
            if current_total < max_total - 10:
                best_add_s = current_score
                best_add_pos = -1
                best_add_mat = None
                best_add_thk = None
                for pos in range(len(materials) + 1):
                    for mat in ['L', 'H']:
                        n_mat = problem["materials"][mat]
                        # Try a few thicknesses
                        for thk in [30, 50, 80, 100, 120, 150]:
                            if current_total + thk > max_total:
                                continue
                            new_mats = list(materials)
                            new_mats.insert(pos, mat)
                            new_thks = list(thicknesses)
                            new_thks.insert(pos, thk)
                            s = eval_coarse(new_mats, new_thks)
                            if s < best_add_s - 1e-6:
                                best_add_s = s
                                best_add_pos = pos
                                best_add_mat = mat
                                best_add_thk = thk
                if best_add_pos >= 0:
                    materials.insert(best_add_pos, best_add_mat)
                    thicknesses.insert(best_add_pos, best_add_thk)
                    current_score = best_add_s
                    improved = True
                    no_improve = 0
        
        # Try removing a layer
        if len(materials) > 1 and not improved:
            if time.time() > _DEADLINE - _MARGIN:
                break
            for pos in range(len(materials)):
                new_mats = materials[:pos] + materials[pos+1:]
                new_thks = thicknesses[:pos] + thicknesses[pos+1:]
                s = eval_coarse(new_mats, new_thks)
                if s < current_score - 1e-6:
                    materials = new_mats
                    thicknesses = new_thks
                    current_score = s
                    improved = True
                    no_improve = 0
                    break
        
        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                # Reduce step size
                if step_idx < len(step_sizes) - 1:
                    step_idx += 1
                    no_improve = 0
                else:
                    break
            if no_improve >= max_no_improve * 2:
                break
    
    # Final evaluation with full band
    final_score = _evaluate(materials, thicknesses, problem)
    return materials, thicknesses

def _generate_candidates(problem):
    """Generate diverse initial candidates."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    candidates = []
    
    # Quarter-wave L
    qw_L = centre / (4.0 * nL)
    candidates.append((["L"], [qw_L]))
    
    # Quarter-wave H
    qw_H = centre / (4.0 * nH)
    candidates.append((["H"], [qw_H]))
    
    # L-H quarter-wave pair
    if qw_L + qw_H <= max_total:
        candidates.append((["L", "H"], [qw_L, qw_H]))
        candidates.append((["H", "L"], [qw_H, qw_L]))
    
    # Multi-layer quarter-wave stacks
    for n in range(3, 7):
        total = 0.0
        mats = []
        thks = []
        for i in range(n):
            if i % 2 == 0:
                mat = "L"
                thk = centre / (4.0 * nL)
            else:
                mat = "H"
                thk = centre / (4.0 * nH)
            if total + thk > max_total:
                break
            mats.append(mat)
            thks.append(thk)
            total += thk
        if len(mats) >= 2:
            candidates.append((mats, thks))
    
    # Non-quarter-wave starting points
    for factor in [0.7, 0.8, 1.2, 1.3]:
        t = centre * factor / (4.0 * nL)
        if t <= max_layer:
            candidates.append((["L"], [t]))
        t = centre * factor / (4.0 * nH)
        if t <= max_layer:
            candidates.append((["H"], [t]))
    
    # Two-layer non-quarter-wave
    for f1 in [0.8, 1.0, 1.2]:
        for f2 in [0.8, 1.0, 1.2]:
            t1 = centre * f1 / (4.0 * nL)
            t2 = centre * f2 / (4.0 * nH)
            if t1 <= max_layer and t2 <= max_layer and t1 + t2 <= max_total:
                candidates.append((["L", "H"], [t1, t2]))
                candidates.append((["H", "L"], [t2, t1]))
    
    # Three-layer
    for f1 in [0.8, 1.0]:
        for f2 in [0.8, 1.0]:
            for f3 in [0.8, 1.0]:
                t1 = centre * f1 / (4.0 * nL)
                t2 = centre * f2 / (4.0 * nH)
                t3 = centre * f3 / (4.0 * nL)
                if (t1 <= max_layer and t2 <= max_layer and t3 <= max_layer and
                    t1 + t2 + t3 <= max_total):
                    candidates.append((["L", "H", "L"], [t1, t2, t3]))
    
    return candidates

def design(problem):
    """Multi-start local optimization for broadband antireflection coating."""
    start_time = time.time()
    
    candidates = _generate_candidates(problem)
    
    best_score = float('inf')
    best_mats = None
    best_thks = None
    
    # Evaluate and optimize each candidate
    for mats, thks in candidates:
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        # Quick pre-evaluation
        pre_score = _evaluate(mats, thks, problem, n_wl=100)
        if pre_score < best_score:
            best_score = pre_score
            best_mats = list(mats)
            best_thks = list(thks)
        
        # Local optimization
        opt_mats, opt_thks = _local_optimize(
            list(mats), list(thks), problem,
            max_iters=100, n_wl_eval=40
        )
        
        opt_score = _evaluate(opt_mats, opt_thks, problem, n_wl=100)
        if opt_score < best_score:
            best_score = opt_score
            best_mats = opt_mats
            best_thks = opt_thks
        
        # If we have good time remaining, do more optimization on best
        if time.time() < _DEADLINE - 30:
            opt_mats2, opt_thks2 = _local_optimize(
                list(best_mats), list(best_thks), problem,
                max_iters=80, n_wl_eval=50
            )
            opt_score2 = _evaluate(opt_mats2, opt_thks2, problem, n_wl=100)
            if opt_score2 < best_score:
                best_score = opt_score2
                best_mats = opt_mats2
                best_thks = opt_thks2
    
    # Final refinement with more iterations on the best candidate
    if best_mats is not None and time.time() < _DEADLINE - 20:
        opt_mats, opt_thks = _local_optimize(
            list(best_mats), list(best_thks), problem,
            max_iters=150, n_wl_eval=60
        )
        opt_score = _evaluate(opt_mats, opt_thks, problem, n_wl=100)
        if opt_score < best_score:
            best_mats = opt_mats
            best_thks = opt_thks
    
    # Ensure validity
    if best_mats is None or not _is_valid(best_mats, best_thks, 
                                           problem["max_layers"],
                                           problem["max_total_nm"],
                                           problem["max_layer_nm"]):
        # Fallback: single quarter-wave L
        centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
        best_mats = ["L"]
        best_thks = [centre / (4.0 * problem["materials"]["L"])]
    
    # Clamp thicknesses
    best_thks = [max(0.0, min(problem["max_layer_nm"], t)) for t in best_thks]
    
    return best_mats, best_thks
# EVOLVE-BLOCK-END
