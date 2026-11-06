"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _reflectance_vectorized(materials, thicknesses, wls, n_amb, n_sub, mat_dict):
    """
    Vectorized reflectance calculation for a stack of layers.
    Returns an array of reflectance values (fraction, not percent) for the given wavelengths.
    """
    n_layers = len(materials)
    if n_layers == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.full_like(wls, r * r)
    
    ns = np.array([mat_dict[m] for m in materials])
    ds = np.array(thicknesses)
    
    M_total = np.zeros((len(wls), 2, 2), dtype=complex)
    M_total[:, 0, 0] = 1.0
    M_total[:, 1, 1] = 1.0
    
    for i in range(n_layers):
        n = ns[i]
        d = ds[i]
        if d < 1e-9:
            continue
        beta = 2.0 * np.pi * n * d / wls
        c = np.cos(beta)
        s = np.sin(beta)
        
        M_layer = np.zeros((len(wls), 2, 2), dtype=complex)
        M_layer[:, 0, 0] = c
        M_layer[:, 0, 1] = 1j * s / n
        M_layer[:, 1, 0] = 1j * n * s
        M_layer[:, 1, 1] = c
        
        M_total = np.einsum('wij,wjk->wik', M_total, M_layer)
        
    Y = (M_total[:, 1, 0] + M_total[:, 1, 1] * n_sub) / (M_total[:, 0, 0] + M_total[:, 0, 1] * n_sub)
    r = (n_amb - Y) / (n_amb + Y)
    return np.abs(r) ** 2

def _mean_reflectance(materials, thicknesses, wls, n_amb, n_sub, mat_dict):
    """Calculate mean reflectance in percent."""
    refls = _reflectance_vectorized(materials, thicknesses, wls, n_amb, n_sub, mat_dict)
    return 100.0 * np.mean(refls)

def _is_valid(materials, thicknesses, problem):
    """Check constraints."""
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True

def design(problem):
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit

    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    mat_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    best_materials = ["L"]
    best_thicknesses = [600.0 / (4.0 * mat_dict["L"])]
    best_score = _mean_reflectance(best_materials, best_thicknesses, wls, n_amb, n_sub, mat_dict)
    
    def try_candidate(materials, thicknesses):
        nonlocal best_materials, best_thicknesses, best_score
        if not _is_valid(materials, thicknesses, problem):
            return
        score = _mean_reflectance(materials, thicknesses, wls, n_amb, n_sub, mat_dict)
        if score < best_score:
            best_score = score
            best_materials = materials[:]
            best_thicknesses = thicknesses[:]

    # Strategy: Discrete combinatorial search over QWOT multipliers.
    # We restrict to alternating L/H stacks starting with L (standard for AR on glass).
    # Thickness of layer i is k_i * (lambda_ref / (4 * n_i)).
    # We vary lambda_ref and the set of k_i.
    
    # Common good multipliers for broadband AR: 1, 3, 5, 7...
    # Let's try a few reference wavelengths and a set of multiplier patterns.
    
    # To speed up, we can precompute the matrix for a base QWOT (k=1) and then
    # effectively just change the angle beta. But beta depends on lambda and d.
    # d = k * lambda_ref / (4n).
    # beta = 2 pi n d / lambda = 2 pi n (k lambda_ref / 4n) / lambda = pi k lambda_ref / (2 lambda).
    # So beta is linear in k.
    
    # We will iterate over:
    # 1. Number of layers N (2 to 10)
    # 2. Reference wavelength lambda_ref (500, 550, 600)
    # 3. Multiplier patterns. Instead of all combinations (4^N is too big for N=10),
    #    we will use a small set of heuristic patterns and then a local search on the multipliers.
    
    # Heuristic patterns:
    # All 1s (standard QW stack)
    # 1, 3, 1, 3...
    # 1, 1, 3, 3...
    # 1, 2, 1, 2... (not strictly odd, but let's allow 1-5)
    
    # Let's define a function to evaluate a specific pattern of multipliers for a given N and lambda_ref
    def evaluate_pattern(N, lambda_ref, multipliers):
        mats = []
        thicks = []
        for i in range(N):
            mat = 'L' if i % 2 == 0 else 'H'
            mats.append(mat)
            n = mat_dict[mat]
            d = multipliers[i] * lambda_ref / (4.0 * n)
            thicks.append(d)
        
        if not _is_valid(mats, thicks, problem):
            return None, None
        score = _mean_reflectance(mats, thicks, wls, n_amb, n_sub, mat_dict)
        return mats, thicks, score

    # Local search on multipliers
    def optimize_multipliers(N, lambda_ref, init_mults):
        current_mults = list(init_mults)
        current_mats, current_thicks, current_score = evaluate_pattern(N, lambda_ref, current_mults)
        if current_score is None:
            return None, None, None
        
        best_mults = current_mults[:]
        best_score = current_score
        best_mats = current_mats
        best_thicks = current_thicks
        
        step_size = 1 # Change by 1 multiplier unit
        max_iters = 200
        
        for i in range(max_iters):
            if time.time() - start_time > deadline:
                break
                
            improved = False
            # Try changing one multiplier by +/- 1, +/- 2
            for j in range(N):
                for delta in [-2, -1, 1, 2]:
                    new_mults = current_mults[:]
                    new_mults[j] += delta
                    # Keep multipliers in a reasonable range, e.g., 1 to 7
                    if new_mults[j] < 1 or new_mults[j] > 7:
                        continue
                        
                    new_mats, new_thicks, new_score = evaluate_pattern(N, lambda_ref, new_mults)
                    if new_score is None:
                        continue
                    
                    if new_score < current_score - 1e-5:
                        current_mults = new_mults
                        current_mats = new_mats
                        current_thicks = new_thicks
                        current_score = new_score
                        improved = True
                        
                        if current_score < best_score:
                            best_score = current_score
                            best_mults = current_mults[:]
                            best_mats = current_mats
                            best_thicks = current_thicks
                        break
                if improved:
                    break
            
            if not improved:
                # No improvement, maybe try a random jump?
                # Or just break if we've been stuck too long.
                # For simplicity, let's just break if no improvement in this iteration.
                # A more robust search would use simulated annealing, but this is a start.
                pass

        return best_mats, best_thicks, best_score

    # Search over N and lambda_ref
    for N in range(2, 11):
        if time.time() - start_time > deadline:
            break
            
        for lambda_ref in [500, 550, 600]:
            # Initialize with all 1s
            init_mults = [1] * N
            mats, thicks, score = optimize_multipliers(N, lambda_ref, init_mults)
            if score is not None:
                try_candidate(mats, thicks)
                
            # Try a few other initial patterns
            # Pattern: 1, 3, 1, 3...
            init_mults = [1 if i % 2 == 0 else 3 for i in range(N)]
            mats, thicks, score = optimize_multipliers(N, lambda_ref, init_mults)
            if score is not None:
                try_candidate(mats, thicks)
                
            # Pattern: 1, 1, 3, 3...
            init_mults = [1 if i % 4 < 2 else 3 for i in range(N)]
            mats, thicks, score = optimize_multipliers(N, lambda_ref, init_mults)
            if score is not None:
                try_candidate(mats, thicks)

    # Final refinement: If we have time, do a continuous optimization on the best discrete solution
    # This helps fine-tune the "lambda_ref" effectively by scaling all thicknesses.
    if best_materials and time.time() - start_time < deadline - 10:
        # Simple scaling search
        base_thicks = best_thicknesses[:]
        base_score = best_score
        
        for scale in np.linspace(0.8, 1.2, 20):
            new_thicks = [t * scale for t in base_thicks]
            if _is_valid(best_materials, new_thicks, problem):
                score = _mean_reflectance(best_materials, new_thicks, wls, n_amb, n_sub, mat_dict)
                if score < best_score:
                    best_score = score
                    best_thicknesses = new_thicks

    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
