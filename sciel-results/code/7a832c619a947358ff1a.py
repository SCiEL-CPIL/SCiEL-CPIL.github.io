import time
import numpy as np

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns (materials, thicknesses).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Problem parameters
    band = problem["band_nm"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation
    wavelengths = np.linspace(band[0], band[1], 401)
    k = 2.0 * np.pi / wavelengths  # (401,)
    N = len(wavelengths)
    
    def compute_reflectance(materials_list, thicknesses):
        """
        Compute mean reflectance percentage for a given design.
        materials_list: list of 'L'/'H'
        thicknesses: list of floats
        """
        n_layers = len(materials_list)
        
        # Initialize identity matrix components
        M00 = np.ones(N, dtype=np.complex128)
        M01 = np.zeros(N, dtype=np.complex128)
        M10 = np.zeros(N, dtype=np.complex128)
        M11 = np.ones(N, dtype=np.complex128)
        
        for j in range(n_layers):
            n_j = n_L if materials_list[j] == 'L' else n_H
            d_j = thicknesses[j]
            delta_j = k * n_j * d_j
            
            cos_d = np.cos(delta_j)
            sin_d = np.sin(delta_j)
            
            # Layer matrix
            L00 = cos_d
            L01 = 1j * sin_d / n_j
            L10 = 1j * n_j * sin_d
            L11 = cos_d
            
            # M_new = M_old * L
            new_M00 = M00 * L00 + M01 * L10
            new_M01 = M00 * L01 + M01 * L11
            new_M10 = M10 * L00 + M11 * L10
            new_M11 = M10 * L01 + M11 * L11
            
            M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
        
        # Admittance at front
        Y = M01 / M00
        R = np.abs((n_ambient - Y) / (n_ambient + Y)) ** 2
        
        return float(np.mean(R) * 100.0)

    def is_valid(materials_list, thicknesses):
        if len(materials_list) != len(thicknesses):
            return False
        if len(materials_list) > max_layers or len(materials_list) == 0:
            return False
        if sum(thicknesses) > max_total_nm:
            return False
        for t in thicknesses:
            if t < 0 or t > max_layer_nm:
                return False
        for m in materials_list:
            if m not in ('L', 'H'):
                return False
        return True

    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Generate candidate patterns to optimize
    # We will try alternating patterns with varying layer counts
    # and optimize their thicknesses using Simulated Annealing.
    
    candidates = []
    
    # Quarter-wave estimates for initialization
    center_wl = 0.5 * (band[0] + band[1])
    d_L_qw = center_wl / (4.0 * n_L)
    d_H_qw = center_wl / (4.0 * n_H)
    
    # Try layer counts: 3, 5, 7, 9, 11, 13, 15
    # Up to 20 layers allowed, but more layers means more parameters and slower convergence.
    # 15 layers is a good upper bound for SA within the time limit.
    
    for n_layers in [3, 5, 7, 9, 11, 13, 15]:
        if n_layers > max_layers:
            break
            
        # Pattern 1: Starts with L
        mats_L = []
        thicks_L = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats_L.append('L')
                thicks_L.append(d_L_qw)
            else:
                mats_L.append('H')
                thicks_L.append(d_H_qw)
        if sum(thicks_L) <= max_total_nm:
            candidates.append((mats_L, thicks_L))
            
        # Pattern 2: Starts with H
        mats_H = []
        thicks_H = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats_H.append('H')
                thicks_H.append(d_H_qw)
            else:
                mats_H.append('L')
                thicks_H.append(d_L_qw)
        if sum(thicks_H) <= max_total_nm:
            candidates.append((mats_H, thicks_H))
            
    # Also add a simple 1-layer and 2-layer case just in case
    candidates.append((['L'], [d_L_qw]))
    candidates.append((['H'], [d_H_qw]))
    candidates.append((['L', 'H'], [d_L_qw, d_H_qw]))
    candidates.append((['H', 'L'], [d_H_qw, d_L_qw]))
    
    # Sort candidates by initial score to prioritize better starts
    scored_candidates = []
    for mats, thicks in candidates:
        if is_valid(mats, thicks):
            score = compute_reflectance(mats, thicks)
            scored_candidates.append((score, mats, thicks))
            
    scored_candidates.sort(key=lambda x: x[0])
    
    # Run Simulated Annealing on the top candidates
    # We have ~14 candidates. We can run SA on all of them or just the top few.
    # Given 140s, we can afford to run SA on several.
    # Let's allocate time per candidate.
    
    num_candidates = len(scored_candidates)
    if num_candidates == 0:
        return ['L'], [d_L_qw]
    
    time_per_candidate = (deadline - time.time()) / num_candidates
    
    for i, (init_score, mats, thicks) in enumerate(scored_candidates):
        if time.time() > deadline - 5.0:
            break
            
        # If we already have a very good score, we might skip worse candidates
        # But SA might find a global minimum in a worse-looking local basin.
        # We'll proceed with all, but limit iterations based on remaining time.
        
        # Simulated Annealing Parameters
        T0 = 1.0  # Initial temperature (in percent reflectance units)
        # T_final should be low
        # Number of iterations: let's aim for ~1000-5000 iterations per candidate
        # Estimate time per iteration: reflectance calc is fast (~1ms for 15 layers)
        # So 1000 iterations ~ 1-2s.
        
        current_thicks = np.array(thicks, dtype=float)
        current_score = init_score
        
        best_local_thicks = current_thicks.copy()
        best_local_score = current_score
        
        # Adaptive step size
        # Start with a large step, decrease over time
        step_size = 20.0  # nm
        
        # Cooling schedule
        # We will run for a fixed number of iterations or until time runs out
        # Let's use time-based loop
        sa_start = time.time()
        sa_time_limit = time_per_candidate * 0.9  # Leave some margin
        
        T = T0
        cooling_factor = 0.995
        
        while time.time() - sa_start < sa_time_limit:
            # Perturb
            # Pick a random layer
            idx = np.random.randint(len(current_thicks))
            # Random delta
            delta = np.random.uniform(-step_size, step_size)
            
            new_thicks = current_thicks.copy()
            new_thicks[idx] += delta
            
            # Check constraints
            valid = True
            if np.any(new_thicks < 0) or np.any(new_thicks > max_layer_nm):
                valid = False
            if np.sum(new_thicks) > max_total_nm:
                valid = False
                
            if valid:
                new_score = compute_reflectance(mats, new_thicks)
                delta_E = new_score - current_score
                
                # Metropolis criterion
                if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                    current_thicks = new_thicks
                    current_score = new_score
                    
                    if current_score < best_local_score:
                        best_local_score = current_score
                        best_local_thicks = current_thicks.copy()
            else:
                # If invalid, we can just reject, or try to fix.
                # Rejecting is simpler.
                pass
                
            # Cool
            T *= cooling_factor
            # Decrease step size slightly to refine
            step_size *= 0.999
            
            # If T is very small, we can break early if no improvement for a while
            # But let's stick to time limit for simplicity and robustness.
            
        # Update global best
        if best_local_score < best_score:
            best_score = best_local_score
            best_materials = mats
            best_thicknesses = list(best_local_thicks)
            
        # Optional: Local refinement (Nelder-Mead) on the best found so far
        # This can help polish the SA result.
        if time.time() < deadline - 10.0:
            # Simple gradient-free refinement
            # We can do a few steps of Nelder-Mead or just a fine local search
            # Given the complexity, let's skip NM to save code complexity and risk,
            # SA with shrinking step size usually gets close.
            # However, a quick local search can help.
            
            # Let's do a simple coordinate descent / local search
            # for a few iterations to polish.
            
            polish_thicks = best_local_thicks.copy()
            polish_score = best_local_score
            polish_step = 1.0  # nm
            
            for _ in range(50):  # 50 iterations of local search
                improved = False
                for j in range(len(polish_thicks)):
                    # Try +/- step
                    for sign in [1, -1]:
                        new_t = polish_thicks[j] + sign * polish_step
                        if 0 <= new_t <= max_layer_nm:
                            trial_thicks = polish_thicks.copy()
                            trial_thicks[j] = new_t
                            if np.sum(trial_thicks) <= max_total_nm:
                                trial_score = compute_reflectance(mats, trial_thicks)
                                if trial_score < polish_score - 1e-10:
                                    polish_score = trial_score
                                    polish_thicks = trial_thicks
                                    improved = True
                if not improved:
                    polish_step *= 0.5
                    if polish_step < 0.01:
                        break
                
            if polish_score < best_score:
                best_score = polish_score
                best_materials = mats
                best_thicknesses = list(polish_thicks)

    # Final check
    if best_materials is None:
        return ['L'], [d_L_qw]
        
    if not is_valid(best_materials, best_thicknesses):
        return ['L'], [d_L_qw]
        
    return best_materials, best_thicknesses
