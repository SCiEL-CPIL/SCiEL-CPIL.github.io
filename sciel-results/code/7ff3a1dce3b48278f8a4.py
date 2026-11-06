import numpy as np
import time
from scipy.optimize import minimize

def _compute_reflectance(wavelengths, n0, ns, n_layers, t_layers):
    """
    Compute mean reflectance for given wavelengths and layer stack using
    the characteristic-matrix method.
    """
    if len(n_layers) == 0:
        r = (n0 - ns) / (n0 + ns)
        return float(np.mean(r**2) * 100.0)

    N = len(n_layers)
    
    # Initialize M as identity
    M11 = np.ones_like(wavelengths, dtype=complex)
    M12 = np.zeros_like(wavelengths, dtype=complex)
    M21 = np.zeros_like(wavelengths, dtype=complex)
    M22 = np.ones_like(wavelengths, dtype=complex)

    for i in range(N):
        ni = n_layers[i]
        ti = t_layers[i]
        delta = 2.0 * np.pi * ni * ti / wavelengths
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        m11 = cos_d
        m12 = 1j * sin_d / ni
        m21 = 1j * ni * sin_d
        m22 = cos_d
        
        new_M11 = M11 * m11 + M12 * m21
        new_M12 = M11 * m12 + M12 * m22
        new_M21 = M21 * m11 + M22 * m21
        new_M22 = M21 * m12 + M22 * m22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22

    numerator = ns * M11 + 1j * M12
    denominator = ns * M21 + 1j * M22
    
    Y = numerator / denominator
    r = (Y - n0) / (Y + n0)
    R = np.abs(r)**2
    
    return float(np.mean(R) * 100.0)


def design(problem):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit

    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    nL = materials["L"]
    nH = materials["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band[0], band[1], 401)
    
    best_score = float('inf')
    best_materials = []
    best_thicknesses = []
    
    def _valid_solution(materials_list, thicknesses_list):
        if len(materials_list) > max_layers:
            return False
        if len(materials_list) != len(thicknesses_list):
            return False
        if any(t < -1e-9 for t in thicknesses_list):
            return False
        if any(t > max_layer_nm + 1e-9 for t in thicknesses_list):
            return False
        if sum(thicknesses_list) > max_total_nm + 1e-9:
            return False
        for m in materials_list:
            if m not in ["L", "H"]:
                return False
        return True

    def _clamp_thicknesses(t):
        """Clamp thicknesses to valid range and scale to fit total constraint."""
        t = np.array(t, dtype=float)
        t = np.clip(t, 0.0, max_layer_nm)
        total = np.sum(t)
        if total > max_total_nm:
            t = t * (max_total_nm / total)
            t = np.clip(t, 0.0, max_layer_nm)
        return t

    def _objective(t):
        """Objective function with constraint handling."""
        t_clamped = _clamp_thicknesses(t)
        if np.any(t_clamped < -1e-9) or np.any(t_clamped > max_layer_nm + 1e-9):
            return 1000.0
        if np.sum(t_clamped) > max_total_nm + 1e-9:
            return 1000.0
        return _compute_reflectance(wavelengths, n0, ns, n_seq, t_clamped)

    def _optimize_thicknesses(materials_list, n_seq, initial_starts):
        """
        Optimize thicknesses for a fixed material sequence using Nelder-Mead
        with multiple starts.
        """
        N = len(materials_list)
        if N == 0:
            return None, float('inf')
        
        best_t = None
        best_val = float('inf')
        
        for start_t in initial_starts:
            if time.time() > deadline - 5.0:
                break
            
            start_t = _clamp_thicknesses(start_t)
            
            # Use Nelder-Mead for optimization
            try:
                result = minimize(
                    _objective,
                    start_t,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 0.5,
                        'fatol': 0.01,
                        'adaptive': True
                    }
                )
                
                t_opt = _clamp_thicknesses(result.x)
                val_opt = _objective(t_opt)
                
                if val_opt < best_val:
                    best_val = val_opt
                    best_t = t_opt.copy()
            except Exception:
                # If optimization fails, try the start point directly
                val_start = _objective(start_t)
                if val_start < best_val:
                    best_val = val_start
                    best_t = start_t.copy()
        
        return best_t, best_val

    # Generate candidate sequences with physical motivation
    sequences = []
    
    # Quarter-wave thickness helper
    def _qw_thicknesses(n_seq, lam0):
        return [lam0 / (4.0 * n) for n in n_seq]
    
    # 1. Alternating stacks (most physically motivated for broadband AR)
    for N in range(1, min(max_layers, 14) + 1):
        # Start with L
        seq = ["L" if i % 2 == 0 else "H" for i in range(N)]
        sequences.append(seq)
        # Start with H
        seq = ["H" if i % 2 == 0 else "L" for i in range(N)]
        sequences.append(seq)
    
    # 2. Some non-alternating patterns that might help
    # L H H L L H H L ... (period-4)
    for N in range(4, min(max_layers, 12) + 1, 2):
        seq = []
        for i in range(N):
            phase = i % 4
            if phase in [0, 1]:
                seq.append("L" if (i // 4) % 2 == 0 else "H")
            else:
                seq.append("H" if (i // 4) % 2 == 0 else "L")
        sequences.append(seq)
    
    # 3. Random sequences for diversity (limited count)
    import random
    random.seed(42)
    for N in range(4, 10):
        for _ in range(3):
            seq = [random.choice(["L", "H"]) for _ in range(N)]
            sequences.append(seq)
    
    # Deduplicate
    unique_seqs = []
    seen = set()
    for seq in sequences:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            unique_seqs.append(seq)
    
    # Sort: prioritize shorter stacks first (faster to optimize), then longer
    unique_seqs.sort(key=lambda s: len(s))
    
    # Optimize each sequence
    for seq in unique_seqs:
        if time.time() > deadline - 10.0:
            break
        
        n_seq = [nL if m == "L" else nH for m in seq]
        N = len(seq)
        
        # Generate multiple starting points
        starts = []
        
        # Quarter-wave at various center wavelengths
        for lam0 in [500.0, 550.0, 600.0, 650.0, 700.0]:
            t_qw = _qw_thicknesses(n_seq, lam0)
            if sum(t_qw) <= max_total_nm * 1.1:  # Allow slight overflow, will be clamped
                starts.append(t_qw)
        
        # Uniform thickness
        t_uniform = [max_total_nm / N] * N
        starts.append(t_uniform)
        
        # Random starts
        import random
        rng = random.Random(hash(tuple(seq)) % 10000)
        for _ in range(3):
            t_rand = [rng.uniform(10.0, max_layer_nm) for _ in range(N)]
            total = sum(t_rand)
            if total > max_total_nm:
                t_rand = [t * (max_total_nm / total) for t in t_rand]
            starts.append(t_rand)
        
        # Optimize
        t_opt, val_opt = _optimize_thicknesses(seq, n_seq, starts)
        
        if t_opt is not None and val_opt < best_score:
            if _valid_solution(seq, t_opt):
                best_score = val_opt
                best_materials = list(seq)
                best_thicknesses = [float(t) for t in t_opt]
    
    # Fallback: single quarter-wave layer
    if not best_materials:
        t0 = 600.0 / (4.0 * nL)
        if t0 <= max_layer_nm and t0 <= max_total_nm:
            best_materials = ["L"]
            best_thicknesses = [t0]
        else:
            best_materials = ["L"]
            best_thicknesses = [1.0]
    
    return best_materials, best_thicknesses
