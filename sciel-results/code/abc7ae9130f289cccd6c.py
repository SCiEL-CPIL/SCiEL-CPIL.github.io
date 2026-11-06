import numpy as np
import time
from scipy.optimize import minimize_scalar
import itertools

def _compute_reflectance(wavelengths, n0, ns, n_layers, t_layers):
    """
    Compute mean reflectance for given wavelengths and layer stack.
    wavelengths: array of wavelengths in nm
    n0: ambient index
    ns: substrate index
    n_layers: array of layer indices
    t_layers: array of layer thicknesses in nm
    """
    if len(n_layers) == 0:
        # Bare substrate
        r = (n0 - ns) / (n0 + ns)
        return np.mean(r**2) * 100.0

    N = len(n_layers)
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
    
    return np.mean(R) * 100.0

def design(problem):
    start_time = time.time()
    deadline = start_time + 50.0
    
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
    
    # Search over N up to 10
    # Generate all sequences of L/H of length N
    # For N > 10, we only try alternating to save time, or stop.
    # Given thickness constraints, N > 10 is unlikely to be optimal with QWOTs.
    
    for N in range(1, 11):
        if time.time() > deadline:
            break
            
        # Generate sequences
        if N <= 10:
            # All combinations
            seqs = list(itertools.product(['L', 'H'], repeat=N))
        else:
            # Just alternating
            seqs = [tuple('L' if i%2==0 else 'H' for i in range(N)),
                    tuple('H' if i%2==0 else 'L' for i in range(N))]
            
        for seq_tuple in seqs:
            if time.time() > deadline:
                break
                
            seq = list(seq_tuple)
            n_seq = [nL if m == 'L' else nH for m in seq]
            
            sum_inv_n = sum(1.0 / n for n in n_seq)
            max_lambda_by_total = max_total_nm / (0.25 * sum_inv_n)
            max_lambda_by_layer = 4.0 * max_layer_nm * min(n_seq)
            
            max_lambda = min(max_lambda_by_total, max_lambda_by_layer, 1000.0)
            min_lambda = 300.0
            
            if min_lambda >= max_lambda:
                continue
                
            def obj(lam, n_seq=n_seq):
                if lam <= 0:
                    return 100.0
                t_seq = [lam / (4.0 * n) for n in n_seq]
                if any(t > max_layer_nm for t in t_seq):
                    return 100.0
                if sum(t_seq) > max_total_nm:
                    return 100.0
                return _compute_reflectance(wavelengths, n0, ns, n_seq, t_seq)
            
            try:
                res = minimize_scalar(obj, bounds=(min_lambda, max_lambda), method='bounded', options={'xatol': 1e-3})
                if res.success:
                    score = res.fun
                    if score < best_score:
                        best_score = score
                        best_materials = seq
                        best_thicknesses = [res.x / (4.0 * n) for n in n_seq]
            except Exception:
                pass

    if not best_materials:
        best_materials = ["L"]
        best_thicknesses = [600.0 / (4.0 * nL)]
        
    return best_materials, best_thicknesses
