"""
Broadband AR coating design using Multi-start Nelder-Mead on continuous thicknesses.
Strategy: 
1. Fix the number of layers N.
2. Use a heuristic to determine the material sequence (alternating L/H, starting with L).
3. Optimize the continuous thicknesses using Nelder-Mead (scipy.optimize.minimize).
4. Perform multi-starts with different initial thickness distributions.
"""
import numpy as np
import time
import sys
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _reflectance_calc(mats, thicks, n0, ns, materials, lambdas):
    """
    Fast vectorized reflectance calculation.
    mats: list of 'L'/'H'
    thicks: numpy array of thicknesses
    """
    N = len(thicks)
    if N == 0:
        r = (n0 - ns) / (n0 + ns)
        return np.full_like(lambdas, np.abs(r)**2)
        
    # Start with identity matrix components
    M11 = np.ones(len(lambdas), dtype=complex)
    M12 = np.zeros(len(lambdas), dtype=complex)
    M21 = np.zeros(len(lambdas), dtype=complex)
    M22 = np.ones(len(lambdas), dtype=complex)
    
    for i in range(N):
        n_i = materials[mats[i]]
        d_i = thicks[i]
        
        delta = 2.0 * np.pi * n_i * d_i / lambdas
        c = np.cos(delta)
        s = np.sin(delta)
        
        m11 = c
        m12 = 1j * s / n_i
        m21 = 1j * n_i * s
        m22 = c
        
        # Matrix multiplication
        new_M11 = M11 * m11 + M12 * m21
        new_M12 = M11 * m12 + M12 * m22
        new_M21 = M21 * m11 + M22 * m21
        new_M22 = M21 * m12 + M22 * m22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
        
    Y_s = ns
    Y_in = (M21 + M22 * Y_s) / (M11 + M12 * Y_s)
    r = (Y_in - n0) / (Y_in + n0)
    R = np.abs(r)**2
    return R

def design(problem):
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s budget
    
    band = problem["band_nm"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lambdas = np.linspace(band[0], band[1], 401)
    
    # Pre-compute material indices for speed
    n_L = materials['L']
    n_H = materials['H']
    
    # Objective function for Nelder-Mead
    # x: array of thicknesses
    # mats: fixed sequence of materials
    def objective(x, mats):
        # Penalty for constraint violations
        # Total thickness constraint
        total_t = np.sum(x)
        if total_t > max_total_nm:
            # Strong penalty
            return 100.0 + (total_t - max_total_nm) * 10.0
        
        # Individual layer constraint
        if np.any(x > max_layer_nm) or np.any(x < 0):
            return 100.0 + np.sum(np.abs(x - np.clip(x, 0, max_layer_nm))) * 10.0
        
        R = _reflectance_calc(mats, x, n0, ns, materials, lambdas)
        return np.mean(R) * 100.0

    best_score = np.inf
    best_mats = []
    best_thicks = np.array([])
    
    # Heuristic material sequences
    # Typically, for broadband AR, alternating L/H is good.
    # Start with L (closer to air) or H (closer to glass)?
    # Air (1.0) -> L (1.38) is a smaller step than Air -> H (2.35).
    # Glass (1.52) -> L (1.38) is a smaller step than Glass -> H (2.35).
    # So L...L or H...H at boundaries might be good, but alternating is standard for interference.
    # Let's try a few sequence patterns for each N.
    
    def get_sequences(N):
        seqs = []
        # Pattern 1: Start with L, alternate
        seq1 = ['L' if i % 2 == 0 else 'H' for i in range(N)]
        seqs.append(seq1)
        
        # Pattern 2: Start with H, alternate
        seq2 = ['H' if i % 2 == 0 else 'L' for i in range(N)]
        seqs.append(seq2)
        
        # Pattern 3: All L (if N is small, maybe a single high-index layer isn't optimal, but all L might be)
        # Actually, for AR, we need index contrast. All L or All H is bad.
        # Pattern 4: L...L H...H (block)
        half = N // 2
        seq3 = ['L'] * half + ['H'] * (N - half)
        seqs.append(seq3)
        
        seq4 = ['H'] * half + ['L'] * (N - half)
        seqs.append(seq4)
        
        return seqs

    # Try different numbers of layers
    # The optimal N is likely between 5 and 15.
    # We will iterate over N and run Nelder-Mead for a few initial guesses.
    
    # Initial guess generation
    def generate_initial_guesses(N, total_nm, mats):
        guesses = []
        # 1. Quarter-wave at center wavelength
        center_lambda = 0.5 * (band[0] + band[1])
        qw_L = center_lambda / (4.0 * n_L)
        qw_H = center_lambda / (4.0 * n_H)
        
        x_qw = np.array([qw_L if m == 'L' else qw_H for m in mats])
        # Scale to fit total nm if necessary, but QW might be small
        # Total QW thickness is usually around N * 100nm = 1000-2000nm.
        # Our limit is 1000nm. So we might need to scale down.
        if np.sum(x_qw) > total_nm:
            x_qw = x_qw * (total_nm / np.sum(x_qw))
        guesses.append(x_qw)
        
        # 2. Uniform thickness
        x_unif = np.full(N, total_nm / N)
        guesses.append(x_unif)
        
        # 3. Random (with seed for reproducibility in logic, but we want variety)
        # Use a few fixed random-like patterns
        for i in range(3):
            x_rand = np.random.uniform(0, total_nm / N, N)
            x_rand *= (total_nm / np.sum(x_rand)) if np.sum(x_rand) > 0 else 1.0
            guesses.append(x_rand)
            
        return guesses

    # We will test N from 3 to 15.
    # To save time, we will run Nelder-Mead with a limited number of iterations.
    # Nelder-Mead is local, so multi-start is key.
    
    # Budget allocation:
    # ~58 seconds.
    # 13 values of N (3..15).
    # ~4 seconds per N.
    # Within each N, try 2-3 sequences and 2-3 initial guesses.
    # That's ~18 optimizations per N.
    # Each optimization should be fast (<0.2s).
    
    for N in range(3, 16):
        if time.time() > deadline - 5.0:
            break
            
        seqs = get_sequences(N)
        
        for seq in seqs:
            if time.time() > deadline - 5.0:
                break
                
            guesses = generate_initial_guesses(N, max_total_nm, seq)
            
            for x0 in guesses:
                if time.time() > deadline - 5.0:
                    break
                
                # Ensure x0 is valid
                x0 = np.clip(x0, 0, max_layer_nm)
                if np.sum(x0) > max_total_nm:
                    x0 *= (max_total_nm / np.sum(x0))
                
                try:
                    # Nelder-Mead options
                    # maxiter: 200 is usually enough for 15D
                    # xatol, fatol: tight enough for nm precision
                    res = minimize(
                        objective, 
                        x0, 
                        args=(seq,), 
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-8, 'adaptive': True}
                    )
                    
                    if res.success or res.fun < best_score:
                        x_opt = res.x
                        # Clean up: clip and handle small negatives
                        x_opt = np.clip(x_opt, 0, max_layer_nm)
                        
                        # Check constraints strictly
                        if np.sum(x_opt) <= max_total_nm + 1e-9:
                            score = objective(x_opt, seq)
                            if score < best_score:
                                best_score = score
                                best_mats = seq
                                best_thicks = x_opt
                except Exception:
                    pass

    # If no valid solution found (shouldn't happen), return a simple valid one
    if len(best_mats) == 0:
        # Simple 1-layer AR
        N = 1
        seq = ['L']
        # Quarter wave at 600nm
        d = 600.0 / (4.0 * n_L)
        best_mats = seq
        best_thicks = np.array([d])
        best_score = objective(best_thicks, best_mats)

    # Final check and return
    # Ensure types are correct
    return best_mats, list(best_thicks)

# EVOLVE-BLOCK-END
