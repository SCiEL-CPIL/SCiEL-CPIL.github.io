"""
Broadband AR coating design using Simulated Annealing over joint material-thickness space.
"""
import numpy as np
import time
import sys

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
        
    # Precompute sin/cos for all layers and wavelengths
    # To save memory/time, we process layer by layer but vectorize over wavelengths.
    # M_total = M1 * M2 * ... * MN
    # Start with identity
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
    deadline = start_time + 55.0  # Safety margin
    
    band = problem["band_nm"]
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    lambdas = np.linspace(band[0], band[1], 401)
    
    # Helper to evaluate a specific state
    def evaluate(mats, thicks):
        if len(mats) == 0:
            return np.mean(_reflectance_calc(mats, np.array([]), n0, ns, materials, lambdas)) * 100.0
        return np.mean(_reflectance_calc(mats, thicks, n0, ns, materials, lambdas)) * 100.0

    # Initial best found
    best_score = np.inf
    best_mats = []
    best_thicks = np.array([])
    
    # Multi-start Simulated Annealing
    # We will run several independent SA chains with different initial depths and sequences.
    # Time budget is split among them.
    
    num_starts = 5
    time_per_start = (deadline - start_time) / num_starts
    
    # Pre-define some good heuristic starting points
    center_lambda = 0.5 * (band[0] + band[1])
    qw_L = center_lambda / (4.0 * materials['L'])
    qw_H = center_lambda / (4.0 * materials['H'])
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 2.0:
            break
            
        # Determine initial depth and sequence
        # Vary depth: 5, 8, 10, 12, 15
        depths = [5, 8, 10, 12, 15]
        depth = depths[start_idx % len(depths)]
        
        # Randomize starting sequence, but bias towards alternating L/H
        # Start with L (low index) to match air better
        current_mats = []
        current_thicks = np.zeros(depth)
        
        for i in range(depth):
            if np.random.rand() < 0.8:
                # Alternating pattern
                if i % 2 == 0:
                    current_mats.append('L')
                    current_thicks[i] = qw_L * np.random.uniform(0.8, 1.2)
                else:
                    current_mats.append('H')
                    current_thicks[i] = qw_H * np.random.uniform(0.8, 1.2)
            else:
                # Random
                current_mats.append('L' if np.random.rand() < 0.5 else 'H')
                current_thicks[i] = np.random.uniform(10, max_layer_nm)
        
        # Ensure constraints
        total_t = np.sum(current_thicks)
        if total_t > max_total_nm:
            current_thicks *= (max_total_nm / total_t)
        
        current_score = evaluate(current_mats, current_thicks)
        
        if current_score < best_score:
            best_score = current_score
            best_mats = current_mats.copy()
            best_thicks = current_thicks.copy()
            
        # Simulated Annealing Loop
        T = 10.0  # Initial temperature
        T_min = 0.01
        alpha = 0.98  # Cooling rate
        
        # Calculate max steps based on time
        # We don't know exact step time, so we run until time runs out for this start
        local_deadline = time.time() + time_per_start
        
        while time.time() < local_deadline and T > T_min:
            # Generate neighbor
            neighbor_mats = current_mats.copy()
            neighbor_thicks = current_thicks.copy()
            
            # Mutation: 1. Change thickness, 2. Change material, 3. Swap materials
            mut_type = np.random.rand()
            
            if mut_type < 0.7:
                # Perturb a random thickness
                idx = np.random.randint(0, depth)
                delta = np.random.normal(0, 20)
                neighbor_thicks[idx] += delta
                neighbor_thicks[idx] = np.clip(neighbor_thicks[idx], 0, max_layer_nm)
            elif mut_type < 0.9:
                # Change material of a random layer
                idx = np.random.randint(0, depth)
                neighbor_mats[idx] = 'L' if neighbor_mats[idx] == 'H' else 'H'
            else:
                # Swap two layers (both material and thickness)
                if depth > 1:
                    i, j = np.random.choice(depth, 2, replace=False)
                    neighbor_mats[i], neighbor_mats[j] = neighbor_mats[j], neighbor_mats[i]
                    neighbor_thicks[i], neighbor_thicks[j] = neighbor_thicks[j], neighbor_thicks[i]
            
            # Check constraints
            if np.sum(neighbor_thicks) > max_total_nm:
                # Scale down to satisfy constraint (penalty approach alternative)
                scale = max_total_nm / np.sum(neighbor_thicks)
                neighbor_thicks *= scale
            
            neighbor_score = evaluate(neighbor_mats, neighbor_thicks)
            
            # Acceptance criterion
            delta_E = neighbor_score - current_score
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                current_mats = neighbor_mats
                current_thicks = neighbor_thicks
                current_score = neighbor_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_mats = current_mats.copy()
                    best_thicks = current_thicks.copy()
            
            T *= alpha
    
    # Final Polish: Local Nelder-Mead or simple gradient-free refinement on best solution
    if time.time() < deadline - 5.0 and len(best_mats) > 0:
        # Use a simple local search with small steps
        polish_deadline = time.time() + (deadline - time.time()) * 0.5
        step_size = 5.0
        current_mats = best_mats.copy()
        current_thicks = best_thicks.copy()
        current_score = best_score
        
        while time.time() < polish_deadline:
            improved = False
            for i in range(len(current_thicks)):
                # Try small steps in thickness
                for delta in [-step_size, step_size]:
                    trial_thicks = current_thicks.copy()
                    trial_thicks[i] += delta
                    trial_thicks[i] = np.clip(trial_thicks[i], 0, max_layer_nm)
                    
                    if np.sum(trial_thicks) > max_total_nm:
                        continue
                    
                    trial_score = evaluate(current_mats, trial_thicks)
                    if trial_score < current_score:
                        current_thicks = trial_thicks
                        current_score = trial_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_thicks = current_thicks.copy()
                            best_mats = current_mats.copy()
            
            if not improved:
                step_size *= 0.8
                if step_size < 0.1:
                    break

    # Return best result
    # Ensure types are correct: list of strings, list of floats
    return best_mats, list(best_thicks)

# EVOLVE-BLOCK-END
