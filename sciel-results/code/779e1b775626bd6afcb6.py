"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import time

N = 400

# EVOLVE-BLOCK-START

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 120.0  # 120s limit, keep margin
    
    # Parameters:
    # alpha: chirp rate
    # phi: chirp phase
    # c1..c4: harmonic amplitudes
    # b1..b4: harmonic frequencies
    # psi1..psi4: harmonic phases
    # A: amplitude scaling
    # mu: DC offset
    
    # We want to minimize V(h)
    # h = A * (cos(chirp) + sum harmonics) + mu
    
    def h_from_params(params):
        alpha, phi = params[0], params[1]
        c = params[2:6]
        b = params[6:10]
        psi = params[10:14]
        A, mu = params[14], params[15]
        
        k = np.arange(N)
        # Chirp
        chirp = np.cos(2 * np.pi * alpha * k**2 / N + phi)
        
        # Harmonics
        harm = np.zeros(N)
        for i in range(4):
            harm += c[i] * np.cos(2 * np.pi * b[i] * k / N + psi[i])
        
        h = A * (chirp + harm) + mu
        return h

    def objective(params):
        h = h_from_params(params)
        S = np.sum(h)
        if abs(S) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2 * N * max_conv / (S**2))
        return val

    # Bounds for DE
    # alpha: [0, 10]
    # phi: [0, 2pi]
    # c: [-1, 1]
    # b: [0, 10]
    # psi: [0, 2pi]
    # A: [0.1, 10]
    # mu: [-10, 10]
    
    bounds = [
        (0, 10),      # alpha
        (0, 2*np.pi), # phi
        (-1, 1),      # c1
        (-1, 1),      # c2
        (-1, 1),      # c3
        (-1, 1),      # c4
        (0, 10),      # b1
        (0, 10),      # b2
        (0, 10),      # b3
        (0, 10),      # b4
        (0, 2*np.pi), # psi1
        (0, 2*np.pi), # psi2
        (0, 2*np.pi), # psi3
        (0, 2*np.pi), # psi4
        (0.1, 10),    # A
        (-10, 10)     # mu
    ]
    
    # Initial best is seed
    best_h = np.ones(N)
    best_val = 2.0
    
    # Run DE if time permits
    # DE is stochastic, so we run it for a limited time or iterations
    # We use a wrapper to check time
    
    def de_objective_with_time(params):
        if time.time() > deadline - 5:
            return 1e10
        return objective(params)
        
    try:
        # Use a small population and few iterations to fit in time
        # 16 dims
        res = differential_evolution(
            de_objective_with_time, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=1e-5, 
            seed=42,
            polish=False
        )
        
        if time.time() < deadline - 5:
            current_val = objective(res.x)
            if current_val < best_val:
                best_val = current_val
                best_h = h_from_params(res.x)
                
    except Exception as e:
        pass

    # Final check
    if time.time() < deadline - 2:
        # Try a few random perturbations of the best found
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            # Simple local search or random restart
            pass
            
    return best_h

# EVOLVE-BLOCK-END
