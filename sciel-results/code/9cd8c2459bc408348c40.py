import numpy as np
import time
from scipy.optimize import minimize

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    # Initial guess: Zeros of Laguerre polynomial L_12
    # numpy.polynomial.laguerre.laggauss(n) returns zeros and weights
    # The zeros are the roots of L_n(x)
    try:
        x0, w = np.polynomial.laguerre.laggauss(12)
        x0 = np.sort(x0)
    except Exception:
        x0 = np.linspace(1.0, 20.0, 12)
        
    # Ensure positive
    x0 = np.abs(x0) + 1e-6

    # Define the Calogero-Sutherland Energy
    # V(x) = sum x_i + sum 1/(x_i - x_j)^2
    # This proxy is based on the hint about the Calogero model and Laguerre weight (linear potential).
    def calogero_energy(x):
        x = np.sort(x)
        # Repulsive term
        repulsive = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                diff = x[i] - x[j]
                if diff == 0:
                    return 1e10
                repulsive += 1.0 / (diff * diff)
        # Confining term (Laguerre weight x * exp(-x) suggests linear potential for the 'center of mass' or similar)
        # Actually, the ground state of the Calogero model with potential sum x_i^2 (Hermite) or sum x_i (Laguerre)
        # Let's use sum x_i as the confining potential for Laguerre.
        confining = np.sum(x)
        return repulsive + confining

    def calogero_grad(x):
        x = np.sort(x)
        grad = np.zeros(12)
        for i in range(12):
            grad[i] = 1.0 # derivative of sum x_i
            for j in range(12):
                if i == j: continue
                diff = x[i] - x[j]
                if diff == 0: continue
                grad[i] -= 2.0 / (diff**3)
        return grad

    best_x = x0.copy()
    best_energy = calogero_energy(best_x)
    
    # Optimization loop
    # We can try multiple starting points or just one good one.
    # Since we have time, we can try a few random perturbations or different initializations.
    
    inits = [x0]
    # Add a perturbed version
    inits.append(x0 + np.random.normal(0, 0.1, 12))
    # Add the seed roots (just in case they are local minima of the actual score, though unlikely)
    inits.append(np.array([4.0 * (i + 1) for i in range(12)]))
    
    for init in inits:
        if time.time() > deadline:
            break
            
        # Ensure positivity
        init = np.abs(init) + 1e-6
        
        result = minimize(
            calogero_energy, 
            init, 
            method='L-BFGS-B', 
            bounds=[(1e-6, None)] * 12,
            options={'maxiter': 1000, 'maxfun': 10000, 'disp': False}
        )
        
        if result.success and result.fun < best_energy:
            best_energy = result.fun
            best_x = result.x
            
        # Check time
        if time.time() > deadline:
            break

    # Final refinement with high precision if time permits
    if time.time() < deadline - 10:
        result = minimize(
            calogero_energy, 
            best_x, 
            method='BFGS', 
            jac=calogero_grad,
            bounds=[(1e-6, None)] * 12,
            options={'maxiter': 500, 'disp': False}
        )
        if result.success and result.fun < best_energy:
            best_x = result.x

    return [float(x) for x in best_x]
