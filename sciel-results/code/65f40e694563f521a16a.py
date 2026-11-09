import numpy as np
import time
import random

# Precompute the evaluation grid
N_POINTS = 4096
THETAS = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)
K = np.arange(24)
# Shape (24, N_POINTS)
M = np.exp(1j * np.outer(K, THETAS))

def sup_norm(coeffs):
    """
    Calculate the sup norm of the polynomial with given coefficients.
    coeffs: list of 24 integers (+1 or -1)
    """
    c = np.array(coeffs, dtype=np.complex128)
    # Evaluate on grid
    vals = M @ c
    mags = np.abs(vals)
    
    # Find the index of the maximum
    max_idx = np.argmax(mags)
    max_val = mags[max_idx]
    
    # Refine the peak using golden section search
    step = 2 * np.pi / N_POINTS
    theta_center = THETAS[max_idx]
    
    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1 / phi
    
    def f(theta):
        exps = np.exp(1j * K * theta)
        val = np.dot(c, exps)
        return np.abs(val)
    
    a = theta_center - step
    b = theta_center + step
    
    if a < 0: a += 2 * np.pi
    if b > 2 * np.pi: b -= 2 * np.pi
    
    x1 = b - inv_phi * (b - a)
    x2 = a + inv_phi * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    
    for _ in range(50):
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - inv_phi * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + inv_phi * (b - a)
            f2 = f(x2)
            
    theta_opt = (a + b) / 2
    max_val_refined = f(theta_opt)
    
    return max_val_refined

def construct():
    # Seed program: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    seed_coeffs = p[:24]
    
    try:
        seed_max = sup_norm(seed_coeffs)
        seed_score = seed_max / np.sqrt(24)
        
        best_coeffs = seed_coeffs[:]
        best_max = seed_max
        best_score = seed_score
        
        current_coeffs = seed_coeffs[:]
        current_max = seed_max
        current_score = seed_score
        
        start_time = time.time()
        timeout = 140.0
        
        T = 0.05
        T_min = 1e-5
        cooling_rate = 0.9995
        
        random.seed(42)
        
        while time.time() - start_time < timeout:
            idx = random.randint(0, 23)
            neighbor = current_coeffs[:]
            neighbor[idx] = -neighbor[idx]
            
            neighbor_max = sup_norm(neighbor)
            neighbor_score = neighbor_max / np.sqrt(24)
            
            delta = neighbor_score - current_score
            if delta < 0 or (T > T_min and random.random() < np.exp(-delta / T)):
                current_coeffs = neighbor
                current_max = neighbor_max
                current_score = neighbor_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_max = current_max
                    best_coeffs = current_coeffs[:]
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
    except Exception:
        # Fallback to seed if anything goes wrong
        pass

    return best_coeffs
