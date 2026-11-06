import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def _accurate_supnorm(coeffs):
    n = len(coeffs)
    coeffs = np.array(coeffs)
    
    N_grid = 10000
    thetas = np.linspace(0, 2 * np.pi, N_grid, endpoint=False)
    z = np.exp(1j * thetas)
    
    P_vals = np.zeros(N_grid, dtype=complex)
    z_power = np.ones(N_grid, dtype=complex)
    for k in range(n):
        P_vals += coeffs[k] * z_power
        z_power *= z
        
    abs_vals = np.abs(P_vals)
    
    K = 10
    top_indices = np.argsort(abs_vals)[-K:]
    
    max_val = 0.0
    
    window_size = 4 * 2 * np.pi / N_grid
    
    for idx in top_indices:
        theta_center = thetas[idx]
        theta_left = theta_center - window_size
        theta_right = theta_center + window_size
        
        gr = (np.sqrt(5) - 1) / 2
        a = theta_left
        b = theta_right
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        def f(theta):
            z_val = np.exp(1j * theta)
            powers = z_val ** np.arange(n)
            return np.abs(np.dot(coeffs, powers))
            
        fc = f(c)
        fd = f(d)
        
        for _ in range(20):
            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
                
        theta_max = (a + b) / 2
        val = f(theta_max)
        if val > max_val:
            max_val = val
            
    return max_val

def _approx_supnorm(coeffs, powers_grid):
    P_vals = powers_grid @ coeffs
    return np.max(np.abs(P_vals))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 24
    
    N_grid = 4096
    thetas = np.linspace(0, 2 * np.pi, N_grid, endpoint=False)
    z_grid = np.exp(1j * thetas)
    powers_grid = np.empty((N_grid, n), dtype=complex)
    powers_grid[:, 0] = 1
    for k in range(1, n):
        powers_grid[:, k] = powers_grid[:, k-1] * z_grid
        
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    seed_seq = np.array(p[:n])
    
    best_seq = seed_seq.copy()
    best_score = _accurate_supnorm(best_seq)
    
    current_seq = best_seq.copy()
    current_approx_score = _approx_supnorm(current_seq, powers_grid)
    
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            acc_score = _accurate_supnorm(current_seq)
            if acc_score < best_score:
                best_score = acc_score
                best_seq = current_seq.copy()
            
            if random.random() < 0.5:
                current_seq = np.random.choice([-1, 1], size=n)
            else:
                current_seq = best_seq.copy()
                indices = np.random.choice(n, size=5, replace=False)
                current_seq[indices] *= -1
            
            current_approx_score = _approx_supnorm(current_seq, powers_grid)
            no_improve_count = 0
            max_no_improve = 100
            
        order = list(range(n))
        random.shuffle(order)
        
        for idx in order:
            old_val = current_seq[idx]
            current_seq[idx] *= -1
            
            new_approx_score = _approx_supnorm(current_seq, powers_grid)
            
            if new_approx_score < current_approx_score:
                current_approx_score = new_approx_score
                no_improve_count = 0
            else:
                current_seq[idx] = old_val
                no_improve_count += 1
                
    if time.time() < deadline - 2.0:
        final_acc_score = _accurate_supnorm(current_seq)
        if final_acc_score < best_score:
            best_seq = current_seq.copy()
            
    return list(best_seq)
# EVOLVE-BLOCK-END
