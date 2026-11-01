"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _max_sidelobe(h):
    """Compute max |autocorrelation| at non-zero lags using FFT."""
    n = len(h)
    # Use next power of 2 for FFT efficiency
    n_fft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(h.astype(np.float64), n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:n]
    # Exclude zero lag
    return np.max(np.abs(acf[1:]))

def _continuous_objective(z, p=10):
    """
    Objective function for continuous relaxation.
    z: real-valued vector of length N
    Returns: L_p norm of the non-zero autocorrelation sidelobes.
    Using a high p approximates the max-norm (peak sidelobe) while remaining differentiable.
    """
    h = np.tanh(z)
    n = len(h)
    n_fft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:n]
    sidelobes = acf[1:]
    # L_p norm
    return np.sum(np.abs(sidelobes) ** p) ** (1.0 / p)

def _continuous_gradient(z, p=10):
    """
    Numerical gradient for the continuous objective.
    Since deriving the analytic gradient through FFT and tanh is complex,
    we use finite differences if analytic is too slow, but L-BFGS-B can also
    use its own numerical gradients. However, providing an analytic or semi-analytic
    gradient is faster. 
    For simplicity and robustness in this constrained time, we will let L-BFGS-B 
    compute numerical gradients, but we can speed it up by using a smaller p 
    or just relying on the smoothness of tanh.
    
    Actually, let's just use L-BFGS-B with numerical differentiation. It's robust.
    """
    # Return None to let L-BFGS-B use numerical gradient
    return None

def _discrete_local_search(h, deadline, max_iter=500):
    """
    Fast discrete local search using single-bit flips.
    """
    n = len(h)
    h = h.copy()
    current_val = _max_sidelobe(h)
    best_val = current_val
    best_h = h.copy()
    rng = random.Random(42)
    
    # Sort indices randomly to avoid bias
    indices = list(range(n))
    rng.shuffle(indices)
    
    start_time = time.time()
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_i = -1
        best_new_val = current_val
        
        # Iterate through a subset of indices or all if time permits
        # For N=400, full pass is 400 FFTs. 
        # FFT size ~1024. 400 * 1024 * log(1024) ~ 400 * 10k ops ~ 4M ops. 
        # This is fast in numpy (< 0.1s per pass).
        for i in indices:
            if time.time() > deadline:
                break
            
            h[i] = -h[i]
            val = _max_sidelobe(h)
            h[i] = -h[i]
            
            if val < best_new_val:
                best_new_val = val
                best_i = i
        
        if best_i >= 0 and best_new_val < current_val:
            h[best_i] = -h[best_i]
            current_val = best_new_val
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
            improved = True
            # Re-shuffle indices to prevent cyclic patterns
            rng.shuffle(indices)
            
    return best_h, best_val

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, aim to finish in 140s
    deadline = start_time + 140.0
    
    np.random.seed(42)
    
    # Phase 1: Continuous Relaxation
    # Initialize z with small random values
    z0 = np.random.randn(N) * 0.1
    
    # Use L-BFGS-B to minimize the L_p norm of sidelobes
    # p=10 is a good balance between smoothness and peak minimization
    p_val = 10
    
    # We need to be careful with time. L-BFGS-B can be slow.
    # Set a strict time limit for this phase.
    opt_deadline = start_time + 60.0
    
    # L-BFGS-B does not have a direct time stop, but we can limit iterations
    # or use a callback to break.
    # Let's try a few iterations.
    
    try:
        # Using numerical gradient is slow. Let's try to provide a simple 
        # analytic-like gradient or just let it run for a limited number of steps.
        # Actually, for N=400, numerical gradient requires 2*N evaluations of the objective.
        # Objective is one FFT. 800 FFTs per gradient step. 
        # If we do 100 steps, that's 80,000 FFTs. 
        # 1000-point FFT is ~10k flops. 800M flops. Should be < 1s in numpy? 
        # Maybe 5-10s. Let's try 50 iterations.
        
        res = minimize(
            _continuous_objective, 
            z0, 
            method='L-BFGS-B', 
            args=(p_val,),
            options={'maxiter': 50, 'maxfun': 5000}
        )
        
        z_opt = res.x
    except Exception as e:
        # Fallback to random z if optimization fails
        z_opt = z0
        print(f"Optimization error: {e}")
        
    # Check time
    if time.time() > opt_deadline:
        print("Optimization timed out, using partial result or initial")
        # If it timed out, we might have a bad z. 
        # But minimize usually returns the best found.
        
    # Phase 2: Discretization
    # Map z to {-1, 1}
    h_discrete = np.sign(z_opt)
    # Handle zeros (rare)
    h_discrete[h_discrete == 0] = 1.0
    
    # Phase 3: Discrete Local Search Polish
    # This will fix any issues from the tanh mapping and local minima
    polish_deadline = deadline - 2.0
    h_final, val_final = _discrete_local_search(h_discrete, polish_deadline)
    
    # Phase 4: Additional Random Restarts with Local Search if time permits
    # If the continuous relaxation didn't yield a great result, try some discrete restarts
    if time.time() < deadline - 10.0:
        # Try 2 more random restarts
        for _ in range(2):
            if time.time() > deadline - 5.0:
                break
            h_rand = np.random.choice([-1.0, 1.0], size=N).astype(np.float64)
            h_try, val_try = _discrete_local_search(h_rand, deadline - 5.0)
            if val_try < val_final:
                h_final = h_try
                val_final = val_try
                
    return h_final
# EVOLVE-BLOCK-END
