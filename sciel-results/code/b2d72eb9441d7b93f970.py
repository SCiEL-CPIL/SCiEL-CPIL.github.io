import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 115.0  # Safety margin under 120s limit

def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length 2*N.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac

def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes.
    """
    return np.max(np.abs(ac[1:N]))

def _soft_max_sidelobe(ac, beta=50.0):
    """
    Smooth approximation of max |ac[1:N]| using log-sum-exp.
    """
    sidelobes = np.abs(ac[1:N])
    max_val = np.max(sidelobes)
    shifted = beta * (sidelobes - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta

def _objective_and_gradient(x, beta=50.0):
    """
    Objective: Smooth approximation of max sidelobe.
    Gradient: Analytic.
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes_abs = np.abs(ac[1:N])
    
    max_val = np.max(sidelobes_abs)
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    weights = beta * exp_shifted / sum_exp
    
    grad = np.zeros(N)
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k == 0:
            continue
        
        sign_k = np.sign(ac[k])
        c_k = sign_k * w_k
        
        if c_k == 0:
            continue
            
        grad[0:N-k] += c_k * h[k:N]
        grad[k:N] += c_k * h[0:N-k]
        
    return obj, grad

def _local_search(h, max_iterations=10000, time_limit=5.0):
    """
    Fast local search using FFT for evaluation.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        i = np.random.randint(N)
        old_h_i = h[i]
        h[i] = -old_h_i
        
        ac_new = _compute_autocorr(h)
        new_val = _max_sidelobe(ac_new)
        
        if new_val < current_val:
            current_val = new_val
            ac = ac_new
        else:
            h[i] = old_h_i
            
    return h, current_val

def _parallel_tempering_sa(h, n_chains=3, start_temp=1.0, end_temp=0.01, 
                           iterations=5000, time_limit=15.0):
    """
    Parallel tempering simulated annealing to escape local minima.
    Multiple chains at different temperatures swap states periodically.
    """
    start_time = time.time()
    
    # Initialize chains
    chains = [h.copy() for _ in range(n_chains)]
    chain_vals = []
    for i in range(n_chains):
        ac = _compute_autocorr(chains[i])
        chain_vals.append(_max_sidelobe(ac))
    
    # Temperatures for each chain (higher index = hotter)
    temps = np.linspace(start_temp, end_temp, n_chains)
    
    best_h = h.copy()
    best_val = min(chain_vals)
    
    swap_interval = 50  # Swap every 50 iterations
    
    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Update each chain
        for c in range(n_chains):
            # Try flipping a random bit
            idx = np.random.randint(N)
            old_h_i = chains[c][idx]
            chains[c][idx] = -old_h_i
            
            ac_new = _compute_autocorr(chains[c])
            new_val = _max_sidelobe(ac_new)
            
            delta = new_val - chain_vals[c]
            
            # Metropolis criterion
            if delta < 0 or np.random.random() < np.exp(-delta / max(temps[c], 1e-10)):
                chain_vals[c] = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_h = chains[c].copy()
            else:
                chains[c][idx] = old_h_i
        
        # Periodic swap between adjacent chains
        if it % swap_interval == 0 and it > 0:
            for c in range(n_chains - 1):
                # Swap chains c and c+1 with probability based on energy difference
                delta_e = chain_vals[c+1] - chain_vals[c]
                swap_prob = np.exp(-delta_e / max(temps[c+1], 1e-10))
                if np.random.random() < swap_prob:
                    chains[c], chains[c+1] = chains[c+1], chains[c]
                    chain_vals[c], chain_vals[c+1] = chain_vals[c+1], chain_vals[c]
    
    return best_h, best_val

def _gradient_descent_step(h, lr=0.1, beta=50.0, steps=100):
    """
    Perform gradient descent on the continuous relaxation, then round.
    """
    x = h.astype(float)
    for _ in range(steps):
        obj, grad = _objective_and_gradient(x, beta)
        x = x - lr * grad
        x = np.clip(x, -1.0, 1.0)
    
    h_bin = np.sign(x)
    zero_mask = (h_bin == 0)
    if np.any(zero_mask):
        h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
    
    return h_bin

def _gray_code_init():
    """
    Generate a balanced Gray code pattern for structured initialization.
    This creates a sequence where adjacent elements differ by one bit,
    providing a diverse but structured starting point.
    """
    h = np.zeros(N)
    # Generate Gray code sequence
    for i in range(N):
        gray = i ^ (i >> 1)
        # Convert Gray code to binary-like pattern
        bits = np.array([(gray >> j) & 1 for j in range(8)])
        # Map to signed values: 0 -> -1, 1 -> 1
        h[i] = 2.0 * np.sum(bits) - 4.0
        h[i] = np.sign(h[i]) if h[i] != 0 else 1.0
    # Ensure balance
    if np.sum(h) > 0:
        h[np.random.choice(N, size=int(np.sum(h) - N/2), replace=False)] = -1.0
    elif np.sum(h) < 0:
        h[np.random.choice(N, size=int(-np.sum(h) - N/2), replace=False)] = 1.0
    return h

def construct():
    start_time = time.time()
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Structured initializations with continuous relaxation
    # Use Gray code pattern and random starts
    num_starts = 8
    betas = [20.0, 50.0, 100.0, 200.0]
    
    # First start with Gray code pattern
    h_gray = _gray_code_init()
    ac_gray = _compute_autocorr(h_gray)
    val_gray = _max_sidelobe(ac_gray)
    if val_gray < best_val:
        best_val = val_gray
        best_h = h_gray.copy()
    
    # Continue with random starts and continuous optimization
    for i in range(num_starts):
        if time.time() >= DEADLINE - 35.0:
            break
        
        # Alternate between random and structured initializations
        if i == 0:
            x0 = h_gray.astype(float)
        elif i % 3 == 0:
            # Block pattern
            block_size = N // 4
            x0 = np.tile(np.array([1.0, -1.0, -1.0, 1.0]), N // 4)[:N]
        elif i % 3 == 1:
            # Alternating
            x0 = np.tile(np.array([1.0, -1.0]), N // 2)[:N]
        else:
            x0 = np.random.choice([-1.0, 1.0], size=N)
        
        beta = betas[i % len(betas)]
        
        def obj_grad(x, beta=beta):
            return _objective_and_gradient(x, beta)
        
        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 150, 'ftol': 1e-9, 'gtol': 1e-7, 'maxls': 20})
        except Exception:
            continue
        
        h_bin = np.sign(res.x)
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
            
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
    
    # Phase 2: Parallel tempering simulated annealing
    if best_h is not None and time.time() < DEADLINE - 20.0:
        h_pt, val_pt = _parallel_tempering_sa(best_h.copy(), n_chains=3, 
                                              start_temp=0.8, end_temp=0.01,
                                              iterations=8000, time_limit=12.0)
        if val_pt < best_val:
            best_val = val_pt
            best_h = h_pt
    
    # Phase 3: Final local search polish
    remaining_time = DEADLINE - time.time()
    if remaining_time > 2.0 and best_h is not None:
        h_final, val_final = _local_search(best_h.copy(), max_iterations=50000, 
                                           time_limit=remaining_time - 1.0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
