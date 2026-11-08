import numpy as np
import time

N = 400
DEADLINE = 120.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    The signed step function has N equal intervals on [-1/4, 1/4].
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    M = 4 * N  # fine grid
    dx = (0.5) / M  # domain length is 0.5
    
    # Build f on fine grid using vectorized repeat
    f_fine = np.repeat(signed_step, M // N)
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # The autoconvolution is defined on [-0.5, 0.5]
    # C3 = max |autoconv|
    c3 = np.max(np.abs(autoconv))
    
    return c3

def _structured_initialization():
    """
    Create a structured initial step function that likely has lower C3 than all-ones.
    Try several patterns and return the best one.
    """
    candidates = []
    
    # All ones (seed)
    candidates.append(np.ones(N, dtype=np.int8))
    
    # Alternating
    alt = np.ones(N, dtype=np.int8)
    alt[1::2] = -1
    candidates.append(alt)
    
    # Block pattern: + - + - in blocks of N/4
    block = np.zeros(N, dtype=np.int8)
    quarter = N // 4
    block[0:quarter] = 1
    block[quarter:2*quarter] = -1
    block[2*quarter:3*quarter] = 1
    block[3*quarter:N] = -1
    candidates.append(block)
    
    # Smooth cosine-like: sign of cos(2*pi*t)
    t = np.linspace(-0.25, 0.25, N, endpoint=False)
    cos_sign = np.sign(np.cos(2 * np.pi * t))
    cos_sign[cos_sign == 0] = 1
    candidates.append(cos_sign.astype(np.int8))
    
    # Two transitions: + - +
    two_trans = np.ones(N, dtype=np.int8)
    two_trans[N//4 : 3*N//4] = -1
    candidates.append(two_trans)
    
    # Three transitions
    three_trans = np.ones(N, dtype=np.int8)
    three_trans[N//6 : N//2] = -1
    three_trans[5*N//6 : N] = -1
    candidates.append(three_trans)
    
    # Random with fixed seed
    np.random.seed(123)
    rand1 = np.random.choice([-1, 1], size=N).astype(np.int8)
    candidates.append(rand1)
    
    np.random.seed(456)
    rand2 = np.random.choice([-1, 1], size=N).astype(np.int8)
    candidates.append(rand2)
    
    # Evaluate all candidates
    best_step = None
    best_c3 = np.inf
    for cand in candidates:
        c3 = _compute_c3(cand)
        if c3 < best_c3:
            best_c3 = c3
            best_step = cand.copy()
    
    return best_step, best_c3

def _simulated_annealing(step, c3, start_time, deadline_margin=10.0):
    """
    Run simulated annealing to improve the step function.
    Accepts uphill moves with probability exp(-delta/T).
    """
    current_step = step.copy()
    current_c3 = c3
    best_step = step.copy()
    best_c3 = c3
    
    # Temperature schedule
    T_start = 0.5
    T_end = 0.01
    total_time = deadline_margin
    start = time.time()
    
    np.random.seed(42)
    
    # Pre-compute for speed
    n_iterations = 0
    
    while time.time() - start < deadline_margin:
        # Adaptive temperature: linear cooling
        elapsed = time.time() - start
        progress = min(elapsed / total_time, 1.0)
        T = T_start * (T_end / T_start) ** progress
        
        # Choose move size: mostly 1 flip, sometimes 2
        r = np.random.random()
        if r < 0.6:
            n_flips = 1
        elif r < 0.9:
            n_flips = 2
        else:
            n_flips = 3
        
        # Select random indices to flip
        idxs = np.random.choice(N, size=n_flips, replace=False)
        old_vals = current_step[idxs].copy()
        current_step[idxs] = -current_step[idxs]
        
        new_c3 = _compute_c3(current_step)
        delta = new_c3 - current_c3
        
        # Acceptance criterion
        if delta < 0 or (T > 0 and np.random.random() < np.exp(-delta / T)):
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            current_step[idxs] = old_vals
        
        n_iterations += 1
        
        # Periodically try a larger perturbation if stuck
        if n_iterations % 1000 == 0:
            # Check if we've been stuck for a while
            if current_c3 >= best_c3 * (1 - 1e-6):
                # Do a random restart from best
                if np.random.random() < 0.1:
                    n_perturb = np.random.randint(10, 30)
                    idxs = np.random.choice(N, size=n_perturb, replace=False)
                    current_step[idxs] = -current_step[idxs]
                    current_c3 = _compute_c3(current_step)
                    if current_c3 < best_c3:
                        best_c3 = current_c3
                        best_step = current_step.copy()
    
    return best_step, best_c3

def _greedy_polish(step, c3, time_budget=20.0):
    """
    Final greedy polish pass: try all single flips and keep the best improvement.
    Repeat until no improvement or time budget exhausted.
    """
    current_step = step.copy()
    current_c3 = c3
    start = time.time()
    
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        for i in range(N):
            if time.time() - start >= time_budget:
                break
            old_val = current_step[i]
            current_step[i] = -old_val
            new_c3 = _compute_c3(current_step)
            if new_c3 < current_c3 - 1e-10:
                current_c3 = new_c3
                improved = True
            else:
                current_step[i] = old_val
    
    return current_step, current_c3

def construct():
    start_time = time.time()
    
    # Phase 1: Structured initialization
    best_step, best_c3 = _structured_initialization()
    
    # Phase 2: Simulated annealing
    sa_time_budget = DEADLINE - 35.0  # Leave time for polish and safety
    if time.time() - start_time < DEADLINE - 35.0:
        sa_start = time.time()
        best_step, best_c3 = _simulated_annealing(
            best_step, best_c3, sa_start, 
            deadline_margin=sa_time_budget
        )
    
    # Phase 3: Greedy polish
    if time.time() - start_time < DEADLINE - 10.0:
        polish_budget = min(20.0, DEADLINE - 10.0 - (time.time() - start_time))
        if polish_budget > 1.0:
            best_step, best_c3 = _greedy_polish(best_step, best_c3, time_budget=polish_budget)
    
    # Safety check
    if time.time() - start_time > DEADLINE - 2.0:
        # Return immediately
        pass
    
    return best_step
