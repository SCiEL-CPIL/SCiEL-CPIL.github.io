import numpy as np
import time

N = 400
DEADLINE = 118.0
M = 4 * N  # fine grid size = 1600
dx = 0.5 / M  # domain length is 0.5
REP = M // N  # repeat factor = 4


def _compute_c3(signed_step):
    """
    Compute the C3 upper bound: max |autoconv(t)| for the step function.
    Uses FFT on a fine grid for efficiency.
    """
    f_fine = np.repeat(signed_step.astype(np.float64), REP)
    fft_f = np.fft.rfft(f_fine, n=2 * M)
    autoconv = np.fft.irfft(fft_f * np.conj(fft_f), n=2 * M)
    autoconv *= dx
    return np.max(np.abs(autoconv))


def _vectorized_greedy_step(current_step, current_c3):
    """
    Evaluate all single-flip neighbors using vectorized FFT.
    Returns (new_step, new_c3, flipped_idx) if an improvement is found, else (None, None, None).
    """
    # Create all single-flip candidates
    # current_step is (N,) int8
    # We want a matrix (N, N) where row i is current_step with bit i flipped
    candidates = np.tile(current_step, (N, 1))
    candidates[np.arange(N), np.arange(N)] = -current_step
    
    # Expand to fine grid
    # candidates is (N, N). We need (N, M).
    candidates_fine = np.repeat(candidates, REP, axis=1).astype(np.float64)
    
    # Compute FFT for all candidates
    # rfft on last axis
    fft_cands = np.fft.rfft(candidates_fine, n=2 * M, axis=1)
    
    # Compute autoconv
    autoconv_cands = np.fft.irfft(fft_cands * np.conj(fft_cands), n=2 * M, axis=1)
    autoconv_cands *= dx
    
    # Compute score for each candidate
    scores = np.max(np.abs(autoconv_cands), axis=1)
    
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    
    if best_score < current_c3:
        new_step = candidates[best_idx].copy()
        return new_step, best_score, best_idx
    else:
        return None, None, None


def _run_ils(initial_step, deadline):
    """
    Iterated Local Search using vectorized greedy steps.
    """
    current_step = initial_step.copy().astype(np.int8)
    current_c3 = _compute_c3(current_step)
    
    best_step = current_step.copy()
    best_c3 = current_c3
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Initial Greedy Polish
    while time.time() < deadline - 10.0:
        new_step, new_c3, _ = _vectorized_greedy_step(current_step, current_c3)
        if new_step is None:
            break
        current_step = new_step
        current_c3 = new_c3
        if current_c3 < best_c3:
            best_step = current_step.copy()
            best_c3 = current_c3
            
    # Phase 2: ILS Loop
    while time.time() < deadline - 2.0:
        # Perturb
        n_flips = rng.randint(3, 8)
        flip_indices = rng.choice(N, size=n_flips, replace=False)
        perturbed_step = current_step.copy()
        perturbed_step[flip_indices] = -perturbed_step[flip_indices]
        
        # Local Search on perturbed step
        p_c3 = _compute_c3(perturbed_step)
        
        # Greedy polish the perturbed step
        while time.time() < deadline - 2.0:
            new_step, new_c3, _ = _vectorized_greedy_step(perturbed_step, p_c3)
            if new_step is None:
                break
            perturbed_step = new_step
            p_c3 = new_c3
            
        # Accept if better
        if p_c3 < best_c3:
            best_step = perturbed_step.copy()
            best_c3 = p_c3
            current_step = perturbed_step
            current_c3 = p_c3
        else:
            # Reset to best known solution for next perturbation
            current_step = best_step
            current_c3 = best_c3
            
    return best_step, best_c3


def construct():
    start_time = time.time()
    
    # Generate diverse initializations
    inits = []
    # 1. All ones
    inits.append(np.ones(N, dtype=np.int8))
    # 2. Alternating
    alt = np.ones(N, dtype=np.int8)
    alt[1::2] = -1
    inits.append(alt)
    # 3. Random
    for seed in [42, 123, 456]:
        rng = np.random.RandomState(seed)
        inits.append(rng.choice([-1, 1], size=N).astype(np.int8))
        
    best_step = None
    best_c3 = float('inf')
    
    # Allocate time for each init
    time_per_init = (DEADLINE - 10.0) / len(inits)
    
    for i, init in enumerate(inits):
        if time.time() > DEADLINE - 5.0:
            break
            
        sa_deadline = time.time() + max(time_per_init, 5.0)
        if sa_deadline > DEADLINE - 5.0:
            sa_deadline = DEADLINE - 5.0
            
        step_sa, c3_sa = _run_ils(init, sa_deadline)
        
        if c3_sa < best_c3:
            best_step = step_sa
            best_c3 = c3_sa
            
    # Final polish on the absolute best
    if time.time() < DEADLINE - 1.0 and best_step is not None:
        # Do one more vectorized greedy pass on the best
        while time.time() < DEADLINE - 0.5:
            new_step, new_c3, _ = _vectorized_greedy_step(best_step, best_c3)
            if new_step is None:
                break
            best_step = new_step
            best_c3 = new_c3
            
    return best_step.astype(np.int8)
