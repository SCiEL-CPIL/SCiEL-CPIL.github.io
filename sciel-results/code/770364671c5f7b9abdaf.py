"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_autocorr(h):
    """Compute linear autocorrelation R(k) for k=0..N-1 using FFT."""
    h_f = h.astype(float)
    H = np.fft.fft(h_f, n=2*N)
    power = np.real(H * np.conj(H))
    r = np.fft.ifft(power).real[:N]
    return r


def _score_from_autocorr(r):
    """Score: max|R(k)| for k>0 / N^2."""
    if N <= 1:
        return 1.0
    return np.max(np.abs(r[1:])) / (N * N)


def _energy_from_autocorr(r):
    """Sum of squared sidelobes."""
    if N <= 1:
        return 0.0
    return np.sum(r[1:] ** 2)


def _autocorr_flip_delta(h, idx, delta):
    """
    Compute the change in autocorrelation vector when h[idx] changes by delta.
    delta = new_val - old_val = -2*old_val for a flip.
    Returns vector of length N.
    """
    # For lag k:
    #   term1: h[idx+k] if idx+k < N (i.e., k < N-idx)
    #   term2: h[idx-k] if idx-k >= 0 (i.e., k <= idx)
    
    # Precompute the two parts
    # Part 1: for k = 0, 1, ..., N-1-idx, contribution is h[idx+k]
    # Part 2: for k = 0, 1, ..., idx, contribution is h[idx-k]
    
    dr = np.zeros(N)
    
    # Part 1: k from 0 to N-1-idx
    if N - idx > 0:
        dr[:N-idx] += h[idx:idx + (N-idx)]
    
    # Part 2: k from 0 to idx
    if idx >= 0:
        dr[:idx+1] += h[idx::-1]
    
    return delta * dr


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Parallel Tempering setup
    n_replicas = 6
    h_test = np.random.choice([-1, 1], size=N)
    r_test = _compute_autocorr(h_test)
    E_scale = _energy_from_autocorr(r_test)
    if E_scale < 1:
        E_scale = 1.0
    
    temperatures = np.geomspace(E_scale * 0.05, E_scale * 5, n_replicas)
    
    # Initialize replicas
    replicas_h = [np.ones(N, dtype=np.int8)]
    replicas_r = [_compute_autocorr(replicas_h[0].astype(float))]
    for i in range(n_replicas - 1):
        h = np.random.choice([-1, 1], size=N).astype(np.int8)
        replicas_h.append(h)
        replicas_r.append(_compute_autocorr(h.astype(float)))
    
    # Track best
    best_score = 1e10
    best_h = None
    for i in range(n_replicas):
        s = _score_from_autocorr(replicas_r[i])
        if s < best_score:
            best_score = s
            best_h = replicas_h[i].copy()
    
    # Precompute flip deltas for each position (for speed)
    # Actually, the delta depends on h, so we can't precompute fully.
    # But we can speed up the autocorr update.
    
    n_max_steps = 1000000
    step_count = 0
    swap_interval = 10
    
    while step_count < n_max_steps:
        if time.time() > deadline:
            break
        
        for rep_idx in range(n_replicas):
            if time.time() > deadline:
                break
            
            h = replicas_h[rep_idx]
            r = replicas_r[rep_idx]
            T = temperatures[rep_idx]
            
            idx = np.random.randint(N)
            old_val = h[idx]
            delta = -2.0 * old_val
            
            dr = _autocorr_flip_delta(h, idx, delta)
            new_r = r + dr
            
            E_old = _energy_from_autocorr(r)
            E_new = _energy_from_autocorr(new_r)
            dE = E_new - E_old
            
            if dE < 0 or np.random.random() < np.exp(-dE / T):
                h[idx] = -old_val
                replicas_r[rep_idx] = new_r
                if rep_idx == n_replicas - 1:
                    s = _score_from_autocorr(new_r)
                    if s < best_score:
                        best_score = s
                        best_h = h.copy()
            
            step_count += 1
        
        if step_count % swap_interval == 0:
            for i in range(n_replicas - 1):
                if time.time() > deadline:
                    break
                E_i = _energy_from_autocorr(replicas_r[i])
                E_ip1 = _energy_from_autocorr(replicas_r[i + 1])
                T_i = temperatures[i]
                T_ip1 = temperatures[i + 1]
                
                log_p = -E_i / T_i - E_ip1 / T_ip1 + E_i / T_ip1 + E_ip1 / T_i
                if log_p > 0 or np.random.random() < np.exp(min(log_p, 0)):
                    replicas_h[i], replicas_h[i + 1] = replicas_h[i + 1], replicas_h[i]
                    replicas_r[i], replicas_r[i + 1] = replicas_r[i + 1], replicas_r[i]
    
    # Final polish
    if best_h is not None and time.time() < deadline - 5.0:
        h_polish = best_h.copy()
        r_polish = _compute_autocorr(h_polish.astype(float))
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            E_best = _energy_from_autocorr(r_polish)
            for idx in range(N):
                if time.time() > deadline - 1.0:
                    break
                old_val = h_polish[idx]
                delta = -2.0 * old_val
                dr = _autocorr_flip_delta(h_polish, idx, delta)
                new_r = r_polish + dr
                E_new = _energy_from_autocorr(new_r)
                if E_new < E_best:
                    h_polish[idx] = -old_val
                    r_polish = new_r
                    E_best = E_new
                    improved = True
        
        s = _score_from_autocorr(r_polish)
        if s < best_score:
            best_score = s
            best_h = h_polish.copy()
    
    if best_h is None:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END

