"""Combined program: Chirp-based initialization with Fast Local Search."""
import numpy as np
import time
import random

N = 400
# FFT size must be at least 2*N - 1
_M = 1
while _M < 2 * N - 1:
    _M *= 2

# Precompute FFT basis vectors for fast spectrum updates
_FREQS = np.arange(_M // 2 + 1)
_BASIS = np.exp(-2j * np.pi * np.outer(_FREQS, np.arange(N)) / _M)

def _compute_autoconv_from_spectrum(H):
    ac = np.fft.irfft(H * H, _M)
    return ac[:2 * N - 1]

def _score_from_spectrum(H, s):
    if abs(s) < 1e-9:
        return 1e10
    ac = _compute_autoconv_from_spectrum(H)
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)

def _get_spectrum(h):
    return np.fft.rfft(h, _M)

def _chirp_sequence(x, mu):
    """Generate real part of a Zadoff-Chu-like chirp sequence."""
    # h[n] = exp(j * pi * mu * n^2 / N)
    # We take the real part to get signs, or just sign of real part.
    # Actually, for binary signs, we can just use sign(cos(...))
    # But let's generate the complex sequence and take sign of real part?
    # Or better: h[n] = sign(cos(pi * mu * n^2 / N))
    # Let's try h[n] = sign(sin(pi * mu * n^2 / N)) as well.
    # The directive says: "Represent the signed step sequence as the real part of a complex Zadoff-Chu sequence... optimizing the continuous chirp parameter".
    # So h[n] = sign(Re(exp(j * pi * mu * n^2 / N))) = sign(cos(pi * mu * n^2 / N))
    return np.sign(np.cos(np.pi * mu * x**2 / N))

def _optimize_chirp(x):
    """Find best chirp parameter mu via golden section search."""
    # Search range for mu.
    # mu in [0, 1] corresponds to one full cycle of the chirp phase over n in [0, N].
    # Actually phase is pi * mu * n^2 / N.
    # If mu=1, phase goes from 0 to pi*N.
    # Let's search mu in [0.1, 2.0]
    a, b = 0.1, 2.0
    gr = (np.sqrt(5) - 1) / 2  # Golden ratio
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    
    best_mu = 0.5
    best_score = np.inf
    
    # Evaluate initial points
    h_c = _chirp_sequence(x, c)
    h_c = np.sign(h_c)
    h_c[h_c == 0] = 1
    s_c = np.sum(h_c)
    H_c = _get_spectrum(h_c)
    score_c = _score_from_spectrum(H_c, s_c)
    
    h_d = _chirp_sequence(x, d)
    h_d = np.sign(h_d)
    h_d[h_d == 0] = 1
    s_d = np.sum(h_d)
    H_d = _get_spectrum(h_d)
    score_d = _score_from_spectrum(H_d, s_d)
    
    if score_c < best_score:
        best_score = score_c
        best_mu = c
    if score_d < best_score:
        best_score = score_d
        best_mu = d
        
    # Iterate
    t_start = time.time()
    while (b - a) > 1e-4 and time.time() - t_start < 2.0:
        if score_c < score_d:
            b = d
            d = c
            score_d = score_c
            c = b - gr * (b - a)
            h_c = _chirp_sequence(x, c)
            h_c = np.sign(h_c)
            h_c[h_c == 0] = 1
            s_c = np.sum(h_c)
            H_c = _get_spectrum(h_c)
            score_c = _score_from_spectrum(H_c, s_c)
            if score_c < best_score:
                best_score = score_c
                best_mu = c
        else:
            a = c
            c = d
            score_c = score_d
            d = a + gr * (b - a)
            h_d = _chirp_sequence(x, d)
            h_d = np.sign(h_d)
            h_d[h_d == 0] = 1
            s_d = np.sum(h_d)
            H_d = _get_spectrum(h_d)
            score_d = _score_from_spectrum(H_d, s_d)
            if score_d < best_score:
                best_score = score_d
                best_mu = d
                
    return best_mu, best_score

def _local_search(h, max_time=1.0):
    """Fast local search with single bit flips and small block flips."""
    s = np.sum(h)
    H = _get_spectrum(h)
    current_score = _score_from_spectrum(H, s)
    best_score = current_score
    best_h = h.copy()
    
    t_start = time.time()
    improved = True
    
    while improved and time.time() - t_start < max_time:
        improved = False
        # Try single bit flips
        for i in range(N):
            if time.time() - t_start > max_time:
                break
            h[i] = -h[i]
            # Update H and s
            # old_h[i] was -h[i] (current is new)
            # delta_h = 2 * new_h[i]
            delta_H = _BASIS[:, i] * (2.0 * h[i])
            H_trial = H + delta_H
            s_trial = s + 2.0 * h[i]
            
            if abs(s_trial) < 1e-9:
                h[i] = -h[i]
                continue
                
            score_trial = _score_from_spectrum(H_trial, s_trial)
            
            if score_trial < current_score:
                H = H_trial
                s = s_trial
                current_score = score_trial
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                h[i] = -h[i]
        
        # Try small block flips (length 2-5)
        for _ in range(100):
            if time.time() - t_start > max_time:
                break
            length = random.choice([2, 3, 4, 5])
            start = random.randint(0, N - length)
            
            # Save old values
            old_h = h[start:start+length].copy()
            
            # Flip
            h[start:start+length] = -h[start:start+length]
            
            # Update H and s
            # delta_h = -2 * old_h
            delta_H = _BASIS[:, start:start+length] @ (-2.0 * old_h)
            H_trial = H + delta_H
            s_trial = s - 4.0 * np.sum(old_h)
            
            if abs(s_trial) < 1e-9:
                h[start:start+length] = old_h
                continue
                
            score_trial = _score_from_spectrum(H_trial, s_trial)
            
            if score_trial < current_score:
                H = H_trial
                s = s_trial
                current_score = score_trial
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                h[start:start+length] = old_h
                
    return best_h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    x = np.arange(N)
    
    # 1. Optimize Chirp
    best_mu, best_chirp_score = _optimize_chirp(x)
    
    # Generate best chirp sequence
    best_h = _chirp_sequence(x, best_mu)
    best_h = np.sign(best_h)
    best_h[best_h == 0] = 1.0
    
    # 2. Local Search on Chirp
    best_h, best_score = _local_search(best_h, max_time=5.0)
    
    # 3. Generate more diverse initializations based on chirp variations and random
    inits = [best_h.copy()]
    
    # Add chirp with different mu values
    for mu in [0.2, 0.5, 0.8, 1.2, 1.5, 1.8]:
        h_c = _chirp_sequence(x, mu)
        h_c = np.sign(h_c)
        h_c[h_c == 0] = 1.0
        inits.append(h_c)
        
    # Add random signs
    rng = np.random.default_rng(42)
    for _ in range(5):
        inits.append(rng.choice([1.0, -1.0], N))
        
    # Add structured patterns
    inits.append(np.sign(np.sin(5 * np.pi * x / N)))
    inits.append(np.sign(np.cos(3 * np.pi * x / N)))
    
    # 4. Local Search on each initialization
    for h_init in inits:
        if time.time() > deadline - 10:
            break
            
        h_local, score_local = _local_search(h_init.copy(), max_time=2.0)
        
        if score_local < best_score:
            best_score = score_local
            best_h = h_local.copy()
            
    # 5. Final Polish
    if time.time() < deadline - 2:
        best_h, best_score = _local_search(best_h.copy(), max_time=5.0)
        
    return best_h
