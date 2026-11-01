import numpy as np
import time
from scipy.fft import fft, ifft

N = 400

def _autoconv_peak_fft(h):
    """Compute the peak of the linear autoconvolution of h using FFT."""
    # Pad to avoid circular convolution
    n = len(h)
    size = 1 << (2 * n - 1).bit_length()
    H = fft(h, n=size)
    conv = np.real(ifft(H * H, n=size))[:2 * n - 1]
    return np.max(np.abs(conv))

def _delta_peak_fft(h, idx):
    """
    Compute change in peak autoconvolution when flipping h[idx].
    Uses the fact that flipping h[idx] changes conv by -2*h[idx]*(h shifted).
    Returns (new_peak, old_peak) for efficiency.
    """
    # Current conv
    n = len(h)
    size = 1 << (2 * n - 1).bit_length()
    H = fft(h, n=size)
    conv = np.real(ifft(H * H, n=size))[:2 * n - 1]
    old_peak = np.max(np.abs(conv))
    
    # Flipping h[idx] means h_new = h - 2*h[idx]*e_idx
    # conv_new = conv - 2*h[idx]*(h*e_idx + e_idx*h) + 4*h[idx]^2*(e_idx*e_idx)
    # The cross terms: h*e_idx is h shifted by idx, e_idx*h is h reversed and shifted
    # This is complex; simpler to just recompute
    
    h_new = h.copy()
    h_new[idx] = -h[idx]
    H_new = fft(h_new, n=size)
    conv_new = np.real(ifft(H_new * H_new, n=size))[:2 * n - 1]
    new_peak = np.max(np.abs(conv_new))
    
    return new_peak, old_peak

def _gray_code_neighbor(h, idx):
    """Flip one bit in the sign pattern."""
    h_new = h.copy()
    h_new[idx] = -h[idx]
    return h_new

def _generate_initial_h(pattern_type, rng):
    """Generate initial sign patterns."""
    if pattern_type == 'random':
        return rng.choice([-1, 1], size=N)
    elif pattern_type == 'alternating':
        return np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    elif pattern_type == 'blocks':
        block_sizes = rng.choice([5, 10, 20, 40], size=8)
        signs = rng.choice([-1, 1], size=len(block_sizes))
        h = np.concatenate([np.full(s, sign) for s, sign in zip(block_sizes, signs)])
        return h[:N]
    elif pattern_type == 'sine':
        t = np.linspace(0, 4 * np.pi, N)
        return np.sign(np.sin(t + rng.uniform(0, 2 * np.pi)))
    else:
        return np.ones(N)

def _parallel_tempering_search(h_init, deadline, rng):
    """
    Parallel tempering (replica exchange) simulated annealing.
    Multiple chains at different temperatures explore the landscape.
    """
    n_replicas = 8
    # Temperatures: geometric progression from low to high
    T_base = 0.1
    T_ratio = 2.0
    temperatures = [T_base * (T_ratio ** i) for i in range(n_replicas)]
    
    # Initialize replicas with different initial patterns
    patterns = ['random', 'alternating', 'blocks', 'sine', 'random', 'blocks', 'sine', 'random']
    replicas = []
    for i in range(n_replicas):
        h = _generate_initial_h(patterns[i], rng)
        score = _autoconv_peak_fft(h)
        replicas.append({'h': h, 'score': score, 'T': temperatures[i]})
    
    best_h = replicas[0]['h'].copy()
    best_score = replicas[0]['score']
    for r in replicas:
        if r['score'] < best_score:
            best_score = r['score']
            best_h = r['h'].copy()
    
    exchange_interval = 100  # Every 100 steps, try to exchange replicas
    step_count = 0
    no_improve_steps = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve_steps < max_no_improve:
        # Pick a random replica to update
        ri = rng.randint(n_replicas)
        r = replicas[ri]
        T = r['T']
        
        # Pick a random index to flip (Gray code neighbor)
        idx = rng.randint(N)
        h_new = _gray_code_neighbor(r['h'], idx)
        score_new = _autoconv_peak_fft(h_new)
        
        # Metropolis acceptance
        delta = score_new - r['score']
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            r['h'] = h_new
            r['score'] = score_new
            
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        
        step_count += 1
        
        # Replica exchange
        if step_count % exchange_interval == 0:
            # Sort replicas by score and try to exchange adjacent ones
            # Use Metropolis criterion for exchange based on energy difference and temperature
            for i in range(n_replicas - 1):
                if not time.time() < deadline:
                    break
                r1, r2 = replicas[i], replicas[i + 1]
                T1, T2 = r1['T'], r2['T']
                s1, s2 = r1['score'], r2['score']
                
                # Exchange probability: exp(-(s1/T2 + s2/T1 - s1/T1 - s2/T2))
                exponent = -(s1 / T2 + s2 / T1 - s1 / T1 - s2 / T2)
                if exponent > 0 or rng.random() < np.exp(min(exponent, 700)):
                    # Swap h and scores
                    r1['h'], r2['h'] = r2['h'], r1['h']
                    r1['score'], r2['score'] = r2['score'], r1['score']
        
        # Anneal: slowly decrease temperatures
        if step_count % 10000 == 0:
            for i in range(n_replicas):
                replicas[i]['T'] *= 0.95
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.RandomState(42)
    
    # Start with a reasonable initial guess
    h_init = _generate_initial_h('alternating', rng)
    initial_score = _autoconv_peak_fft(h_init)
    
    best_h = h_init.copy()
    best_score = initial_score
    
    # Run parallel tempering search
    try:
        best_h_pt, best_score_pt = _parallel_tempering_search(h_init, deadline, rng)
        if best_score_pt < best_score:
            best_score = best_score_pt
            best_h = best_h_pt
    except Exception:
        pass
    
    # Final polish: try all single flips on the best solution
    if time.time() < deadline:
        h = best_h.copy()
        current_score = best_score
        for idx in range(N):
            if time.time() >= deadline:
                break
            old_val = h[idx]
            h[idx] = -old_val
            new_score = _autoconv_peak_fft(h)
            if new_score < current_score:
                current_score = new_score
            else:
                h[idx] = old_val
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
    
    return best_h
