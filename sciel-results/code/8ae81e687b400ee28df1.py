import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2

def _ac_full(h):
    """Full aperiodic autocorrelation."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac

def _peak_sidelobe(h):
    ac = _ac_full(h)
    return float(np.max(np.abs(ac[1:])))

def _flip_ac_fast(ac, h, i):
    """Vectorised O(N) update of ac after flipping h[i]."""
    new_ac = ac.copy()
    hi = h[i]
    k_hi = N - 1 - i
    if k_hi >= 0:
        new_ac[0:k_hi + 1] += -2.0 * hi * h[i:i + k_hi + 1]
    k_lo = i
    if k_lo >= 1:
        new_ac[1:k_lo + 1] += -2.0 * hi * h[i - k_lo:i]
    return new_ac

def _continuous_relax_energy(h0, time_budget, start):
    """
    Minimize the L2 norm of the autocorrelation sidelobes (energy)
    using gradient descent on the continuous relaxation.
    Energy E = sum_{k>=1} ac[k]^2 = (1/N) sum_{m} |FFT(h)[m]|^4.
    Gradient dE/dh = (2/N) IFFT( FFT(h) * conj(FFT(h))^2 )? 
    Let's derive:
    L = sum_m H_m^2 \bar{H}_m^2
    dL/dH_m = 2 H_m \bar{H}_m^2
    dL/dh = F^* (2 H \bar{H}^2)
    So grad = (2/N) IFFT( H * conj(H)^2 )? No, H is complex.
    Let's use real operations or just trust the complex gradient.
    """
    h = h0.astype(np.float64).copy()
    best_peak = _peak_sidelobe(h)
    best_h = h.copy()
    
    step = 0.1
    while time.time() - start < time_budget:
        H = np.fft.fft(h, _FFT_SIZE)
        # Gradient of sum |H|^4
        # d/dh sum |H|^4 = 2 * IFFT( H * conj(H)^2 ) / N ?
        # Let's check scaling.
        # ac = IFFT(|H|^2). E = sum ac^2.
        # Using Parseval: sum ac^2 = (1/N) sum |FFT(ac)|^2 = (1/N) sum |H|^4.
        # So we minimize (1/N) sum |H|^4.
        # Gradient wrt h:
        # d/dh (1/N) sum_m H_m^2 \bar{H}_m^2
        # = (1/N) F^* ( 2 H_m \bar{H}_m^2 )
        
        grad_complex = 2.0 * H * (np.conj(H) ** 2)
        grad = np.fft.ifft(grad_complex, _FFT_SIZE).real[:N]
        grad /= N # scaling factor
        
        # Normalize gradient to control step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-8:
            # Adaptive step size
            step_adaptive = step * (1.0 / (1.0 + grad_norm * 0.1))
        else:
            step_adaptive = step
            
        h = h - step_adaptive * grad
        h = np.clip(h, -1.0, 1.0)
        
        # Check peak periodically
        if int((time.time() - start) * 10) % 10 == 0:
            peak = _peak_sidelobe(h)
            if peak < best_peak:
                best_peak = peak
                best_h = h.copy()
                
        step *= 0.999
        
    hr = np.where(best_h >= 0.0, 1.0, -1.0)
    return hr, _peak_sidelobe(hr)

def _greedy_polish_peak(h, time_budget, start):
    """
    Greedy single-flip polish minimizing the peak sidelobe.
    """
    h = h.copy()
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        # Sweep all indices
        for i in range(N):
            ac2 = _flip_ac_fast(ac, h, i)
            c = float(np.max(np.abs(ac2[1:])))
            if c < best_c:
                best_c = c
                best_i = i
        
        if best_i >= 0:
            h[best_i] *= -1.0
            ac = _flip_ac_fast(ac, h, best_i)
            cur = best_c
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            break
            
    return best_h, best

def _anneal(seed, time_budget, init_h=None):
    rng = np.random.default_rng(seed)
    if init_h is not None:
        h = init_h.copy()
    else:
        h = rng.choice([-1.0, 1.0], size=N)
        
    ac = _ac_full(h)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T = 4.0
    T_min = 1e-7
    start = time.time()
    
    while time.time() - start < time_budget:
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        T = max(T_min, T * (1.0 - 0.001 * progress))
        
        i = int(rng.integers(0, N))
        
        ac_new = _flip_ac_fast(ac, h, i)
        new = float(np.max(np.abs(ac_new[1:])))
        
        if new < cur:
            h[i] *= -1.0
            ac = ac_new
            cur = new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            # Accept worsening moves
            if rng.random() < np.exp(-(new - cur) / max(T, 1e-9)):
                h[i] *= -1.0
                ac = ac_new
                cur = new
                
    return best_h, best

def construct():
    try:
        hard_deadline = 140.0
        start = time.time()

        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        def _remaining():
            return hard_deadline - (time.time() - start)

        # Phase 1: Continuous relaxation from random and structured inits
        # Try a few different inits
        inits = []
        # Random
        rng = np.random.default_rng(42)
        inits.append(rng.choice([-1.0, 1.0], size=N))
        # Sine
        x = np.linspace(0, 2 * np.pi * 5, N)
        inits.append(np.sign(np.sin(x)))
        # Sine 2
        x2 = np.linspace(0, 2 * np.pi * 10, N)
        inits.append(np.sign(np.sin(x2 + 0.5)))
        
        for h_init in inits:
            if _remaining() < 25.0: break
            budget = min(10.0, _remaining() - 20.0)
            if budget > 2.0:
                h, c = _continuous_relax_energy(h_init, budget, time.time())
                if c < global_best:
                    global_best = c
                    global_best_h = h
                    # Quick polish after relaxation
                    sub = min(5.0, _remaining() - 20.0)
                    if sub > 1.0:
                        h2, c2 = _greedy_polish_peak(h, sub, time.time())
                        if c2 < global_best:
                            global_best = c2
                            global_best_h = h2

        # Phase 2: Greedy polish on the best result so far
        if _remaining() > 20.0:
            budget = min(15.0, _remaining() - 15.0)
            if budget > 1.0:
                h, c = _greedy_polish_peak(global_best_h, budget, time.time())
                if c < global_best:
                    global_best = c
                    global_best_h = h

        # Phase 3: Simulated Annealing
        # Use the best sequence as start
        while _remaining() > 10.0:
            budget = min(15.0, _remaining() - 10.0)
            if budget < 2.0: break
            h, c = _anneal(12345, budget, init_h=global_best_h)
            if c < global_best:
                global_best = c
                global_best_h = h
            # Polish after anneal
            sub = min(5.0, _remaining() - 10.0)
            if sub > 1.0:
                h2, c2 = _greedy_polish_peak(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 4: Final high-precision polish
        remaining = _remaining() - 2.0
        if remaining > 1.0:
            h, c = _greedy_polish_peak(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
