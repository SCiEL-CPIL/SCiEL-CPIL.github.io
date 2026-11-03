import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0  # safety margin from 150s

def _autoconv_linear(h):
    """Compute linear autoconvolution of h."""
    n = len(h)
    size = 2 * n - 1
    f = np.fft.rfft(h, n=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)
    return ac

def _max_abs_ac_offpeak(ac):
    """Max absolute autoconvolution excluding k=0 (index N-1)."""
    ac_no_peak = ac.copy()
    ac_no_peak[N - 1] = 0.0
    return np.max(np.abs(ac_no_peak))

def _score(h):
    """Compute the ratio: max |C(k)| for k!=0 divided by (sum h)^2."""
    S = np.sum(h)
    if abs(S) < 1e-15:
        return np.inf
    ac = _autoconv_linear(h)
    max_abs = _max_abs_ac_offpeak(ac)
    return max_abs / (S * S)

def _smooth_obj(h, beta):
    """Smooth objective for gradient-based optimization."""
    S = np.sum(h)
    if abs(S) < 1e-15:
        return 1e10
    ac = _autoconv_linear(h)
    ac_no_peak = ac.copy()
    ac_no_peak[N - 1] = 0.0
    abs_ac = np.abs(ac_no_peak)
    if beta < 1e6:
        log_sum = np.logaddexp.reduce(np.log(abs_ac + 1e-30) * beta) / beta
    else:
        log_sum = np.max(abs_ac)
    return log_sum / (S * S)

def _local_search_flip(h, max_iters=2000):
    """Sign-flip local search to minimize max |C(k)| for k!=0."""
    h = h.copy().astype(float)
    S = np.sum(h)
    if abs(S) < 1e-15:
        return h
    h = h / S
    
    best_score = _score(h)
    for _ in range(max_iters):
        improved = False
        for i in range(N):
            h[i] = -h[i]
            S = np.sum(h)
            if abs(S) < 1e-15:
                h[i] = -h[i]
                continue
            ac = _autoconv_linear(h)
            max_abs = _max_abs_ac_offpeak(ac)
            new_score = max_abs / (S * S)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
    return h

def _local_search_block(h, max_iters=500):
    """Block sign-flip local search."""
    h = h.copy().astype(float)
    S = np.sum(h)
    if abs(S) < 1e-15:
        return h
    h = h / S
    
    best_score = _score(h)
    block_sizes = [1, 2, 4, 8, 16]
    
    for it in range(max_iters):
        improved = False
        for bs in block_sizes:
            for start in range(0, N - bs + 1, bs):
                h_old = h[start:start+bs].copy()
                h[start:start+bs] = -h_old
                S = np.sum(h)
                if abs(S) < 1e-15:
                    h[start:start+bs] = h_old
                    continue
                ac = _autoconv_linear(h)
                max_abs = _max_abs_ac_offpeak(ac)
                new_score = max_abs / (S * S)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    h[start:start+bs] = h_old
        if not improved:
            break
    return h

def _build_pulse_vector(pulses):
    """
    Build a length-N vector from a list of (center, width, sign) pulses.
    Pulses are disjoint rectangular blocks.
    """
    h = np.zeros(N)
    for center, width, sign in pulses:
        start = int(round(center - width / 2))
        end = start + int(round(width))
        start = max(0, start)
        end = min(N, end)
        if end > start:
            h[start:end] = sign
    return h

def _pulse_beam_search(max_pulses, beam_width, time_limit):
    """
    Beam search over pulse configurations (centers, widths, signs).
    Each pulse is a disjoint rectangular block.
    """
    start_time = time.time()
    
    # Widths to try: powers of 2 and some intermediate
    widths = [4, 8, 16, 24, 32, 48, 64, 96, 128]
    # Centers: spread across the domain
    center_step = 8
    centers = list(range(0, N, center_step))
    
    # Initial beam: single pulses
    beam = []
    for c in centers:
        for w in widths:
            if c - w//2 >= 0 and c + w//2 < N:
                beam.append([(c, w, 1.0)])
                beam.append([(c, w, -1.0)])
    
    # Score initial beam and keep top
    def score_pulses(pulses):
        h = _build_pulse_vector(pulses)
        S = np.sum(h)
        if abs(S) < 1e-15:
            return np.inf
        return _score(h)
    
    beam_with_scores = [(score_pulses(p), p) for p in beam]
    beam_with_scores.sort(key=lambda x: x[0])
    beam_with_scores = beam_with_scores[:beam_width]
    
    # Expand beam
    for depth in range(1, max_pulses):
        if time.time() - start_time > time_limit:
            break
        
        new_beam = []
        for _, pulses in beam_with_scores:
            # Try adding a new pulse at various positions
            existing_spans = []
            for c, w, s in pulses:
                existing_spans.append((c - w//2, c + w//2))
            
            # Find gaps where we can place new pulses
            for c in centers:
                for w in widths[:6]:  # limit widths for expansion
                    # Check overlap
                    overlap = False
                    for s1, e1 in existing_spans:
                        if c - w//2 < e1 and c + w//2 > s1:
                            overlap = True
                            break
                    if overlap:
                        continue
                    
                    for sign in [1.0, -1.0]:
                        new_pulses = pulses + [(c, w, sign)]
                        sc = score_pulses(new_pulses)
                        new_beam.append((sc, new_pulses))
            
            if time.time() - start_time > time_limit:
                break
        
        if not new_beam:
            break
        
        new_beam.sort(key=lambda x: x[0])
        beam_with_scores = new_beam[:beam_width]
    
    # Return best pulse configuration
    if not beam_with_scores:
        return None, np.inf
    
    best_score, best_pulses = beam_with_scores[0]
    return best_pulses, best_score

def _refine_pulse_continuous(pulses, time_limit):
    """
    Refine pulse centers and widths using continuous optimization.
    """
    start_time = time.time()
    
    # Convert pulses to parameter vector: [center1, width1, sign1, center2, width2, sign2, ...]
    params = []
    for c, w, s in pulses:
        params.extend([c, w, s])
    params = np.array(params, dtype=float)
    
    def obj(p):
        if time.time() - start_time > time_limit:
            return 1e10
        # Parse parameters
        new_pulses = []
        for i in range(len(p) // 3):
            c = p[3*i]
            w = max(1, p[3*i+1])
            s = 1.0 if p[3*i+2] >= 0 else -1.0
            new_pulses.append((c, w, s))
        h = _build_pulse_vector(new_pulses)
        S = np.sum(h)
        if abs(S) < 1e-15:
            return 1e10
        return _score(h)
    
    try:
        res = minimize(obj, params, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1.0, 'fatol': 1e-8})
        best_params = res.x
    except Exception:
        best_params = params
    
    # Convert back to pulses
    refined_pulses = []
    for i in range(len(best_params) // 3):
        c = int(round(best_params[3*i]))
        w = max(1, int(round(best_params[3*i+1])))
        s = 1.0 if best_params[3*i+2] >= 0 else -1.0
        c = max(0, min(N-1, c))
        w = max(1, min(N, w))
        refined_pulses.append((c, w, s))
    
    return refined_pulses

def construct():
    start_time = time.time()
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    np.random.seed(42)
    
    def elapsed():
        return time.time() - start_time
    
    # Phase 1: Pulse-based beam search (structural change)
    # Try different numbers of pulses
    for max_pulses in [2, 3, 4]:
        if elapsed() > DEADLINE - 40:
            break
        
        beam_width = 50
        time_for_beam = 15.0
        
        pulses, sc = _pulse_beam_search(max_pulses, beam_width, time_for_beam)
        if pulses is not None and sc < best_score:
            best_score = sc
            h_pulse = _build_pulse_vector(pulses)
            best_h = h_pulse.copy()
            
            # Refine continuously
            if elapsed() < DEADLINE - 20:
                refined = _refine_pulse_continuous(pulses, time_limit=10.0)
                h_refined = _build_pulse_vector(refined)
                sc_ref = _score(h_refined)
                if sc_ref < best_score:
                    best_score = sc_ref
                    best_h = h_refined.copy()
                
                # Polish with local search
                if elapsed() < DEADLINE - 10:
                    h_polished = _local_search_block(h_refined, max_iters=100)
                    sc_pol = _score(h_polished)
                    if sc_pol < best_score:
                        best_score = sc_pol
                        best_h = h_polished.copy()
    
    # Phase 2: Traditional initialization + L-BFGS
    inits = []
    
    # All ones
    inits.append(np.ones(N))
    
    # Top-hat of various widths
    for w in [50, 100, 150, 200]:
        h = np.zeros(N)
        h[:w] = 1.0
        inits.append(h)
    
    # Gaussian-like
    t = np.arange(N)
    for sigma in [50, 100, 150]:
        h = np.exp(-0.5 * ((t - N//2) / sigma) ** 2)
        inits.append(h)
    
    # Cosine window
    t = np.arange(N)
    h = np.cos(np.pi * t / N) ** 2
    inits.append(h)
    
    # Random smooth
    for _ in range(5):
        h = np.random.randn(N)
        h = np.convolve(h, np.ones(10)/10, mode='same')
        inits.append(h)
    
    # Random with bias
    for _ in range(3):
        h = np.random.uniform(-1, 1, size=N)
        h += 0.3
        inits.append(h)
    
    for h0 in inits:
        if elapsed() > DEADLINE - 30:
            break
        
        S = np.sum(h0)
        if abs(S) < 1e-15:
            h0 = h0 + 1.0
            S = np.sum(h0)
        h0 = h0 / S
        
        # L-BFGS optimization with smooth surrogate
        beta = 50.0
        try:
            res = minimize(
                _smooth_obj, h0, args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0
        
        S = np.sum(h_opt)
        if abs(S) < 1e-15:
            h_opt = h_opt + 1.0
            S = np.sum(h_opt)
        h_opt = h_opt / S
        
        sc = _score(h_opt)
        if sc < best_score:
            best_score = sc
            best_h = h_opt.copy()
        
        # Polish with local search
        if elapsed() > DEADLINE - 10:
            break
        
        h_polished = _local_search_block(h_opt, max_iters=200)
        sc = _score(h_polished)
        if sc < best_score:
            best_score = sc
            best_h = h_polished.copy()
    
    # Phase 3: More aggressive local search on best
    if elapsed() < DEADLINE - 5:
        h_final = _local_search_flip(best_h, max_iters=3000)
        sc = _score(h_final)
        if sc < best_score:
            best_score = sc
            best_h = h_final.copy()
    
    return best_h
