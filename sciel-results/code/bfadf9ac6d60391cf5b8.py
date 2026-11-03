"""Cellular-automaton ground-state search for signed step functions minimizing max|autoconv|/(sum h)^2."""
import numpy as np
import time

N = 400


def _autoconv(h):
    """Full linear autoconvolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]


def _score(h):
    """max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv(h)
    return np.max(np.abs(ac)) / (s * s)


def _delta_autoconv(h, i):
    """
    Compute the change in autoconvolution if h[i] is flipped (h[i] -> -h[i]).
    The autoconvolution ac[k] = sum_j h[j]*h[j+k].
    Flipping h[i] changes ac[k] by -2*h[i]*h[i+k] for k>=0 and -2*h[i]*h[i-k] for k<0.
    In the full linear convolution of length 2n-1, the contribution of h[i] to ac[k] 
    (where k ranges from -(n-1) to n-1) is:
      ac[k] includes h[i]*h[i+k] if 0<=i+k<n, and h[i+k]*h[i] if 0<=i-k<n.
    Actually for linear autoconvolution: ac[k] = sum_{j} h[j]*h[j+k] for k>=0
    and ac[k] = sum_{j} h[j+k]*h[j] for k<0 (same thing by symmetry).
    
    More precisely, the full linear convolution c = h * h has c[m] = sum_j h[j]*h[m-j] for m=0..2n-2.
    Flipping h[i] changes c[m] by -2*h[i]*h[m-i] for all m where both indices are valid.
    """
    n = len(h)
    size = 2 * n - 1
    delta = np.zeros(size)
    hi = h[i]
    # c[m] = sum_j h[j] * h[m-j]
    # Flipping h[i]: h[i] -> -h[i], so c[m] changes by -2*h[i]*h[m-i] 
    # for all m where 0 <= i < n and 0 <= m-i < n, i.e., i <= m <= i+n-1
    # Also, h[i] appears as the second factor: h[m-i]*h[i], same term.
    # Wait, in c[m] = sum_j h[j]*h[m-j], h[i] appears when j=i (first factor) giving h[i]*h[m-i],
    # and when m-j=i i.e. j=m-i (second factor) giving h[m-i]*h[i].
    # So if i != m-i, there are two contributions, each changing by -2*h[i]*h[m-i].
    # If i == m-i, i.e., m=2i, there's one contribution h[i]*h[i], changing by -2*h[i]*h[i] = -2*hi^2.
    # Actually let me be more careful. The change in c[m] is:
    # new_c[m] - old_c[m] = sum_j (h'[j]*h'[m-j] - h[j]*h[m-j])
    # where h'[i] = -h[i], h'[j] = h[j] for j != i.
    # Terms involving h[i]: 
    #   j=i: h'[i]*h'[m-i] = (-h[i])*h[m-i] if m-i != i, or (-h[i])*(-h[i]) if m-i==i
    #   j=m-i (if m-i != i): h[m-i]*h'[i] = h[m-i]*(-h[i])
    # So for m != 2i: change = -h[i]*h[m-i] - h[m-i]*h[i] = -2*h[i]*h[m-i]
    # For m == 2i: change = h[i]*h[i] - h[i]*h[i] = ... wait
    #   j=i: h'[i]*h'[i] = (-h[i])^2 = h[i]^2, old was h[i]^2, change = 0? No wait.
    #   Actually when m=2i, the term j=i gives h[i]*h[i] in old, and (-h[i])*(-h[i])=h[i]^2 in new. Change = 0.
    #   But j=m-i=i is the same term! So it's a single term, not two.
    #   For m=2i: only j=i contributes (since m-j=i means j=i). Change = h[i]^2 - h[i]^2 = 0? 
    #   No: old contribution is h[i]*h[i]=h[i]^2, new is (-h[i])*(-h[i])=h[i]^2. Change = 0.
    #   Hmm, that can't be right. Let me reconsider.
    #   Actually wait: c[m] = sum_j h[j]*h[m-j]. For m=2i, the term j=i is h[i]*h[i].
    #   After flip: h'[i]*h'[i] = (-h[i])^2 = h[i]^2. So no change at m=2i from this term.
    #   But are there other j where m-j = i? That's j = m-i = i. Same term.
    #   So for m=2i, change = 0.
    #   For m != 2i: two distinct terms j=i and j=m-i, each contributing -h[i]*h[m-i], total -2*h[i]*h[m-i].
    
    # So: delta[m] = -2*h[i]*h[m-i] for i <= m <= i+n-1, m != 2i
    #              delta[m] = 0 for m = 2i
    # But wait, I need to check: does h[m-i] exist? We need 0 <= m-i < n, i.e., i <= m <= i+n-1.
    
    lo = max(0, i)
    hi_idx = min(size - 1, i + n - 1)
    for m in range(lo, hi_idx + 1):
        if m - i < n:  # valid index for h[m-i]
            if m == 2 * i:
                # Single term, no change (as analyzed)
                pass
            else:
                delta[m] = -2.0 * hi * h[m - i]
    
    # Hmm wait, I think I made an error. Let me re-examine.
    # When we flip h[i], the new autoconvolution is c'[m] = sum_j h'[j]*h'[m-j]
    # The difference c'[m] - c[m] comes only from terms where j=i or m-j=i (i.e., j=m-i).
    # Case 1: j=i. Term changes from h[i]*h[m-i] to (-h[i])*h[m-i] (if m-i != i) or (-h[i])*(-h[i]) (if m-i==i).
    #   If m-i != i: change = -h[i]*h[m-i] - h[i]*h[m-i] = -2*h[i]*h[m-i]? No!
    #   Old term: h[i]*h[m-i]. New term: (-h[i])*h[m-i] = -h[i]*h[m-i]. Change = -2*h[i]*h[m-i].
    #   If m-i == i: old term h[i]*h[i], new term (-h[i])^2 = h[i]^2. Change = 0.
    # Case 2: j=m-i (and m-i != i, so distinct from case 1). Term changes from h[m-i]*h[i] to h[m-i]*(-h[i]).
    #   Change = -h[m-i]*h[i] - h[m-i]*h[i] = -2*h[i]*h[m-i].
    # So total for m != 2i: -2*h[i]*h[m-i] - 2*h[i]*h[m-i] = -4*h[i]*h[m-i]? 
    # No wait, I'm double counting. Let me be very careful.
    
    # c[m] = sum_{j=0}^{n-1} h[j] * h[m-j] (where h[k]=0 if k outside [0,n-1])
    # The terms involving h[i] are:
    #   j=i: h[i]*h[m-i] (valid if 0<=m-i<n)
    #   j=m-i: h[m-i]*h[i] (valid if 0<=m-i<n, i.e., same condition, and m-i != i)
    # After flip, h'[i]=-h[i]:
    #   j=i: (-h[i])*h[m-i]
    #   j=m-i: h[m-i]*(-h[i])
    # So new sum of these terms: -h[i]*h[m-i] - h[m-i]*h[i] = -2*h[i]*h[m-i] (if m!=2i)
    # Old sum: h[i]*h[m-i] + h[m-i]*h[i] = 2*h[i]*h[m-i] (if m!=2i)
    # Change: -4*h[i]*h[m-i]
    # If m=2i: only j=i term (since m-i=i). Old: h[i]^2. New: (-h[i])^2=h[i]^2. Change: 0.
    
    # So actually the change is -4*h[i]*h[m-i] for m != 2i, and 0 for m=2i.
    # Let me redo this.
    
    delta[:] = 0.0
    for m in range(max(0, i), min(size, i + n)):
        if m - i >= n:
            break
        if m == 2 * i:
            continue
        delta[m] = -4.0 * hi * h[m - i]
    
    return delta


def _flip_and_update(h, i, ac):
    """Flip h[i] and update autoconvolution incrementally. Returns new h, new ac."""
    n = len(h)
    size = 2 * n - 1
    delta = _delta_autoconv(h, i)
    ac_new = ac + delta
    h_new = h.copy()
    h_new[i] = -h[i]
    return h_new, ac_new


def _greedy_refine(h, ac, deadline, max_rounds=200):
    """
    Greedy sign-flip refinement. In each round, try flipping each position and accept if it reduces the score.
    """
    n = len(h)
    size = 2 * n - 1
    s = np.sum(h)
    current_score = np.max(np.abs(ac)) / (s * s)
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        # Try random order
        rng = np.random.default_rng(round_idx * 1000 + 7)
        order = rng.permutation(n)
        for i in order:
            if time.time() > deadline:
                break
            h_new, ac_new = _flip_and_update(h, i, ac)
            s_new = s - 2 * h[i]
            if abs(s_new) < 1e-9:
                continue
            new_score = np.max(np.abs(ac_new)) / (s_new * s_new)
            if new_score < current_score:
                h = h_new
                ac = ac_new
                s = s_new
                current_score = new_score
                improved = True
        if not improved:
            break
    
    return h, ac, current_score


def _simulated_annealing(h, ac, deadline, iters_per_phase=50, n_phases=10):
    """
    Simulated annealing with efficient delta computation.
    """
    n = len(h)
    size = 2 * n - 1
    s = np.sum(h)
    current_score = np.max(np.abs(ac)) / (s * s)
    best_score = current_score
    best_h = h.copy()
    best_ac = ac.copy()
    best_s = s
    
    rng = np.random.default_rng(12345)
    
    for phase in range(n_phases):
        if time.time() > deadline:
            break
        beta = 0.5 * (phase + 1)
        for it in range(iters_per_phase):
            if time.time() > deadline:
                break
            i = rng.integers(0, n)
            h_new, ac_new = _flip_and_update(h, i, ac)
            s_new = s - 2 * h[i]
            if abs(s_new) < 1e-9:
                continue
            new_score = np.max(np.abs(ac_new)) / (s_new * s_new)
            delta = new_score - current_score
            if delta < 0 or rng.random() < np.exp(-beta * delta / max(1e-6, abs(current_score))):
                h = h_new
                ac = ac_new
                s = s_new
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
                    best_ac = ac.copy()
                    best_s = s
    
    return best_h, best_ac, best_score


def _gen_initial(n, seed):
    """Generate diverse initial patterns."""
    rng = np.random.default_rng(seed)
    patterns = []
    
    # Random signs
    patterns.append(rng.choice([-1.0, 1.0], size=n))
    
    # Chirp
    i = np.arange(n)
    patterns.append(np.sign(np.cos(np.pi * i**2 / (n * 0.5))))
    
    # Alternating blocks
    h = np.zeros(n)
    block = n // 8
    for b in range(8):
        h[b*block:(b+1)*block] = 1.0 if b % 2 == 0 else -1.0
    patterns.append(h)
    
    # Random smoothed then signed
    noise = rng.standard_normal(n)
    kernel = np.ones(5) / 5
    smoothed = np.convolve(noise, kernel, mode='same')
    patterns.append(np.sign(smoothed))
    
    # Sine wave
    patterns.append(np.sign(np.sin(2 * np.pi * i / n * 3)))
    
    # Two-tone
    patterns.append(np.sign(np.sin(2 * np.pi * i / n * 7) + 0.5 * np.sin(2 * np.pi * i / n * 13)))
    
    return patterns


def construct():
    t_start = time.time()
    deadline = t_start + 110.0
    
    best_score = np.inf
    best_h = np.ones(N)
    
    # Baseline
    ac_base = _autoconv(best_h)
    best_score = _score(best_h)
    
    # Generate initial patterns
    inits = _gen_initial(N, seed=42)
    
    for idx, h_init in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        # Ensure sum is not zero
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            # Flip a few signs to make sum nonzero
            h_init = h_init.copy()
            h_init[0] = 1.0
            s0 = np.sum(h_init)
            if abs(s0) < 1e-6:
                continue
        
        # Normalize to unit sum for consistent comparison
        h_norm = h_init / s0
        ac_norm = _autoconv(h_norm)
        
        # Phase 1: Simulated annealing
        h_sa, ac_sa, score_sa = _simulated_annealing(h_norm, ac_norm, deadline - 5 - idx * 3, 
                                                      iters_per_phase=30, n_phases=8)
        
        # Phase 2: Greedy refinement
        h_greedy, ac_greedy, score_greedy = _greedy_refine(h_sa, ac_sa, deadline - 2 - idx * 2, max_rounds=100)
        
        if score_greedy < best_score:
            best_score = score_greedy
            best_h = h_greedy.copy()
    
    # Final refinement on best solution
    if time.time() < deadline - 2:
        ac_best = _autoconv(best_h)
        h_final, ac_final, score_final = _greedy_refine(best_h, ac_best, deadline - 1, max_rounds=50)
        if score_final < best_score:
            best_score = score_final
            best_h = h_final.copy()
    
    return best_h
